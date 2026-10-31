#!/usr/bin/env python3
"""
Starts TermsGuard AI. Run it with no arguments and it does everything needed:

    python3 run.py

It finds (or creates) the virtual environment, installs the dependencies the
first time, frees port 5001 if anything is holding it, starts the backend, waits
until the models are loaded, prints what to do in the browser, and shuts down
cleanly on Ctrl+C.

Useful flags:

    python3 run.py --smoke     start the backend, analyse a sample clause set, exit
    python3 run.py --test      run the regression checks instead of the server
    python3 run.py --eval      run the full evaluation against the UNFAIR-ToS test split
    python3 run.py --port 5002 serve on another port
    python3 run.py --setup     create the venv and install dependencies, then exit

Only the standard library is used here, so it runs before anything is installed.
"""
import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
VENV = BACKEND / "venv"
REQUIREMENTS = BACKEND / "requirements.txt"
PREFERRED_PYTHON = "3.11"   # the version the project is documented and tested against
DEFAULT_PORT = 5001

SAMPLE = (
    "We may modify these terms at any time without prior notice to you. "
    "By continuing to use the service you agree to be bound by the updated terms. "
    "You can delete your account and download a copy of your data at any time. "
    "We are not liable for any damages arising from a data breach. "
    "Any dispute shall be resolved by binding arbitration and you waive your right to a class action."
)


def say(message, symbol="•"):
    print(f"{symbol} {message}", flush=True)


def venv_python():
    binary = VENV / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
    return binary if binary.exists() else None


def find_interpreter():
    """A python3.11 if one is installed, otherwise whatever is running this script."""
    for name in (f"python{PREFERRED_PYTHON}", "python3"):
        found = shutil.which(name)
        if found:
            return found
    return sys.executable


def ensure_environment():
    """Create the venv and install requirements if they are missing. Returns the venv python."""
    python = venv_python()
    if python is None:
        interpreter = find_interpreter()
        version = subprocess.run([interpreter, "--version"], capture_output=True, text=True).stdout.strip()
        say(f"Creating virtual environment with {version} (one-off)...")
        subprocess.run([interpreter, "-m", "venv", str(VENV)], check=True)
        python = venv_python()
        if python is None:
            sys.exit("Could not create the virtual environment at " + str(VENV))

    try:
        subprocess.run([str(python), "-c", "import flask, faiss, torch, transformers, sentence_transformers"],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError:
        say("Installing dependencies (a few minutes the first time)...")
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--quiet", "-r", str(REQUIREMENTS)], check=True)
        say("Dependencies installed.", "✓")
    return python


def get_json(url, timeout=3):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url, payload, timeout=300):
    request = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    # macOS urllib would otherwise send a local request through the system proxy
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def holders_of(port):
    """PIDs listening on the port, newest first. POSIX only (lsof)."""
    try:
        found = subprocess.run(["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True).stdout
    except (FileNotFoundError, OSError):
        return []
    return [int(pid) for pid in found.split()]


def describe(pid):
    name = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout.strip()
    return (name[:70] + "...") if len(name) > 73 else (name or f"pid {pid}")


def free_port(port):
    """Stop whatever holds the port so this backend can have it."""
    holders = holders_of(port)
    if not holders:
        return
    for pid in holders:
        say(f"Port {port} is held by {describe(pid)} - stopping it.")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 8
    while time.time() < deadline and holders_of(port):
        time.sleep(0.5)

    for pid in holders_of(port):   # ignored SIGTERM, so insist
        say(f"Forcing {describe(pid)} to stop.")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.5)
    if holders_of(port):
        sys.exit(f"Could not free port {port}. Stop that process yourself and run this again.")
    say(f"Port {port} is free.", "✓")


def backend_is_up(port):
    try:
        return get_json(f"http://127.0.0.1:{port}/health").get("status") == "running"
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False


def start_backend(python, port):
    """Start app.py and wait for the models to load, taking over the port if something holds it."""
    free_port(port)

    environment = dict(os.environ, FLASK_RUN_PORT=str(port), TERMSGUARD_PORT=str(port))
    say("Starting the backend. First run downloads ~2GB of models; after that startup is about 40 seconds.")
    process = subprocess.Popen([str(python), "app.py"], cwd=str(BACKEND), env=environment)

    started = time.time()
    while True:
        if process.poll() is not None:
            sys.exit(f"The backend exited with code {process.returncode} before it was ready.")
        if backend_is_up(port):
            say(f"Backend ready in {time.time() - started:.0f}s on http://127.0.0.1:{port}", "✓")
            return process
        if time.time() - started > 1800:
            process.terminate()
            sys.exit("The backend did not become ready within 30 minutes.")
        time.sleep(2)


def print_options(port):
    try:
        options = get_json(f"http://127.0.0.1:{port}/options")
    except Exception:
        return
    print()
    say("Pipeline options (choose these in the extension popup):")
    for kind, label in (("rag", "Retrieval"), ("classifier", "Verdict"), ("generator", "Summary model")):
        for option in options.get(kind, []):
            if option.get("default"):
                print(f"    {label:14} default: {option['name']} - {option['why']}")
            elif not option.get("available"):
                print(f"    {label:14} unavailable: {option['name']} (run backend/train_legalbert.py)")


def smoke_test(port):
    say("Running a sample analysis...")
    started = time.time()
    result = post_json(f"http://127.0.0.1:{port}/process", {"text": SAMPLE, "links": []})
    print(f"    {time.time() - started:.1f}s round trip, backend reported {result['processing_time_seconds']}s")
    print(f"    risk level: {result['risk_level']} | clauses analysed: {result['clauses_analyzed']}")
    for point in result["non_ethical_points"][:3]:
        category = f" [{point['category']}]" if point.get("category") else ""
        print(f"    unfair  {point['confidence']:.2f}{category}  {point['text'][:70]}")
    for line in result["summary"][:2]:
        print(f"    summary: {line}")
    return result


def browser_instructions():
    browser = "brave://extensions" if platform.system() == "Darwin" else "chrome://extensions"
    print()
    say("In your browser:")
    print(f"    1. Open {browser} (or chrome://extensions), switch Developer mode on.")
    print(f"    2. 'Load unpacked' -> select {ROOT / 'extension'}")
    print("       (already loaded? press Reload on the Terms Guard AI card instead)")
    print("    3. Open any Terms & Conditions page, reload it, click the Terms Guard AI icon,")
    print("       then 'Analyze This Page'.")


def run_script(python, script, extra=()):
    return subprocess.run([str(python), script, *extra], cwd=str(BACKEND)).returncode


def main():
    parser = argparse.ArgumentParser(description="Run TermsGuard AI.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--smoke", action="store_true", help="analyse a sample, then exit")
    parser.add_argument("--test", action="store_true", help="run the regression checks and exit")
    parser.add_argument("--eval", action="store_true", help="run the evaluation comparison and exit")
    parser.add_argument("--setup", action="store_true", help="install dependencies and exit")
    arguments = parser.parse_args()

    print("TermsGuard AI")
    python = ensure_environment()

    if arguments.setup:
        say("Environment ready.", "✓")
        return 0
    if arguments.test:
        return run_script(python, "test_classification.py")
    if arguments.eval:
        return run_script(python, "evaluate.py", ["compare"])

    process = start_backend(python, arguments.port)
    print_options(arguments.port)

    if arguments.smoke:
        smoke_test(arguments.port)
        if process:
            process.terminate()
            process.wait(timeout=30)
        say("Smoke test finished.", "✓")
        return 0

    browser_instructions()
    print()
    say("Backend is running. Press Ctrl+C to stop.", "✓")

    def shutdown(*_):
        say("Stopping the backend...")
        process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        process.wait()
    except KeyboardInterrupt:
        shutdown()
        process.wait()
    say("Stopped.", "✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
