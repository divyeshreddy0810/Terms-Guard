"""
Measures the report's Section IV metrics. Every number is printed and saved to
eval_results/ exactly as measured - nothing here is tuned towards a target.

  python evaluate.py classification [variant]  # accuracy, F1, recall@3, confidence bins (LexGLUE UNFAIR-ToS test set)
  python evaluate.py compare                 # every retrieval variant and Legal-BERT, side by side
  python evaluate.py privacy                 # privacy clauses (OPP-115 held-out policies) + negation minimal pairs
  python evaluate.py generation              # writes a blinded CSV of grounded vs ungrounded outputs for raters
  python evaluate.py score-generation A.csv [B.csv ...]   # ROUGE-L, BERTScore, ratings, hallucination rate
  python evaluate.py latency                 # /process wall-clock time for each data/pages/*.txt
  python evaluate.py memory                  # model weights, FAISS index size, peak process RSS

Unix only (memory uses the stdlib `resource` module).
"""
import csv
import json
import random
import resource
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "eval_results"

# Section IV confidence-score bins
BINS = [(0.90, 1.01, "0.90 - 1.00"), (0.80, 0.90, "0.80 - 0.89"), (0.70, 0.80, "0.70 - 0.79"),
        (0.60, 0.70, "0.60 - 0.69"), (-1.0, 0.60, "Below 0.60")]

READING_WPM = 238  # average adult silent reading rate for non-fiction (Brysbaert, 2019)


def save(name, result):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nsaved eval_results/{name}")


def precision_recall_f1(gold, pred, positive):
    tp = sum(g == positive and p == positive for g, p in zip(gold, pred))
    fp = sum(g != positive and p == positive for g, p in zip(gold, pred))
    fn = sum(g == positive and p != positive for g, p in zip(gold, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def measure(rows, rag, classifier):
    """Metrics for one (retrieval variant, classifier) pair on the given rows."""
    import app

    texts = [r["text"] for r in rows]
    gold = ["non_ethical" if r["labels"] else "ethical" for r in rows]
    started = time.perf_counter()
    predictions = app.classify_clauses(texts, rag=rag, classifier=classifier)
    seconds = time.perf_counter() - started
    # "unclear" is never correct against this dataset's two classes; count it as a miss.
    pred = [label for label, *_ in predictions]
    per_class = {c: precision_recall_f1(gold, pred, c) for c in ("ethical", "non_ethical")}
    # accuracy flatters a 9:1 split (always saying "fair" scores 0.893), so report
    # per-class averages as well
    positives = sum(g == "non_ethical" for g in gold)
    negatives = len(gold) - positives
    true_positive = sum(g == p == "non_ethical" for g, p in zip(gold, pred))
    true_negative = sum(g != "non_ethical" and p != "non_ethical" for g, p in zip(gold, pred))
    balanced = ((true_positive / positives if positives else 0) +
                (true_negative / negatives if negatives else 0)) / 2
    return predictions, gold, pred, {
        "rag": rag,
        "classifier": classifier,
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(rows), 4),
        "balanced_accuracy": round(balanced, 4),
        "majority_baseline_accuracy": round(max(positives, negatives) / len(gold), 4),
        "unclear": pred.count("unclear"),
        "per_class": per_class,
        "macro_f1": round(statistics.mean(m["f1"] for m in per_class.values()), 4),
        "seconds_per_1000_sentences": round(1000 * seconds / len(rows), 1),
    }


def privacy_rows():
    """Held-out privacy clauses as (text, gold label) rows, if the corpus was built."""
    path = DATA / "privacy_test.json"
    if not path.exists():
        return []
    held_out = json.loads(path.read_text())
    return ([{"text": c["text"], "labels": ["privacy"]} for c in held_out["harmful"]] +
            [{"text": c["text"], "labels": []} for c in held_out["protective"]])


def compare():
    """Side-by-side: every retrieval variant and verdict method, on both domains."""
    import app

    rows = app.load_unfair_tos("test")
    privacy = privacy_rows()
    runs = [(rag, "similarity") for rag in app.RAG_VARIANTS]
    if Path(app.LEGALBERT_DIR).is_dir():
        runs += [(app.DEFAULT_RAG, verdict) for verdict in ("legalbert", "legalbert-knn", "legalbert-privacy")]
    else:
        print("Legal-BERT not trained yet (run train_legalbert.py) - comparing retrieval variants only.\n")

    results = []
    for rag, classifier in runs:
        _, _, _, metrics = measure(rows, rag, classifier)
        if privacy:
            _, _, _, privacy_metrics = measure(privacy, rag, classifier)
            metrics["privacy_f1"] = privacy_metrics["per_class"]["non_ethical"]["f1"]
        results.append(metrics)
        print(f"{rag:8s} {classifier:18s} acc {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
              f"unfair-F1 {metrics['per_class']['non_ethical']['f1']:.3f}  "
              f"privacy-F1 {metrics.get('privacy_f1', float('nan')):.3f}", flush=True)

    header = (f"| {'Retrieval':10} | {'Verdict':18} | {'Accuracy':8} | {'Macro F1':8} | {'Unfair P':8} | "
              f"{'Unfair R':8} | {'Unfair F1':9} | {'Privacy F1':10} | {'Unclear':7} | {'s/1k':6} |")
    table = [header, "|" + "|".join("-" * (len(part) + 2) for part in header.split("|")[1:-1]) + "|"]
    for m in results:
        unfair = m["per_class"]["non_ethical"]
        privacy_f1 = f"{m['privacy_f1']:<10.3f}" if "privacy_f1" in m else f"{'-':<10}"
        table.append(f"| {m['rag']:10} | {m['classifier']:18} | {m['accuracy']:<8.3f} | {m['macro_f1']:<8.3f} | "
                     f"{unfair['precision']:<8.3f} | {unfair['recall']:<8.3f} | {unfair['f1']:<9.3f} | "
                     f"{privacy_f1} | {m['unclear']:<7} | {m['seconds_per_1000_sentences']:<6.1f} |")
    markdown = "\n".join(table)
    print("\n" + markdown)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "comparison.md").write_text(markdown + "\n")
    save("comparison.json", {"dataset": "LexGLUE UNFAIR-ToS test split", "sentences": len(rows), "runs": results})


def classification(rag=None, classifier=None):
    import app

    rag = rag or app.DEFAULT_RAG
    classifier = classifier or app.DEFAULT_CLASSIFIER
    rows = app.load_unfair_tos("test")
    predictions, gold, pred, metrics = measure(rows, rag, classifier)

    # Recall@3: for each unfair test sentence, does FAISS's top-3 include a reference
    # clause sharing one of its UNFAIR-ToS categories?
    unfair = [r for r in rows if r["labels"]]
    top3 = app.retrieve_references(app.embed([r["text"] for r in unfair]))
    hits = sum(any(set(r["labels"]) & app.CLAUSE_LABELS[clause] for clause, _, _ in refs)
               for r, refs in zip(unfair, top3))

    distribution = {}
    for cls in ("ethical", "non_ethical"):
        confidences = [conf for label, conf, *_ in predictions if label == cls]
        distribution[cls] = {name: round(100 * sum(lo <= c < hi for c in confidences) / max(len(confidences), 1), 1)
                             for lo, hi, name in BINS}

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "classification_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "gold_labels", "gold", "predicted", "confidence", "matched_clause", "predicted_category"])
        for r, g, (label, conf, clause, category) in zip(rows, gold, predictions):
            writer.writerow([r["text"], "; ".join(r["labels"]), g, label, conf, clause, category])

    save("classification.json", {
        "dataset": "LexGLUE UNFAIR-ToS test split",
        "sentences": len(rows),
        "gold_counts": {c: gold.count(c) for c in ("ethical", "non_ethical")},
        **metrics,
        "majority_class_baseline_accuracy": round(max(gold.count("ethical"), gold.count("non_ethical")) / len(rows), 4),
        "reference_corpus": {"source": "UNFAIR-ToS train split", "ethical": len(app.ETHICAL_CLAUSES),
                             "non_ethical": len(app.NON_ETHICAL_CLAUSES)},
        "retrieval_recall_at_3": {"unfair_sentences": len(unfair), "hits": hits,
                                  "recall": round(hits / len(unfair), 4)},
        "confidence_distribution_percent": distribution,
    })


# identical wording, opposite meaning: the negation test the embeddings fail on their own
MINIMAL_PAIRS = [
    ("We may sell your personal information to advertisers.", True),
    ("We do not sell your personal information to advertisers.", False),
    ("We may sell your personal information to third-party advertisers for marketing purposes.", True),
    ("We track your activity across other websites.", True),
    ("We never track your activity across other websites.", False),
    ("You cannot delete your account or your data.", True),
    ("You cannot delete your account once created.", True),
    ("You can delete your account and your data at any time.", False),
    ("We share your information with third parties for marketing.", True),
    ("We do not share your information with third parties for marketing.", False),
    ("We do not retain your data after you close your account.", False),
    ("We retain your personal data indefinitely after account closure.", True),
    # negation that is itself the unfair practice - these must NOT be flipped
    ("We may modify these terms at any time without notice.", True),
    ("We are not liable for any damages arising from a data breach.", True),
]


def privacy():
    """Privacy-practice detection, and whether the polarity check earns its place."""
    import app

    held_out = json.loads((DATA / "privacy_test.json").read_text())
    texts = [c["text"] for c in held_out["harmful"]] + [c["text"] for c in held_out["protective"]]
    gold = ["non_ethical"] * len(held_out["harmful"]) + ["ethical"] * len(held_out["protective"])
    contract_rows = app.load_unfair_tos("test")

    def measure_set(sentences, truth, rag="hybrid"):
        pred = [label for label, *_ in app.classify_clauses(sentences, rag=rag)]
        return precision_recall_f1(truth, pred, "non_ethical"), \
            round(sum(t == p for t, p in zip(truth, pred)) / len(truth), 4)

    result = {"privacy_test": held_out["source"], "privacy_sentences": len(texts),
              "contract_sentences": len(contract_rows), "polarity_check": {}}
    for state in (False, True):
        app.POLARITY_CHECK = state
        privacy_scores, privacy_accuracy = measure_set(texts, gold)
        contract_scores, _ = measure_set([r["text"] for r in contract_rows],
                                         ["non_ethical" if r["labels"] else "ethical" for r in contract_rows])
        pairs, pair_accuracy = measure_set([t for t, _ in MINIMAL_PAIRS],
                                           ["non_ethical" if u else "ethical" for _, u in MINIMAL_PAIRS])
        result["polarity_check"]["on" if state else "off"] = {
            "privacy": privacy_scores, "privacy_accuracy": privacy_accuracy,
            "contract": contract_scores, "minimal_pairs_accuracy": pair_accuracy}
    app.POLARITY_CHECK = True
    save("privacy.json", result)


def generation(n=30, seed=0):
    import app

    rows = app.load_unfair_tos("test")
    rng = random.Random(seed)
    sample = (rng.sample([r for r in rows if r["labels"]], n // 2) +
              rng.sample([r for r in rows if not r["labels"]], n - n // 2))
    rng.shuffle(sample)
    references = app.retrieve_references(app.embed([r["text"] for r in sample]))

    # Raters see outputs A/B in random order; which one is grounded lives in a separate key file.
    RESULTS.mkdir(exist_ok=True)
    key = {}
    with open(RESULTS / "generation_samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "clause", "output_a", "output_b", "reference_explanation",
                         "rating_a_1to5", "rating_b_1to5", "hallucinated_a_0or1", "hallucinated_b_0or1"])
        for i, (row, refs) in enumerate(zip(sample, references)):
            grounded = app.generate_warning(row["text"], refs)
            baseline = app.generate_warning(row["text"], [])
            a_is_grounded = rng.random() < 0.5
            a, b = (grounded, baseline) if a_is_grounded else (baseline, grounded)
            writer.writerow([i, row["text"], a, b, "", "", "", "", ""])
            key[i] = {"a_is_grounded": a_is_grounded, "gold_labels": row["labels"],
                      "retrieved": [clause for clause, _, _ in refs]}
            print(f"[{i + 1}/{n}]")
    (RESULTS / "generation_key.json").write_text(json.dumps(key, indent=2))
    print("\nwrote eval_results/generation_samples.csv (give a copy to each rater) and generation_key.json (don't).")
    print("Raters fill: reference_explanation (one plain-English sentence per clause, written without looking at the outputs),")
    print("rating_a/b (1-5) and hallucinated_a/b (1 if the output states something not in the clause).")


def rouge_l(candidate, reference):
    """ROUGE-L F-measure over lower-cased whitespace tokens."""
    c, r = candidate.lower().split(), reference.lower().split()
    if not c or not r:
        return 0.0
    lcs = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]
    for i, ct in enumerate(c):
        for j, rt in enumerate(r):
            lcs[i + 1][j + 1] = lcs[i][j] + 1 if ct == rt else max(lcs[i][j + 1], lcs[i + 1][j])
    length = lcs[-1][-1]
    if length == 0:
        return 0.0
    precision, recall = length / len(c), length / len(r)
    return 2 * precision * recall / (precision + recall)


def cohens_kappa(x, y):
    observed = sum(a == b for a, b in zip(x, y)) / len(x)
    expected = sum((x.count(v) / len(x)) * (y.count(v) / len(y)) for v in set(x) | set(y))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score_generation(paths):
    key = json.loads((RESULTS / "generation_key.json").read_text())
    raters = [list(csv.DictReader(open(p, newline=""))) for p in paths]

    outputs = {"grounded": {}, "baseline": {}}
    ratings = {"grounded": [], "baseline": []}
    hallucinated = {"grounded": [[] for _ in raters], "baseline": [[] for _ in raters]}
    references = {}
    for r_index, rows in enumerate(raters):
        for row in rows:
            grounded_col, baseline_col = ("a", "b") if key[row["id"]]["a_is_grounded"] else ("b", "a")
            for system, col in (("grounded", grounded_col), ("baseline", baseline_col)):
                outputs[system][row["id"]] = row[f"output_{col}"]
                ratings[system].append(float(row[f"rating_{col}_1to5"]))
                hallucinated[system][r_index].append(int(row[f"hallucinated_{col}_0or1"]))
            if row["reference_explanation"].strip() and row["id"] not in references:
                references[row["id"]] = row["reference_explanation"].strip()

    ids = sorted(references, key=int)
    result = {"raters": len(raters), "clauses": len(raters[0]), "clauses_with_reference": len(ids)}
    for system in ("grounded", "baseline"):
        per_clause = [statistics.mean(h[i] for h in hallucinated[system]) for i in range(len(raters[0]))]
        result[system] = {
            "rouge_l": round(statistics.mean(rouge_l(outputs[system][i], references[i]) for i in ids), 4) if ids else None,
            "human_rating_mean": round(statistics.mean(ratings[system]), 2),
            "hallucination_rate": round(statistics.mean(per_clause), 4),
        }

    try:
        from bert_score import score as bert_score
        for system in ("grounded", "baseline"):
            _, _, f1 = bert_score([outputs[system][i] for i in ids], [references[i] for i in ids], lang="en")
            result[system]["bertscore_f1"] = round(f1.mean().item(), 4)
    except ImportError:
        result["bertscore_f1"] = "skipped: pip install bert-score"

    g, b = result["grounded"]["hallucination_rate"], result["baseline"]["hallucination_rate"]
    result["hallucination_reduction_percent"] = round(100 * (b - g) / b, 1) if b else None
    if len(raters) == 2:
        result["inter_rater_kappa_hallucination"] = {
            s: round(cohens_kappa(hallucinated[s][0], hallucinated[s][1]), 3) for s in ("grounded", "baseline")}
    save("generation_scores.json", result)


def latency(runs=5):
    import app

    pages = sorted((DATA / "pages").glob("*.txt"))
    if not pages:
        sys.exit("No test pages. Save the visible text of each test site as data/pages/<site>.txt "
                 "(open the page, select all, copy, paste into the file).")
    client = app.app.test_client()
    client.post("/process", json={"text": pages[0].read_text()})  # warm-up: first call pays one-off lazy init

    results = []
    for page in pages:
        text = page.read_text()
        seconds = []
        for _ in range(runs):
            start = time.perf_counter()
            client.post("/process", json={"text": text})
            seconds.append(time.perf_counter() - start)
        words = len(text.split())
        results.append({
            "page": page.stem,
            "runs": runs,
            "median_seconds": round(statistics.median(seconds), 2),
            "min_seconds": round(min(seconds), 2),
            "max_seconds": round(max(seconds), 2),
            "words_on_page": words,
            "estimated_manual_reading_minutes": round(words / READING_WPM, 1),
        })
    save("latency.json", {
        "note": f"backend /process time (warm models); reading time = words / {READING_WPM} wpm",
        "pages": results,
        "average_median_seconds": round(statistics.mean(r["median_seconds"] for r in results), 2),
    })


def memory():
    import app
    import faiss

    pages = sorted((DATA / "pages").glob("*.txt"))
    text = pages[0].read_text() if pages else " ".join(app.ETHICAL_CLAUSES + app.NON_ETHICAL_CLAUSES)
    app.app.test_client().post("/process", json={"text": text})  # exercise the full pipeline once

    mb = 2 ** 20
    weights = lambda m: sum(p.numel() * p.element_size() for p in m.parameters()) / mb
    embedding = weights(app.semantic_model)
    generator = weights(app.model)
    index = sum(len(faiss.serialize_index(i)) for i in (app.ethical_index, app.non_ethical_index)) / mb
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (mb if sys.platform == "darwin" else 1024)

    save("memory.json", {
        "embedding_model_weights_mb": round(embedding, 1),
        "flan_t5_weights_mb": round(generator, 1),
        "faiss_indexes_mb": round(index, 3),
        "everything_else_mb": round(peak - embedding - generator - index, 1),
        "total_peak_process_rss_mb": round(peak, 1),
        "note": "component rows are in-memory weight/index sizes; 'everything else' is Python, "
                "PyTorch/Flask runtime, tokenizers and inference activations",
    })


if __name__ == "__main__":
    commands = {"classification": classification, "compare": compare, "privacy": privacy,
                "generation": generation, "latency": latency, "memory": memory}
    if len(sys.argv) >= 3 and sys.argv[1] == "score-generation":
        score_generation(sys.argv[2:])
    elif sys.argv[1:2] == ["classification"]:
        classification(*sys.argv[2:4])  # optional: <retrieval variant> <classifier>
    elif len(sys.argv) == 2 and sys.argv[1] in commands:
        commands[sys.argv[1]]()
    else:
        sys.exit(__doc__)
