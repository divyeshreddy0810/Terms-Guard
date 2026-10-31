"""
Builds a privacy-practice clause corpus from OPP-115, because UNFAIR-ToS has no
privacy categories at all - it covers contract fairness only, so the system could
not see data selling, tracking or retention. Run with: python build_privacy_corpus.py

OPP-115 (Wilson et al., ACL 2016) is 115 privacy policies annotated by law
students. Each annotation carries attributes, including "Does / Does Not", which
says whether the policy states the practice happens or explicitly denies it. That
attribute is what lets a practice be scored as harmful or protective rather than
merely present.

Mapping (documented here because it is a judgement call, not part of the dataset):

  harmful     third-party sharing for any purpose except a legal requirement or
              running/securing the service (an undisclosed purpose counts)
              first-party collection for advertising, marketing or profiling
              data retained indefinitely, or with no stated period
              Do Not Track signals not honoured
              no way to access or delete your data
  protective  the policy explicitly says it does NOT collect or share ("Does Not")
              opt-in consent, opt-out links or browser privacy controls
              a stated or limited retention period
              any named security measure
              account deletion, editing or viewing offered

Clause text is the full sentence from the policy segment that contains the
annotated span, not the span itself: a span often omits the "we will not" that
carries the meaning, which would put a protective clause into the corpus reading
as a harmful one.

Policies are split 80/20 by document, so the held-out clauses in
data/privacy_test.json come from policies the corpus has never seen - the same
discipline the UNFAIR-ToS split follows.

Writes data/privacy_corpus.json (clauses) and data/polarity_pairs.json (the
"Does" / "Does Not" sentences of the same practice, used to measure negation handling).
"""
import collections
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OPP = os.path.join(DATA, "OPP-115")
MIN_LENGTH, MAX_LENGTH = 40, 300
MIN_ANNOTATORS = 2   # OPP-115 segments carry three annotators; require agreement

HARMFUL = "harmful"
PROTECTIVE = "protective"


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text).strip(" -*•")


def load_segments(policy_file):
    """Policy text split into the numbered segments the annotations refer to."""
    name = os.path.basename(policy_file).replace(".csv", ".html")
    path = os.path.join(OPP, "sanitized_policies", name)
    if not os.path.exists(path):
        return []
    return open(path, encoding="utf-8", errors="replace").read().split("|||")


def sentence_for(segments, index, span):
    """The whole sentence containing the annotated span, so negations survive."""
    try:
        segment = clean(segments[int(index)])
    except (IndexError, ValueError):
        return None
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", segment) if s.strip()]
    key = span[:40].lower()
    for sentence in sentences:
        if key and key in sentence.lower():
            return sentence
    return None


def classify(category, values):
    """(polarity, label) for one annotation's attribute values, or None to skip."""
    does = values.get("Does/Does Not")
    purpose = values.get("Purpose")

    if does == "Does Not":
        return PROTECTIVE, f"Privacy: states it does not {'share' if 'Third Party' in category else 'collect'}"
    if category == "Third Party Sharing/Collection" and does == "Does":
        # sharing to meet the law or to run the service is not unfair; everything else
        # hands your data to someone else, including sharing with no stated purpose
        if purpose not in ("Legal requirement", "Service operation and security", "Merger/Acquisition"):
            if purpose in ("Advertising", "Marketing"):
                return HARMFUL, "Privacy: shares data with third parties for advertising"
            return HARMFUL, "Privacy: shares data with third parties"
    if category == "First Party Collection/Use" and does == "Does" and purpose in (
            "Advertising", "Marketing", "Personalization/Customization"):
        return HARMFUL, "Privacy: collects data for advertising"
    if category == "Data Retention" and values.get("Retention Period") == "Indefinitely":
        return HARMFUL, "Privacy: retains data indefinitely"
    if category == "Data Retention" and values.get("Retention Period") == "Unspecified":
        return HARMFUL, "Privacy: does not say how long data is kept"
    if category == "Data Retention" and values.get("Retention Period") in ("Stated Period", "Limited"):
        return PROTECTIVE, "Privacy: states a retention period"
    if category == "Do Not Track" and values.get("Do Not Track policy") == "Not honored":
        return HARMFUL, "Privacy: ignores Do Not Track"
    if category == "User Access, Edit and Deletion":
        access = values.get("Access Type")
        if access == "None":
            return HARMFUL, "Privacy: no access to your data"
        if access in ("Delete account (full)", "Delete account (partial)", "Edit information", "View"):
            return PROTECTIVE, "Privacy: lets you access or delete your data"
    if category == "User Choice/Control" and values.get("Choice Type") in (
            "Opt-in", "Opt-out link", "Opt-out via contacting company", "Browser/device privacy controls"):
        return PROTECTIVE, "Privacy: offers a choice about data use"
    if category == "Data Security" and values.get("Security Measure") not in (None, "Other"):
        return PROTECTIVE, "Privacy: describes a security measure"
    return None


def main():
    if not os.path.isdir(OPP):
        sys.exit(f"{OPP} not found. Download OPP-115 first:\n"
                 f"  curl -o {DATA}/OPP-115.zip https://usableprivacy.org/static/data/OPP-115_v1_0.zip\n"
                 f"  unzip -q {DATA}/OPP-115.zip -d {DATA}")

    # votes[(policy, segment, span_text)][(polarity, label)] = number of annotators
    votes = collections.defaultdict(collections.Counter)
    pair_source = collections.defaultdict(lambda: {HARMFUL: set(), PROTECTIVE: set()})

    for path in glob.glob(os.path.join(OPP, "annotations", "*.csv")):
        policy = os.path.basename(path)
        segments = load_segments(path)
        for row in csv.reader(open(path, encoding="utf-8", errors="replace")):
            if len(row) < 7:
                continue
            category = row[5]
            try:
                attributes = json.loads(row[6])
            except json.JSONDecodeError:
                continue
            values = {a: p.get("value") for a, p in attributes.items()
                      if p.get("value") not in (None, "null", "not-selected")}
            verdict = classify(category, values)
            if verdict is None:
                continue
            # the longest annotated span is the clearest statement of the practice
            spans = [clean(p.get("selectedText", "")) for p in attributes.values()
                     if p.get("selectedText") not in (None, "null")]
            spans = [s for s in spans if len(s) >= 20]
            if not spans:
                continue
            sentence = sentence_for(segments, row[4], max(spans, key=len))
            if sentence is None or not (MIN_LENGTH <= len(sentence) <= MAX_LENGTH):
                continue
            votes[(policy, row[4], sentence)][verdict] += 1
            pair_source[category][verdict[0]].add(sentence)

    # every fifth policy (alphabetically) is held out, so test clauses come from
    # policies the corpus never saw
    policies = sorted({policy for policy, _, _ in votes})
    held_out = set(policies[::5])

    clauses = {}
    for (policy, _, sentence), counted in votes.items():
        (polarity, label), n = counted.most_common(1)[0]
        if n >= MIN_ANNOTATORS and sentence not in clauses:
            clauses[sentence] = {"text": sentence, "polarity": polarity, "label": label,
                                 "split": "test" if policy in held_out else "train"}

    def side(split, polarity):
        return [c for c in clauses.values() if c["split"] == split and c["polarity"] == polarity]

    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "privacy_corpus.json"), "w") as f:
        json.dump({"source": "OPP-115 (Wilson et al., ACL 2016), training policies",
                   "harmful": side("train", HARMFUL), "protective": side("train", PROTECTIVE)}, f, indent=1)
    with open(os.path.join(DATA, "privacy_test.json"), "w") as f:
        json.dump({"source": f"OPP-115, {len(held_out)} held-out policies",
                   "harmful": side("test", HARMFUL), "protective": side("test", PROTECTIVE)}, f, indent=1)

    print(f"privacy corpus (train, {len(policies) - len(held_out)} policies): "
          f"{len(side('train', HARMFUL))} harmful, {len(side('train', PROTECTIVE))} protective")
    print(f"privacy test   ({len(held_out)} policies): "
          f"{len(side('test', HARMFUL))} harmful, {len(side('test', PROTECTIVE))} protective")
    by_label = collections.Counter(c["label"] for c in clauses.values())
    for label, n in by_label.most_common():
        print(f"  {label:52s} {n}")

    # polarity pairs: same practice, stated vs explicitly denied - for the negation tests
    pairs = {category: {"harmful": sorted(sides[HARMFUL])[:60], "protective": sorted(sides[PROTECTIVE])[:60]}
             for category, sides in pair_source.items() if sides[HARMFUL] and sides[PROTECTIVE]}
    with open(os.path.join(DATA, "polarity_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=1)
    print(f"\npolarity pairs written for {len(pairs)} categories")


if __name__ == "__main__":
    main()
