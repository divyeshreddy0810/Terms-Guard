"""
What is Legal-BERT actually keying on? Run with: python explain_tokens.py [clauses] [samples]

Clause matching explains itself - the decision trace names the clauses, scores and
vote that produced a verdict, which is the mechanism rather than an approximation
of it. Legal-BERT does not: it is the one trained model in the project, so this
script attributes its verdicts to individual words with LIME.

Two questions it answers:

  per clause   which words pushed this clause towards "unfair"
  overall      which words push verdicts across many clauses - if those are legal
               terms ("waive", "arbitration", "indefinitely") the model learned the
               task; if they are company names or formatting artefacts it learned a
               shortcut, which 630 unfair training examples make quite likely

Three modes:

  (default)   per-clause attributions and the aggregate shortcut check
  control     does the verdict survive when the legal words are stripped out, leaving
              only the grammatical skeleton? If it does, the model keys on wording
              habits rather than on meaning
  agreement   do LIME and SHAP, two different approximations, name the same words?
              Methods that disagree with each other cannot both be trusted

Offline only: ~10 seconds per clause, far too slow for the extension, which is why
the live tool explains itself through retrieval evidence instead.

Writes eval_results/explanations/.
"""
import collections
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from lime.lime_text import LimeTextExplainer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval_results", "explanations")
BATCH = 32
# words that carry no legal meaning on their own: if these dominate, the model is
# keying on wording habits rather than on what the clause actually does
FUNCTION_WORDS = {"the", "a", "an", "of", "to", "and", "or", "in", "for", "on", "at", "by", "with",
                  "is", "are", "be", "been", "as", "that", "this", "these", "those", "it", "its",
                  "any", "all", "such", "will", "shall", "under", "from", "into", "if", "than"}


def predictor(classifier, tokeniser):
    """LIME needs probabilities over [fair, unfair] for a list of texts."""
    import torch

    def predict(texts):
        probabilities = []
        for start in range(0, len(texts), BATCH):
            batch = tokeniser(list(texts[start:start + BATCH]), padding=True, truncation=True,
                              max_length=256, return_tensors="pt")
            with torch.no_grad():
                logits = classifier(**batch).logits
            # the model scores eight unfairness categories; a clause is unfair when any fires
            unfair = logits.sigmoid().max(-1).values.numpy()
            probabilities.append(np.stack([1 - unfair, unfair], axis=1))
        return np.concatenate(probabilities)

    return predict


# enough of English's closing words to leave a clause's grammar intact once the
# legal vocabulary is removed
STOPWORDS = FUNCTION_WORDS | {
    "we", "you", "your", "our", "us", "may", "can", "we'll", "not", "no", "nor", "but", "so",
    "their", "they", "them", "he", "she", "his", "her", "who", "whom", "which", "what", "when",
    "where", "how", "why", "there", "here", "about", "after", "before", "between", "during",
    "without", "within", "through", "over", "up", "down", "out", "off", "do", "does", "did",
    "have", "has", "had", "was", "were", "being", "am", "would", "could", "should", "must",
    "also", "other", "another", "each", "every", "some", "more", "most", "only", "own", "same",
}


def split_words(text):
    import re
    return re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)


def strip_content(text):
    """Keep the grammar, drop the legal vocabulary."""
    return "".join(part for part in split_words(text)
                   if not part.isalpha() or part.lower() in STOPWORDS).strip()


def strip_function(text):
    """The opposite: keep the legal vocabulary, drop the grammar."""
    return " ".join(part for part in split_words(text)
                    if part.isalpha() and part.lower() not in STOPWORDS).strip()


def control():
    """Does the verdict survive with the legal words removed?"""
    import app

    tokeniser, classifier = app.get_legalbert()
    predict = predictor(classifier, tokeniser)
    rows = app.load_unfair_tos("test")
    clauses = [r["text"].strip() for r in rows if r["labels"] and 60 <= len(r["text"].strip()) <= 300]
    original = predict(clauses)[:, 1]
    clauses = [c for c, p in zip(clauses, original) if p >= 0.5]      # only ones it catches
    original = np.array([p for p in original if p >= 0.5])

    skeleton = predict([strip_content(c) for c in clauses])[:, 1]
    content = predict([strip_function(c) for c in clauses])[:, 1]

    def summarise(name, scores):
        print(f"  {name:34s} mean unfair probability {scores.mean():.3f}   "
              f"still called unfair: {(scores >= 0.5).mean():.0%}")

    print(f"{len(clauses)} unfair clauses the model catches\n")
    summarise("original clause", original)
    summarise("grammar only (legal words gone)", skeleton)
    summarise("legal words only (grammar gone)", content)
    print(f"\nexample: {clauses[0][:90]}")
    print(f"  grammar only    : {strip_content(clauses[0])[:90]}")
    print(f"  legal words only: {strip_function(clauses[0])[:90]}")
    drop = 1 - skeleton.mean() / original.mean()
    print(f"\nStripping the legal vocabulary removes {drop:.0%} of the model's confidence.")
    print("A small drop would mean it keys on wording habits; a large drop means it needs the words themselves.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "control_test.json"), "w") as f:
        json.dump({"clauses": len(clauses),
                   "original_mean": round(float(original.mean()), 4),
                   "grammar_only_mean": round(float(skeleton.mean()), 4),
                   "legal_words_only_mean": round(float(content.mean()), 4),
                   "grammar_only_still_unfair": round(float((skeleton >= 0.5).mean()), 4),
                   "legal_words_only_still_unfair": round(float((content >= 0.5).mean()), 4),
                   "confidence_lost_without_legal_words": round(float(drop), 4)}, f, indent=1)
    print(f"saved {OUT}/control_test.json")


def agreement(clause_count=20, samples=400):
    """Do LIME and SHAP pick the same words?"""
    import shap
    import app

    tokeniser, classifier = app.get_legalbert()
    predict = predictor(classifier, tokeniser)
    rows = app.load_unfair_tos("test")
    clauses = [r["text"].strip() for r in rows if r["labels"] and 60 <= len(r["text"].strip()) <= 300]
    clauses = [c for c, p in zip(clauses, predict(clauses)[:, 1]) if p >= 0.5][:clause_count]

    lime_explainer = LimeTextExplainer(class_names=["fair", "unfair"], bow=False, random_state=0)
    shap_explainer = shap.Explainer(lambda texts: predict(list(texts))[:, 1],
                                    shap.maskers.Text(r"\W+"), silent=True)

    def clean(token):
        return token.lower().strip(".,;:()\"' ")

    overlaps, rows_out = [], []
    for i, clause in enumerate(clauses, 1):
        lime_top = [clean(t) for t, w in lime_explainer.explain_instance(
            clause, predict, num_features=5, num_samples=samples, labels=(1,)).as_list(label=1) if w > 0]
        values = shap_explainer([clause], max_evals=300)
        order = np.argsort(-values.values[0])
        shap_top = [clean(values.data[0][j]) for j in order[:5] if values.values[0][j] > 0]
        shared = set(lime_top) & set(shap_top)
        overlap = len(shared) / max(len(set(lime_top) | set(shap_top)), 1)
        overlaps.append(overlap)
        rows_out.append({"clause": clause, "lime": lime_top, "shap": shap_top,
                         "shared": sorted(shared), "overlap": round(overlap, 3)})
        print(f"[{i}/{len(clauses)}] overlap {overlap:.2f}  LIME {lime_top[:3]}  SHAP {shap_top[:3]}", flush=True)

    mean = float(np.mean(overlaps))
    print(f"\nmean agreement between LIME and SHAP: {mean:.1%} of the words they name are the same")
    print(f"clauses where they agree on nothing: {sum(1 for o in overlaps if o == 0)}/{len(overlaps)}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "lime_shap_agreement.json"), "w") as f:
        json.dump({"clauses": len(clauses), "mean_overlap": round(mean, 4), "per_clause": rows_out}, f, indent=1)
    print(f"saved {OUT}/lime_shap_agreement.json")


def main():
    clause_count = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    import app

    tokeniser, classifier = app.get_legalbert()
    predict = predictor(classifier, tokeniser)

    rows = app.load_unfair_tos("test")
    unfair = [r for r in rows if r["labels"] and 60 <= len(r["text"].strip()) <= 300]
    predictions = predict([r["text"].strip() for r in unfair])[:, 1]
    caught = [(r, float(p)) for r, p in zip(unfair, predictions) if p >= 0.5][:clause_count]
    print(f"explaining {len(caught)} unfair clauses the model catches, {samples} perturbations each\n")

    explainer = LimeTextExplainer(class_names=["fair", "unfair"], bow=False, random_state=0)
    per_clause, weights = [], collections.defaultdict(list)
    for i, (row, probability) in enumerate(caught, 1):
        text = row["text"].strip()
        explanation = explainer.explain_instance(text, predict, num_features=8,
                                                 num_samples=samples, labels=(1,))
        tokens = explanation.as_list(label=1)
        per_clause.append({"text": text, "categories": row["labels"],
                           "unfair_probability": round(probability, 3),
                           "tokens": [[t, round(w, 4)] for t, w in tokens]})
        for token, weight in tokens:
            if weight > 0:
                weights[token.lower().strip(".,;:()\"'")].append(weight)
        top = ", ".join(f"{t}({w:+.2f})" for t, w in tokens[:4])
        print(f"[{i}/{len(caught)}] {probability:.2f} {row['labels'][0][:22]:24s} {top}", flush=True)

    ranked = sorted(((token, len(ws), float(np.mean(ws))) for token, ws in weights.items() if token),
                    key=lambda row: (row[1], row[2]), reverse=True)
    content = sum(1 for token, count, _ in ranked[:25] if token not in FUNCTION_WORDS)

    print(f"\nwords that pushed clauses towards 'unfair', most consistent first:")
    for token, count, mean in ranked[:25]:
        marker = "" if token not in FUNCTION_WORDS else "   <- function word"
        print(f"  {token:22s} in {count:3d} clauses, mean weight {mean:+.3f}{marker}")
    print(f"\n{content} of the top 25 carry legal meaning; {25 - content} are function words.")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "lime_tokens.json"), "w") as f:
        json.dump({"clauses_explained": len(caught), "samples_per_clause": samples,
                   "per_clause": per_clause,
                   "ranked_tokens": [{"token": t, "clauses": c, "mean_weight": round(m, 4)} for t, c, m in ranked]},
                  f, indent=1)
    print(f"\nsaved {OUT}/lime_tokens.json")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "control":
        control()
    elif mode == "agreement":
        agreement(*(int(a) for a in sys.argv[2:4]))
    else:
        main()
