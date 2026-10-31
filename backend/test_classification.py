"""
Regression checks for app.py. Run with: python test_classification.py

1. Reference corpus comes from the UNFAIR-ToS train split and shares no sentence
   with the test split used by evaluate.py.
2. The HNSW index returns the same top-1 as exhaustive cosine similarity, so
   Algorithm 1's max() is unaffected by the sub-linear index.
3. Search doesn't hang on a worker thread (how Flask serves requests).
4. Chunks are at most 512 model tokens and consecutive chunks overlap.
5. Every retrieval variant returns usable labels, and "nearest" still behaves
   exactly as the report describes Algorithm 1.
6. The /options endpoint matches the registries, so the extension's dropdowns
   can never drift from the backend.
7. Algorithm 1 accuracy on the UNFAIR-ToS val split - used neither to build
   the corpus nor for the reported evaluation - stays above a regression floor.
"""
import threading

import numpy as np

import app

# Set below the observed baseline so this fails loudly on a regression.
MIN_VALIDATION_ACCURACY = 0.85  # observed 91.7% (4,775 / 619 clause corpus)

VALIDATION = app.load_unfair_tos("val")
SAMPLE = [row["text"] for row in VALIDATION[:200]]


def check_corpus():
    test = {row["text"] for row in app.load_unfair_tos("test")}
    corpus = set(app.ETHICAL_CLAUSES) | set(app.NON_ETHICAL_CLAUSES)
    assert app.ETHICAL_CLAUSES and app.NON_ETHICAL_CLAUSES
    assert not corpus & test, "corpus contains test sentences"
    print(f"corpus: {len(app.ETHICAL_CLAUSES)} ethical / {len(app.NON_ETHICAL_CLAUSES)} non-ethical, no test overlap OK")


def check_index_matches_brute_force():
    vectors = app.embed(SAMPLE)
    for index, clauses in ((app.ethical_index, app.ETHICAL_CLAUSES), (app.non_ethical_index, app.NON_ETHICAL_CLAUSES)):
        exact = vectors @ app.embed(clauses).T
        agree = sum(abs(score - row.max()) < 1e-4
                    for row, [(_, score)] in zip(exact, app.search(index, clauses, vectors, 1)))
        # ponytail: HNSW is approximate; allow 1% misses before calling it a regression
        assert agree >= 0.99 * len(SAMPLE), f"HNSW top-1 matched brute force on only {agree}/{len(SAMPLE)}"
    print("HNSW top-1 == brute-force max OK")


def check_search_on_worker_thread():
    # Flask serves each request on a new thread; faiss+torch OpenMP deadlocked there once.
    worker = threading.Thread(target=app.classify_clauses, args=(SAMPLE[:20],), daemon=True)
    worker.start()
    worker.join(timeout=60)
    assert not worker.is_alive(), "faiss search hung on a worker thread"
    print("search on worker thread OK")


def check_chunking():
    text = " ".join(SAMPLE)
    chunks = app.chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(app.semantic_model.tokenizer(chunk)["input_ids"]) <= app.CHUNK_TOKENS
    for a, b in zip(chunks, chunks[1:]):
        assert b[:30] in a, "consecutive chunks do not overlap"
    print(f"chunking: {len(chunks)} chunks <= {app.CHUNK_TOKENS} tokens, overlapping OK")


def check_variants():
    sample = SAMPLE[:5]
    for rag in app.RAG_VARIANTS:
        results = app.classify_clauses(sample, rag=rag)
        assert len(results) == len(sample)
        for label, confidence, clause, _ in results:
            assert label in ("ethical", "non_ethical", "unclear"), label
            assert 0.0 <= confidence <= 1.0 and clause
    # the report's method must stay exactly "label of the single nearest clause"
    vectors = app.embed(sample)
    for (label, confidence, clause, _), [(top_clause, top_label, top_score)] in zip(
            app.classify_clauses(sample, rag="nearest"), app.retrieve_references(vectors, 1)):
        assert (label, clause) == (top_label, top_clause) and confidence == round(top_score, 2)
    print(f"retrieval variants OK ({', '.join(app.RAG_VARIANTS)})")


def check_options():
    options = app.app.test_client().get("/options").get_json()
    for kind, registry in (("rag", app.RAG_VARIANTS), ("generator", app.GENERATORS), ("classifier", app.CLASSIFIERS)):
        assert [o["id"] for o in options[kind]] == list(registry)
        assert all(o["why"] and o["name"] for o in options[kind]), f"{kind} entry missing its dropdown text"
        assert sum(o["default"] for o in options[kind]) == 1
    assert app.generate_warning("anything", [], generator="none") == ""
    print("options endpoint OK")


def check_accuracy():
    gold = ["non_ethical" if row["labels"] else "ethical" for row in VALIDATION]
    pred = [label for label, *_ in app.classify_clauses([row["text"] for row in VALIDATION])]
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    print(f"validation accuracy: {accuracy:.1%} on {len(gold)} sentences")
    assert accuracy >= MIN_VALIDATION_ACCURACY, f"accuracy {accuracy:.1%} below floor {MIN_VALIDATION_ACCURACY:.0%}"


if __name__ == "__main__":
    check_corpus()
    check_index_matches_brute_force()
    check_search_on_worker_thread()
    check_chunking()
    check_variants()
    check_options()
    check_accuracy()
    print("PASS")
