# TermsGuard AI — measured results

All figures on the LexGLUE UNFAIR-ToS test split (1,607 expert-annotated sentences, never seen in training
or corpus construction). Majority-class baseline (always "fair") = 0.893 accuracy.

## Table 1. System comparison

| System | Verdict method | Accuracy | Macro F1 | Unfair P | Unfair R | Unfair F1 |
|---|---|---|---|---|---|---|
| Original project (93 hand-written clauses, keywords, unclear) | — | 0.156 | 0.228 | 0.169 | 0.506 | 0.253 |
| Report's method, 48 hand-written clauses | Clause matching | 0.355 | 0.332 | 0.120 | 0.796 | 0.209 |
| Nearest clause (report's Algorithm 1) | Clause matching | 0.921 | 0.806 | 0.613 | 0.709 | 0.658 |
| 5-clause vote | Clause matching | 0.948 | 0.853 | 0.801 | 0.680 | 0.736 |
| Keyword + meaning (hybrid) | Clause matching | 0.957 | 0.880 | 0.850 | 0.727 | 0.784 |
| Vote with unclear | Clause matching | 0.739 | 0.673 | 0.862 | 0.326 | 0.473 |
| Cross-encoder rerank | Clause matching | 0.950 | 0.872 | 0.749 | 0.796 | 0.772 |
| Legal-BERT small (8 categories) | Legal-BERT | 0.960 | 0.893 | 0.833 | 0.785 | 0.808 |
| Nearest clause | Legal-BERT base (binary) | 0.966 | 0.910 | 0.846 | 0.831 | 0.839 |

## Table 2. Retrieval variants (clause matching, no training)

| Variant | Unfair P | Unfair R | Unfair F1 | Abstentions | Seconds / 1,000 sentences |
|---|---|---|---|---|---|
| Nearest clause (report's Algorithm 1) | 0.613 | 0.709 | 0.658 | 0 | 3.0 |
| 5-clause vote | 0.801 | 0.680 | 0.736 | 0 | 3.1 |
| Keyword + meaning (hybrid) | 0.850 | 0.727 | 0.784 | 0 | 10.7 |
| Vote with unclear | 0.862 | 0.326 | 0.473 | 386 | 3.2 |
| Cross-encoder rerank | 0.749 | 0.796 | 0.772 | 0 | 222.5 |

## Table 3. Recall by unfair category

| Category | Sentences | Keyword + meaning | Legal-BERT |
|---|---|---|---|
| Unilateral change | 38 | 0.76 | 0.74 |
| Unilateral termination | 38 | 0.74 | 0.87 |
| Limitation of liability | 38 | 0.68 | 0.82 |
| Contract by using | 23 | 0.52 | 0.78 |
| Jurisdiction | 16 | 0.94 | 1.00 |
| Content removal | 13 | 0.77 | 0.85 |
| Choice of law | 13 | 1.00 | 1.00 |
| Arbitration | 7 | 0.57 | 1.00 |

## Table 4. Combination experiments (Legal-BERT base + keyword fusion)

| Strategy | Unfair P | Unfair R | Unfair F1 |
|---|---|---|---|
| Legal-BERT alone | 0.846 | 0.831 | 0.839 |
| Keyword fusion alone | 0.850 | 0.727 | 0.784 |
| Either flags (OR) | 0.782 | 0.878 | 0.827 |
| Both flag (AND) | 0.951 | 0.680 | 0.793 |
| Decision threshold swept 0.2-0.7 | best at 0.50 | — | 0.839 |

## Table 5. System resources and runtime

| Measure | Original project | Current |
|---|---|---|
| Reference clauses | 93 hand-written | 5,394 from annotated ToS |
| Embedding model | all-MiniLM-L6-v2 (22.7M params) | unchanged |
| Summary model | flan-t5-base (247.6M) | selectable: base / small / Qwen2.5-0.5B / none |
| Peak memory | 1.07 GB | 1.17 GB (+0.42 GB with Legal-BERT) |
| FAISS index | 0.08 MB | 9.3 MB |
| Backend startup | ~13 s | ~40 s (embeds the corpus) |
| Analysis per page (warm) | 14.0 s | 3.9 s |
| Retrieval recall@3 | 73.6% | 84.3% |
| Validation accuracy | not measured | 94.6% |
| Automated checks | 1 | 7 |

Reproduce: `python evaluate.py compare`, `python evaluate.py classification`, `python evaluate.py memory`.
