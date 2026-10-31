# Terms Guard AI 🛡️

A browser extension + local Flask backend that analyzes Terms & Conditions in real-time, flags clauses as ethical or non-ethical against a reference corpus, and explains the terms in plain language with a locally-run, retrieval-grounded LLM.

## Team Members

1. Divyesh Reddy Ellasiri (25134825)
2. Sai Vivek Yerninti (25138286)
3. Mukesh Saren Ramu (25121448)
4. Alabi Taiwo Oluwatamilore (24335592)

## Features

- **Real-time T&C analysis**: extracts the current page's text on demand and analyzes it locally.
- **Ethical assessment**: labels each sentence ethical / non-ethical with a confidence score and the reference clause it matched.
- **Plain-language summary**: FLAN-T5 warnings grounded in the retrieved reference clauses.
- **Risk scoring**: rolls per-clause labels up into a Low/Medium/Medium-High/High risk badge.
- **In-page highlighting**: colors classified sentences directly on the webpage.
- **Link extraction**: surfaces URLs found on the page (e.g. linked policy pages).
- **Switchable pipeline**: dropdowns in the popup choose the retrieval variant, the verdict method and the summary model, each with a one-line explanation of when to pick it.

## How it works

An extraction → retrieval → generation pipeline (report Section III):

1. The content script scrapes the **main content** (`<main>`/`<article>` when the page has one, otherwise the body). The backend then drops navigation lines — anything under 40 characters or without a space — and truncates what remains to **3,000 characters**. Without this the first 3,000 characters of a real site are menus and headings, not terms.
2. The text is split into **512-token overlapping chunks** (50-token overlap) and each chunk is embedded with `all-MiniLM-L6-v2` (384 dimensions).
3. **FAISS** (HNSW, cosine similarity) retrieves the **top-3** most similar clauses for each chunk from the reference corpus (see below).
4. The **three most confident flagged clauses** and their retrieved reference clauses are passed to `google/flan-t5-base`, which writes a plain-language summary grounded in that context. (The report describes summarising the top-3 *chunks*; summarising the flagged clauses instead is both clearer and about 3x faster, since the inputs are single sentences rather than 512-token blocks.)
5. The first **15 sentences** are classified with **Algorithm 1**: the label of whichever corpus holds the sentence's most similar reference clause, with that cosine similarity as the confidence.

### Choosing the pipeline

The popup has three dropdowns. Their contents come from `GET /options`, so the names and the one-line reasons live only in `app.py`. The defaults reproduce the method described in the report.

**Retrieval** — how the retrieved clauses become a verdict:

| Option | Pick it when |
|---|---|
| `nearest` | The method in the report: the single most similar clause decides the verdict. |
| `vote` | Five nearest clauses vote, so one odd match cannot raise a false alarm on its own. |
| `hybrid` *(default)* | Adds BM25 keyword search, so exact legal terms like arbitration or indemnify carry weight. Measured best of the untrained options (unfair-F1 0.784 vs 0.658 for `nearest`). |
| `gated` | Leaves boilerplate such as addresses unlabelled instead of forcing a verdict. |
| `rerank` | Re-reads 20 candidates with a cross-encoder: usually the most accurate, and the slowest. |

**Verdict** — how the label is decided:

| Option | Contract F1 | Privacy F1 | Pick it when |
|---|---|---|---|
| `similarity` *(default)* | 0.734 | 0.836 | No training needed; the report's Algorithm 1 |
| `legalbert` | 0.808 | 0.000 | Contract terms only; ignores the corpus entirely |
| `legalbert-knn` | 0.807 | 0.000 | Legal-BERT blended with its nearest reference clauses (weight 0.3, tuned on validation) |
| `legalbert-privacy` | 0.772 | 0.830 | **Best overall**: Legal-BERT for contract terms, clause matching for privacy practices |

The three Legal-BERT options stay disabled in the popup until `train_legalbert.py` has run.

#### Can Legal-BERT consult the corpus?

Yes — `legalbert-knn` embeds every reference clause in Legal-BERT's own space and blends a similarity-weighted vote of the nearest ones into the verdict. Measured on the validation split:

| Blend weight | Contract F1 | Privacy F1 |
|---|---|---|
| 0.0 (model only) | 0.834 | 0.000 |
| **0.3** | **0.845** | 0.000 |
| 0.5 | 0.825 | 0.088 |
| 0.7 | 0.796 | 0.651 |
| 1.0 (corpus only) | 0.776 | 0.728 |

Consulting the corpus helps where the model already has a concept (contract F1 0.834 → 0.845), but it cannot create one: privacy stays at 0.000 until the blend is so corpus-heavy that contract performance collapses, and even pure k-NN in Legal-BERT's space reaches only 0.728 against 0.836 for the same corpus in MiniLM's space. Fine-tuning organised that space around the eight contract categories, so privacy clauses are not separated in it. **Retrieval can supply facts a model lacks, but not a concept it was never trained to represent** — which is why `legalbert-privacy` routes privacy decisions to clause matching instead.

**Summary model** — `flan-t5-base` (default), `flan-t5-small` (about 3× faster, blunter), `qwen2.5-0.5b` (follows instructions best, most memory), or `none` (skip the summary for the fastest analysis). Only one summary model is held in memory; switching unloads the previous one.

These also work from the API:

```bash
curl -X POST http://127.0.0.1:5001/process \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "rag": "hybrid", "classifier": "similarity", "generator": "flan-t5-small"}'
```

### Reference corpus

The ethical and non-ethical clauses come from the **train split of LexGLUE UNFAIR-ToS** (Chalkidis et al., 2022; CC BY 4.0) — sentences from real Terms of Service annotated by legal experts:

- **Non-ethical (619 clauses)**: sentences flagged unfair in any of 8 categories — limitation of liability, unilateral termination, unilateral change, content removal, contract by using, choice of law, jurisdiction, arbitration.
- **Ethical (4,775 clauses)**: sentences not flagged unfair. "Ethical" therefore means *not unfair*, not *protective of the user*.

The dataset (`unfair_tos.tar.gz`, ~500 KB) is downloaded once from Zenodo into `backend/data/` on first start. Train sentences that also appear verbatim in the test split are excluded, so the test split stays unseen for evaluation.

**Privacy clauses (OPP-115).** UNFAIR-ToS covers contract fairness only — it has no category for selling data, tracking or retention, so those clauses were invisible (measured: 0.018 F1 on privacy text). A second corpus is derived from [OPP-115](https://usableprivacy.org/data) (Wilson et al., ACL 2016), 115 privacy policies annotated by law students:

```bash
cd backend/data && curl -O https://usableprivacy.org/static/data/OPP-115_v1_0.zip && unzip -q OPP-115_v1_0.zip
cd .. && python3 build_privacy_corpus.py
```

That writes `data/privacy_corpus.json` (304 harmful / 558 protective, from 92 policies) and `data/privacy_test.json` (109 / 166, from 23 held-out policies). Harmful versus protective comes from the annotators' own attributes — `Does/Does Not`, `Retention Period`, `Choice Type`, `Access Type`, `Do Not Track policy` — and the exact mapping is documented at the top of `build_privacy_corpus.py`. The backend picks the corpus up automatically when the file exists.

### The negation problem

Sentence embeddings do not represent negation: "You **can** delete your account" and "You **cannot** delete your account" score **0.911** similarity, and "we never sell" against "we may sell" scores 0.704. A clause can therefore match the exact statement of its opposite.

`apply_polarity` in `app.py` flips a match when the two disagree about negation **and** are near-identical otherwise (cosine ≥ 0.60, and ≥ 80% of the shorter clause's words shared after negation words are removed). Both conditions matter: legal text negates constantly — "we are **not** liable" *is* the unfair practice — so a bare negation mismatch means nothing. Set `POLARITY_CHECK = False` for the ablation.

See `PROJECT_RUNTIME_EXPLANATION.md` for the full request-by-request flow.

## Technology Stack

- **Backend**: Flask (Python 3.11)
- **Extension**: Chrome/Chromium (Manifest V3, JavaScript, HTML, CSS)
- **NLP models**:
  - `sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional embeddings / semantic similarity
  - `google/flan-t5-base` — grounded plain-language summary
- **Retrieval**: `faiss` (`IndexHNSWFlat`, inner product on normalised vectors = cosine similarity)
- **Libraries**: Transformers, Sentence-Transformers, FAISS, PyTorch, Flask-CORS

## Project Structure

```
├── run.py                        # One-command launcher: environment, backend, checks
├── backend/
│   ├── app.py                    # Flask backend API
│   ├── build_privacy_corpus.py   # Derives the privacy clause corpus from OPP-115
│   ├── evaluate.py               # Measures the report's Section IV metrics
│   ├── test_classification.py    # Regression checks (corpus, index, chunking, accuracy)
│   ├── data/                     # UNFAIR-ToS release (downloaded on first start), test pages for latency
│   ├── eval_results/             # Measured evaluation outputs
│   └── requirements.txt          # Python dependencies
├── extension/
│   ├── manifest.json             # Chrome extension manifest
│   ├── popup.html                # Popup UI
│   ├── popup.js                  # Popup logic
│   ├── content.js                # Content script for webpage interaction
│   └── icons/                    # Extension icons
├── PROJECT_RUNTIME_EXPLANATION.md
└── readme.md
```

## Quick start (every time)

```bash
python3 run.py
```

That is the whole thing. `run.py` creates the virtual environment if it is missing, installs dependencies on first use, starts the backend, waits until the models are loaded, prints the current pipeline defaults and the browser steps, and shuts down cleanly on Ctrl+C. It reuses an already-running backend rather than starting a second one.

| Command | What it does |
|---|---|
| `python3 run.py` | Start the backend and keep it running |
| `python3 run.py --smoke` | Start it, analyse a sample clause set, print the result, exit |
| `python3 run.py --test` | Run the regression checks |
| `python3 run.py --eval` | Run the full comparison against the UNFAIR-ToS test split |
| `python3 run.py --setup` | Create the venv and install dependencies, then exit |
| `python3 run.py --port 5002` | Serve on a different port |

Startup is about 40 seconds (it embeds 5,394 reference clauses); the first analysis after that takes ~8 s and later ones about 4 s.

Then in Chrome or Brave: open any Terms & Conditions page, click the **Terms Guard AI** icon, and click **Analyze This Page**.

**After changing anything in `extension/`**, go to `chrome://extensions/` (or `brave://extensions`) and press Reload on the Terms Guard AI card, then reload the web page you are testing.

## Setup

Requires Python 3.11, ~2GB RAM, and ~2GB disk for pre-trained models (first run downloads them).

```bash
git clone https://github.com/divyeshreddy0810/Terms-Guard.git
cd Terms-Guard/backend
python3.11 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: first-time install may take 5-10 minutes to download pre-trained models (~2GB).

### Install the browser extension

1. Open Chrome/Chromium and go to `chrome://extensions/`.
2. Enable **Developer mode** (top right).
3. Click **Load unpacked**.
4. Select the `extension/` folder from this project.
5. The extension icon should appear in the toolbar.

## Running it

### Start the backend

```bash
cd backend
source venv/bin/activate   # Windows: venv\Scripts\activate
python app.py
```

You should see:

```
LOADING MODELS (This takes 1-2 minutes on first run)...
✅ Models loaded in XX.XX seconds
Running on http://127.0.0.1:5001
```

### Use the extension

1. Keep the backend server running.
2. Open any website's Terms & Conditions page.
3. Click the **Terms Guard AI** icon in the toolbar.
4. Click **Analyze This Page** — results and highlights appear within a few seconds.

The extension talks to the backend at `http://127.0.0.1:5001`, so the backend must be running locally. The backend only accepts browser requests from `chrome-extension://` origins.

## API

- **POST** `/process` — analyze text; optional `rag`, `classifier` and `generator` select the pipeline. Returns per-sentence labels, confidence, matched clause, risk level, summary, and which options were used.
- **GET** `/options` — the dropdown contents: id, name, one-line reason, default, and whether the option is available.
- **GET** `/health` — backend liveness check.

```bash
curl -X POST http://127.0.0.1:5001/process \
  -H "Content-Type: application/json" \
  -d '{"text": "Your T&C text here (50+ characters)."}'
```

## Testing

```bash
cd backend
source venv/bin/activate
python test_classification.py
```

Checks that the corpus shares no sentence with the test split, that the HNSW index returns the same nearest clause as exhaustive cosine similarity, that search doesn't hang on a Flask worker thread, that chunks are ≤512 tokens and overlap, and that Algorithm 1's accuracy on the UNFAIR-ToS validation split stays above a regression floor.

## Evaluation

`evaluate.py` measures each Section IV metric and writes the result to `eval_results/`. Numbers are reported as measured.

| Command | Measures | Needs |
|---|---|---|
| `python evaluate.py compare` | Every retrieval variant and Legal-BERT side by side, written to `eval_results/comparison.md` | Legal-BERT row needs the training step |
| `python evaluate.py privacy` | Privacy-clause detection on held-out OPP-115 policies, plus the negation minimal pairs, with the polarity check on and off | `build_privacy_corpus.py` must have run |
| `python evaluate.py classification [variant] [verdict]` | Accuracy, per-class precision/recall/F1, retrieval recall@3, confidence distribution on the LexGLUE UNFAIR-ToS test set (1,607 expert-annotated sentences) | Internet on first run (cached afterwards) |
| `python evaluate.py generation` | Writes a blinded CSV of grounded vs ungrounded FLAN-T5 outputs for 30 clauses | — |
| `python evaluate.py score-generation r1.csv r2.csv` | ROUGE-L, BERTScore, mean human rating, hallucination rate and reduction, inter-rater kappa | Each rater's filled-in copy of the CSV; `pip install bert-score` for BERTScore |
| `python evaluate.py latency` | Median `/process` time per page and estimated manual reading time | Page text saved as `data/pages/<site>.txt` |
| `python evaluate.py memory` | Model weight sizes, FAISS index size, peak process RSS | — |

UNFAIR-ToS marks unfair sentences only, so for the binary Algorithm 1 fair sentences are counted as `ethical`. Recall@3 counts a hit when one of the top-3 retrieved clauses shares an UNFAIR-ToS category with the unfair test sentence.

### Measured comparison

LexGLUE UNFAIR-ToS test split, 1,607 sentences the system never saw (`python evaluate.py compare`):

| Retrieval  | Verdict     | Accuracy | Macro F1 | Unfair P | Unfair R | Unfair F1 | Unclear | s/1k   |
|--------------|---------------|------------|------------|------------|------------|-------------|-----------|----------|
| nearest    | similarity  | 0.883    | 0.750    | 0.469    | 0.715    | 0.567     | 0       | 1.9    |
| vote       | similarity  | 0.928    | 0.813    | 0.665    | 0.669    | 0.667     | 0       | 1.9    |
| hybrid     | similarity  | 0.944    | 0.851    | 0.747    | 0.721    | 0.734     | 0       | 7.9    |
| gated      | similarity  | 0.702    | 0.651    | 0.757    | 0.326    | 0.455     | 438     | 2.4    |
| rerank     | similarity  | 0.930    | 0.832    | 0.641    | 0.779    | 0.703     | 0       | 142.3  |
| hybrid     | legalbert   | 0.960    | 0.893    | 0.833    | 0.785    | 0.808     | 0       | 16.4   |

The `legalbert` row above is Legal-BERT **small**, trained on the eight UNFAIR-ToS categories with class weighting and train+validation data. An earlier Legal-BERT **base** model, trained as a plain binary classifier on the train split only, scored 0.966 accuracy / 0.910 macro-F1 / **0.839** unfair-F1 — better than the small model despite the simpler recipe. The two runs changed both the model size and the training recipe, so neither effect is isolated; `models/legalbert-binary-backup/` keeps the base model for comparison.

Contract-fairness only. Adding the privacy corpus costs every variant here (hybrid was 0.784 before) and buys privacy detection in exchange:

| Approach | Contract F1 | Privacy F1 |
|---|---|---|
| UNFAIR-ToS corpus only | 0.784 | 0.018 |
| OPP-115 corpus only | 0.056 | 0.853 |
| **Both merged** *(current default)* | **0.734** | **0.836** |
| Route by domain, then decide | 0.607 | 0.787 |

Routing — pick whichever corpus matches best, then decide inside it — was measured and is worse than simply merging, because the router picks the wrong domain too often.

| Polarity check | Contract F1 | Privacy F1 | Polarity probes (14) |
|---|---|---|---|
| Off | 0.745 | 0.836 | 71% |
| **On** *(default)* | 0.734 | 0.836 | **86%** |

Three variants of the polarity rule were measured and rejected: flipping on any negation mismatch (contract F1 0.588), counting "without"/"no" as negation cues (flips "modify these terms **without** notice" to ethical), and letting the flipped best match decide instead of vote (contract F1 0.591).

Fair sentences count as `ethical`, so "Unfair P/R/F1" is the interesting column: it measures catching genuinely unfair clauses without crying wolf. `gated` scores low by construction — the dataset has no "unclear" class, so each of its 386 abstentions counts as wrong; it trades recall for not labelling boilerplate. `rerank` is 60× slower than `hybrid` for a worse score, so on this corpus it is not worth it.

**Known limitations**:
- Algorithm 1 has no neutral label, so everything that isn't close to an unfair clause is reported as ethical.
- Combining the two corpora costs accuracy in each domain against a specialist corpus (contract F1 0.784 → 0.734, privacy F1 0.853 → 0.836) in exchange for covering both.
- The polarity check only catches minimal pairs. A clause that states the opposite in different words still slips through.
- Only the first 3,000 characters of a page are analyzed, and only the first 15 sentences are classified.
- `flan-t5-base` summaries are mostly extractive — it tends to restate one sentence from the terms or a retrieved clause rather than paraphrase.

## Legal-BERT

`train_legalbert.py` fine-tunes `nlpaueb/legal-bert-base-uncased` to separate unfair clauses from fair ones, on the same training split the reference corpus is built from, and saves the best epoch by validation macro-F1 to `models/legalbert-unfair-tos/`.

```bash
cd backend
./venv/bin/python train_legalbert.py       # uses the Mac GPU (MPS) when available
./venv/bin/python evaluate.py compare      # side-by-side table against clause matching
```

It is additive: nothing in the existing pipeline changes, the model loads only when the `legalbert` verdict is selected, and retrieval still supplies the "similar to" evidence shown in the popup.

## Explainability

Clause matching explains itself: the decision trace returned with every point names the clauses consulted, their scores, the vote and the margin to the opposite verdict. That is the mechanism, not an approximation of it.

Legal-BERT is the one trained model, so it is explained after the fact with LIME and SHAP. Offline only (~10 s per clause), via `explain_tokens.py`.

```bash
cd backend
./venv/bin/python explain_tokens.py 40 400      # per-clause attributions + aggregate
./venv/bin/python explain_tokens.py control     # does the verdict survive without legal words?
./venv/bin/python explain_tokens.py agreement   # do LIME and SHAP name the same words?
```

### What the model attends to

Across 40 unfair clauses, 15 of the 25 most consistent words carry legal meaning. The strongest by weight:

| Word | Clauses | Mean weight |
|---|---|---|
| terminate | 14 | +0.185 |
| may | 14 | +0.121 |
| discretion | 9 | +0.108 |
| right | 6 | +0.114 |
| change | 5 | +0.209 |
| suspend | 5 | +0.181 |

### Control: is it keying on wording habits?

Function words appear in more clauses than legal terms ("to" in 25 of 40), so the aggregate alone cannot
separate "the model understands the clause" from "the model recognises corporate hedging". Stripping each
clause two ways settles it, over the 95 unfair clauses the model catches:

| Version | Mean unfair probability | Still called unfair |
|---|---|---|
| Original clause | 0.759 | 100% |
| Grammar only, legal words removed | 0.068 | 0% |
| Legal words only, grammar removed | 0.246 | 12% |

Removing the legal vocabulary costs 91% of the model's confidence and flips every verdict, so the
function words ride along with the legal terms rather than driving the decision. Content words alone are
also not enough, which is the expected behaviour for a transformer and argues against a bag-of-words shortcut
in either direction.

### LIME against SHAP

Over 25 clauses, the two methods name the same word **38.4%** of the time, and there is no clause
where they agree on nothing. The words they agree on are the legal ones - terminate (10), suspend (3), laws (3), discretion (2), modify (2) - while the
disagreement is in the low-signal filler each method pads its ranking with. LIME fits a local linear
surrogate and SHAP computes coalition values, so they weigh weak tokens differently.

Reported because two approximations that disagree cannot both be trusted; where a faithful explanation exists
(the retrieval trace) it is preferable to either.

## Development Notes

- Reference clauses are embedded once at backend startup into two FAISS HNSW indexes (ethical / non-ethical) and kept in memory.
- FAISS search is pinned to one thread: faiss and PyTorch each ship their own libomp on macOS, and multi-threaded faiss search deadlocks.
- The extension communicates with the backend over local HTTP only — no page content leaves the machine.
- The reranker, Legal-BERT, the BM25 index and every non-default summary model load on first use, so startup stays the same for the default pipeline. Switching summary model unloads the previous one, so the first analysis after a switch pays that model's load time.
- Generation is greedy with a 40-token cap. Beam search doubled the time for no measurable gain.
- A vote-weight bias toward unfair clauses was swept on the validation split (1.0-1.5) and gave no improvement, so votes are unweighted.

## Future Enhancements

- Swap `flan-t5-base` for a larger instruction-tuned model to improve summary quality.
- Multi-language support.
- Export analysis reports as PDF.
- User-configurable filtering/preferences.
