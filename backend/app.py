# ============================================
# MACOS MULTIPROCESSING FIX (MUST BE FIRST)
# ============================================
import multiprocessing
import os

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# torch + numpy on macOS both bundle libomp; without this the process aborts on import
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# ============================================
# IMPORTS
# ============================================
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import threading
import warnings
import time
import re
from collections import Counter, defaultdict
import hashlib
import json
import math
import tarfile
import urllib.request

warnings.filterwarnings('ignore')

app = Flask(__name__)
# Local-first: only the browser extension may call the backend from a browser.
CORS(app, origins=r"chrome-extension://.*")

# ============================================
# PIPELINE SETTINGS (Section III.D optimisations)
# ============================================
MAX_INPUT_CHARS = 3000   # input text truncated before processing
MAX_SENTENCES = 15       # sentences classified per analysis
CHUNK_TOKENS = 512       # chunk length in embedding-model tokens, [CLS]/[SEP] included
CHUNK_OVERLAP = 50       # tokens shared by consecutive chunks
TOP_K = 3                # reference clauses retrieved per chunk, and chunks passed to the LLM
VOTE_K = 5               # neighbours that vote in the vote/hybrid/rerank variants
RERANK_K = 20            # candidates re-scored by the cross-encoder before voting
GATE_MARGIN = 0.05       # similarity gap below which "gated" answers "unclear"
GATE_FLOOR = 0.45        # similarity below which "gated" answers "unclear"
POLARITY_CHECK = True    # flip a near-duplicate match whose negation disagrees (ablation switch)
POLARITY_FLOOR = 0.60    # similarity below which a match is not trusted enough to flip
POLARITY_OVERLAP = 0.65  # share of the shorter clause's words the two must share to count as a minimal pair

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
LEGALBERT_DIR = os.path.join(MODELS_DIR, "legalbert-unfair-tos")

# ============================================
# SELECTABLE PIPELINES (the extension's dropdowns)
# ============================================
# Each entry carries the one line the popup shows under the dropdown, so the
# wording lives in one place instead of being duplicated in the extension.
RAG_VARIANTS = {
    "nearest": {"name": "Nearest clause",
                "why": "The method in the report: the single most similar clause decides the verdict."},
    "vote":    {"name": "5-clause vote",
                "why": "Five nearest clauses vote, so one odd match cannot raise a false alarm on its own."},
    "hybrid":  {"name": "Keyword + meaning",
                "why": "Adds keyword search, so exact legal terms like arbitration or indemnify carry weight."},
    "gated":   {"name": "Vote, with unclear",
                "why": "Leaves boilerplate such as addresses unlabelled instead of forcing a verdict."},
    "rerank":  {"name": "Cross-encoder rerank",
                "why": "Re-reads 20 candidates in detail: usually the most accurate, and the slowest."},
}

GENERATORS = {
    "templates":     {"name": "Plain-language templates",
                      "why": "Instant and never invents anything: fixed wording for each clause type."},
    "flan-t5-base":  {"name": "FLAN-T5 base", "hf": "google/flan-t5-base",
                      "why": "Mostly repeats the clause back; kept for comparison with the templates."},
    "flan-t5-small": {"name": "FLAN-T5 small", "hf": "google/flan-t5-small",
                      "why": "About three times faster and a quarter of the memory; blunter wording."},
    "qwen2.5-0.5b":  {"name": "Qwen2.5 0.5B Instruct", "hf": "Qwen/Qwen2.5-0.5B-Instruct", "causal": True,
                      "why": "Writes a real explanation of the worst clause, but adds about 6 seconds."},
    "none":          {"name": "No summary",
                      "why": "Skips the written summary for the fastest possible analysis."},
}

CLASSIFIERS = {
    "similarity": {"name": "Clause matching",
                   "why": "The method in the report: the verdict comes from the most similar reference clause."},
    "legalbert":  {"name": "Legal-BERT (trained)",
                   "why": "A model trained on legal text; catches more unfair clauses, and needs the training step first."},
    "legalbert-knn": {"name": "Legal-BERT + corpus",
                      "why": "Blends the trained model with its nearest reference clauses; best on contract terms."},
    "legalbert-privacy": {"name": "Legal-BERT + privacy clauses",
                          "why": "Best overall: the trained model for contract terms, clause matching for privacy practices."},
}

KNN_K = 10               # reference clauses consulted in Legal-BERT's own embedding space
KNN_WEIGHT = 0.3         # share of the verdict taken from those neighbours; tuned on the validation split

# hybrid retrieval measures best of the untrained options (unfair-F1 0.784 vs 0.658
# for nearest); "nearest" is still selectable as the report's exact method.
DEFAULT_RAG, DEFAULT_GENERATOR, DEFAULT_CLASSIFIER = "hybrid", "templates", "similarity"

# ============================================
# MODEL LOADING (GLOBAL - LOAD ONCE)
# ============================================
print("=" * 50)
print("LOADING MODELS (This takes 1-2 minutes on first run)...")
print("=" * 50)
start_time = time.time()

# Embedding model: 384-dimensional sentence embeddings
semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Default is 256 tokens; raise to 512 so a full chunk is embedded, not truncated
# (BERT position embeddings go up to 512).
semantic_model.max_seq_length = CHUNK_TOKENS

# Generators are loaded on demand (see get_generator); the default one is loaded now.
_generators = {}


def get_generator(key):
    """Tokenizer and model for one summary model. Only the last one stays in memory."""
    if "hf" not in GENERATORS[key]:
        raise ValueError(f"'{key}' writes its summary without a model - nothing to load")
    if key not in _generators:
        spec = GENERATORS[key]
        loader = AutoModelForCausalLM if spec.get("causal") else AutoModelForSeq2SeqLM
        print(f"Loading summary model: {spec['name']}...")
        _generators.clear()  # each generator is 300MB-1GB; keep one
        _generators[key] = (AutoTokenizer.from_pretrained(spec["hf"]), loader.from_pretrained(spec["hf"]))
    return _generators[key]


# "templates" and "none" need no model, so there is nothing to load at startup
tokenizer = model = None
if "hf" in GENERATORS[DEFAULT_GENERATOR]:
    tokenizer, model = get_generator(DEFAULT_GENERATOR)

# Limit threads to prevent CPU oversubscription
torch.set_num_threads(4)

load_time = round(time.time() - start_time, 2)
print(f"✅ Models loaded in {load_time} seconds")
print("=" * 50)

# ============================================
# REFERENCE CORPUS: LexGLUE UNFAIR-ToS train split
# ============================================
# Sentences from real Terms of Service annotated by legal experts (Chalkidis et
# al., 2022; CC BY 4.0). Sentences flagged unfair in any of 8 categories form the
# non-ethical corpus; unflagged sentences form the ethical corpus, so "ethical"
# here means "not flagged unfair", not "protective of the user".
# ponytail: stand-in corpus until a dedicated ethical/non-ethical clause dataset is chosen.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_unfair_tos(split):
    """UNFAIR-ToS split ("train", "val" or "test") as [{"text", "labels"}]; the official release is downloaded once into data/."""
    path = os.path.join(DATA_DIR, "unfair_tos.tar.gz")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve("https://zenodo.org/record/5532997/files/unfair_tos.tar.gz", path)
    with tarfile.open(path) as archive:
        rows = [json.loads(line) for line in archive.extractfile("unfair_tos.jsonl")]
    return [{"text": r["text"].strip(), "labels": r["labels"]} for r in rows if r["data_type"] == split]


# Exact duplicates of test sentences (shared boilerplate across companies) are left
# out so evaluate.py never scores a sentence the corpus already contains.
_test_sentences = {row["text"] for row in load_unfair_tos("test")}
CLAUSE_LABELS = {}  # clause text -> set of UNFAIR-ToS categories (empty = not unfair)
for row in load_unfair_tos("train"):
    if row["text"] and row["text"] not in _test_sentences:
        CLAUSE_LABELS.setdefault(row["text"], set()).update(row["labels"])

def is_clause(text):
    """Drop headings and fragments ("therefore , you agree not to :") from the corpus."""
    return len(text) >= 40 and len(text.split()) >= 8


# The validation split's unfair sentences are otherwise unused by the corpus. 80% join
# it; the remaining 20% stay out so test_classification still has unseen sentences.
VALIDATION_UNFAIR = sorted({r["text"] for r in load_unfair_tos("val") if r["labels"]})
# a list, and iterated in order: a set iterates differently in every process, which
# would reorder the corpus run to run and misalign the cached Legal-BERT vectors
CORPUS_VALIDATION = VALIDATION_UNFAIR[:int(len(VALIDATION_UNFAIR) * 0.8)]
for text in CORPUS_VALIDATION:
    if text and text not in _test_sentences:
        CLAUSE_LABELS.setdefault(text, set()).add("Unfair (validation split)")

ETHICAL_CLAUSES = [text for text, labels in CLAUSE_LABELS.items() if not labels and is_clause(text)]
NON_ETHICAL_CLAUSES = [text for text, labels in CLAUSE_LABELS.items() if labels]

# UNFAIR-ToS covers contract fairness only - it has no category for selling data,
# tracking or retention, so those clauses were invisible. build_privacy_corpus.py
# derives this second corpus from OPP-115's annotated privacy policies.
PRIVACY_CORPUS = os.path.join(DATA_DIR, "privacy_corpus.json")
if os.path.exists(PRIVACY_CORPUS):
    with open(PRIVACY_CORPUS) as f:
        privacy = json.load(f)
    for clause in privacy["harmful"]:
        if clause["text"] not in CLAUSE_LABELS:
            CLAUSE_LABELS[clause["text"]] = {clause["label"]}
            NON_ETHICAL_CLAUSES.append(clause["text"])
    for clause in privacy["protective"]:
        if clause["text"] not in CLAUSE_LABELS:
            CLAUSE_LABELS[clause["text"]] = set()
            ETHICAL_CLAUSES.append(clause["text"])
    print(f"Privacy corpus: +{len(privacy['harmful'])} harmful, +{len(privacy['protective'])} protective (OPP-115)")

print(f"Reference corpus: {len(ETHICAL_CLAUSES)} ethical, {len(NON_ETHICAL_CLAUSES)} non-ethical clauses")

# ============================================
# FAISS INDEXES (built ONCE at startup)
# ============================================
def embed(texts):
    """L2-normalised embeddings, so inner product == cosine similarity."""
    return semantic_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype('float32')


CLAUSE_VECTORS = {}  # clause -> embedding, for re-scoring fused/reranked candidates


def build_faiss_index(clauses):
    # HNSW graph index: sub-linear nearest-neighbour search over cosine similarity.
    vectors = embed(clauses)
    CLAUSE_VECTORS.update(zip(clauses, vectors))
    faiss.omp_set_num_threads(1)  # parallel HNSW construction deadlocks too - see search()
    index = faiss.IndexHNSWFlat(vectors.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 64  # wider graph search: near-exact recall at a few thousand clauses
    index.add(vectors)
    return index


ethical_index = build_faiss_index(ETHICAL_CLAUSES)
non_ethical_index = build_faiss_index(NON_ETHICAL_CLAUSES)


def search(index, clauses, vectors, k):
    """For each vector, the k nearest (clause, cosine_similarity) pairs, best first."""
    # faiss and torch each ship libomp on macOS; multi-threaded faiss search then
    # deadlocks. The OpenMP thread count is per calling thread and Flask serves each
    # request on a new thread, so set it here. HNSW keeps single-threaded search fast.
    faiss.omp_set_num_threads(1)
    scores, ids = index.search(vectors, k)
    return [
        [(clauses[i], float(s)) for s, i in zip(row_scores, row_ids) if i != -1]
        for row_scores, row_ids in zip(scores, ids)
    ]


def retrieve_references(vectors, k=TOP_K):
    """For each vector, the top-k reference clauses across both corpora: [(clause, label, similarity)]."""
    ethical = search(ethical_index, ETHICAL_CLAUSES, vectors, k)
    non_ethical = search(non_ethical_index, NON_ETHICAL_CLAUSES, vectors, k)
    return [
        sorted([(c, 'ethical', s) for c, s in e] + [(c, 'non_ethical', s) for c, s in n],
               key=lambda ref: ref[2], reverse=True)[:k]
        for e, n in zip(ethical, non_ethical)
    ]

# ============================================
# RETRIEVAL VARIANTS (the "Retrieval" dropdown)
# ============================================
def tokenise(text):
    return re.findall(r"[a-z0-9']+", text.lower())


_bm25 = None


def get_bm25():
    """Tiny BM25 index over the reference corpus, built on first keyword search."""
    global _bm25
    if _bm25 is None:
        docs = [(c, 'ethical') for c in ETHICAL_CLAUSES] + [(c, 'non_ethical') for c in NON_ETHICAL_CLAUSES]
        tokenised = [tokenise(clause) for clause, _ in docs]
        document_frequency = Counter(term for terms in tokenised for term in set(terms))
        total = len(docs)
        idf = {t: math.log(1 + (total - df + 0.5) / (df + 0.5)) for t, df in document_frequency.items()}
        postings = defaultdict(list)
        for i, terms in enumerate(tokenised):
            for term, frequency in Counter(terms).items():
                postings[term].append((i, frequency))
        lengths = [len(terms) for terms in tokenised]
        _bm25 = (docs, postings, idf, lengths, sum(lengths) / total)
    return _bm25


def bm25_search(query, k):
    """Top-k (clause, label) by BM25, so exact legal wording counts as well as meaning."""
    docs, postings, idf, lengths, average_length = get_bm25()
    k1, b = 1.5, 0.75
    scores = defaultdict(float)
    for term in set(tokenise(query)):
        for i, frequency in postings.get(term, ()):
            denominator = frequency + k1 * (1 - b + b * lengths[i] / average_length)
            scores[i] += idf[term] * frequency * (k1 + 1) / denominator
    ranked = sorted(scores, key=scores.get, reverse=True)[:k]
    return [docs[i] for i in ranked]


def fuse(dense, sparse, vector, k=VOTE_K):
    """
    Reciprocal rank fusion of the embedding and keyword result lists. Scores are
    re-read from the embeddings afterwards so confidence stays on one scale.
    """
    points = defaultdict(float)
    labels = {}
    for ranked in (dense, sparse):
        for rank, candidate in enumerate(ranked):
            clause, label = candidate[0], candidate[1]
            points[clause] += 1 / (60 + rank + 1)
            labels[clause] = label
    best = sorted(points, key=points.get, reverse=True)[:k]
    return [(clause, labels[clause], float(vector @ CLAUSE_VECTORS[clause])) for clause in best]


_reranker = None


def get_reranker():
    """Cross-encoder that scores a sentence and a clause together instead of separately."""
    global _reranker
    if _reranker is None:
        name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        print("Loading reranker...")
        _reranker = (AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name))
    return _reranker


def rerank(sentence, candidates, k=VOTE_K):
    tokeniser, cross_encoder = get_reranker()
    pairs = tokeniser([sentence] * len(candidates), [c[0] for c in candidates],
                      padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        scores = cross_encoder(**pairs).logits.squeeze(-1).sigmoid().tolist()
    ordered = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)[:k]
    return [(clause, label, score) for (clause, label, _), score in ordered]


# Sentence embeddings score "we never sell your data" and "we may sell your data"
# at 0.70 similarity, and "can" vs "cannot" at 0.91: the encoder does not represent
# negation. When a clause matches a reference closely but the two disagree about
# negation, the clause means the opposite of what it matched, so the label flips.
# Only cues that negate the verb. "without", "no" and "none" are excluded on purpose:
# in legal text they usually mark the absence of a protection ("modify without notice",
# "no liability"), which is the unfair practice itself, not a denial of it.
NEGATION = re.compile(r"\b(?:not|never|cannot|can't|won't|wont|doesn't|don't|didn't|"
                      r"isn't|aren't|shan't|neither|nor)\b|n't\b", re.IGNORECASE)
OPPOSITE = {"ethical": "non_ethical", "non_ethical": "ethical"}


def is_negated(text):
    return bool(NEGATION.search(text))


def words_without_negation(text):
    return {w for w in tokenise(NEGATION.sub(" ", text))}


def is_minimal_pair(sentence, clause):
    """
    True when the two say the same thing apart from a negation. Legal text negates
    constantly ("we are not liable" IS the unfair practice), so a bare negation
    mismatch means nothing; only a near-identical wording makes the flip safe.
    """
    a, b = words_without_negation(sentence), words_without_negation(clause)
    if not a or not b:
        return False
    # containment, not Jaccard: a real clause is longer than the sentence it matches,
    # so the test is whether the shorter one is almost entirely inside the longer
    return len(a & b) / min(len(a), len(b)) >= POLARITY_OVERLAP


def apply_polarity(sentence, candidates):
    """
    Flip a matched clause that is the same statement with the negation reversed.
    The flipped clause then votes with the rest. Letting it decide on its own was
    measured and is worse: contract F1 drops from 0.734 to 0.591, because in
    contract text a single near-duplicate match is often the wrong evidence.
    """
    if not POLARITY_CHECK:
        return candidates
    negated = is_negated(sentence)
    flipped = []
    for clause, label, score in candidates:
        if (score >= POLARITY_FLOOR and is_negated(clause) != negated
                and is_minimal_pair(sentence, clause)):
            label = OPPOSITE[label]
        flipped.append((clause, label, score))
    return flipped


def vote(candidates):
    """
    Similarity-weighted vote over the candidates. The winner's clause is returned
    with it; its cosine similarity (not the vote share) is reported as confidence,
    so the number means the same thing in every variant.
    """
    weights = defaultdict(float)
    for _, label, score in candidates:
        # a bias multiplier toward unfair was swept on validation (1.0-1.5): no gain, so plain weights
        weights[label] += max(score, 0.0)
    label = max(weights, key=weights.get)
    clause = next(c for c, candidate_label, _ in candidates if candidate_label == label)
    return label, clause


def retrieve_candidates(sentences, vectors, rag):
    """Candidate clauses per sentence for the chosen retrieval variant."""
    k = 1 if rag == "nearest" else RERANK_K if rag == "rerank" else VOTE_K
    candidates = retrieve_references(vectors, k)
    if rag == "hybrid":
        candidates = [fuse(dense, bm25_search(sentence, VOTE_K), vector)
                      for sentence, vector, dense in zip(sentences, vectors, candidates)]
    elif rag == "rerank":
        candidates = [rerank(sentence, dense) for sentence, dense in zip(sentences, candidates)]
    return candidates

# ============================================
# ALGORITHM 1: FAST SEMANTIC CLAUSE CLASSIFICATION (Section III.C)
# ============================================
_legalbert = None


def get_legalbert():
    """Legal-BERT fine-tuned on UNFAIR-ToS by train_legalbert.py."""
    global _legalbert
    if _legalbert is None:
        if not os.path.isdir(LEGALBERT_DIR):
            raise FileNotFoundError("Legal-BERT is not trained yet - run: python train_legalbert.py")
        print("Loading Legal-BERT classifier...")
        _legalbert = (AutoTokenizer.from_pretrained(LEGALBERT_DIR),
                      AutoModelForSequenceClassification.from_pretrained(LEGALBERT_DIR).eval())
    return _legalbert


def legalbert_labels(sentences):
    """
    (label, probability, category) per sentence from the trained classifier.
    The model scores the eight UNFAIR-ToS categories; a sentence is unfair when
    any of them fires, and the strongest one names the kind of unfairness.
    A two-class model trained by an older version of the script still works.
    """
    tokeniser, classifier = get_legalbert()
    multi_label = classifier.config.num_labels > 2
    results = []
    for start in range(0, len(sentences), 16):
        batch = tokeniser(sentences[start:start + 16], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = classifier(**batch).logits
        if multi_label:
            probabilities, indices = logits.sigmoid().max(-1)
            for unfair, index in zip(probabilities.tolist(), indices.tolist()):
                category = classifier.config.id2label[index]
                results.append(('non_ethical', unfair, category) if unfair >= 0.5
                               else ('ethical', 1 - unfair, None))
        else:
            for row in logits.softmax(-1):
                unfair = float(row[1])
                results.append(('non_ethical' if unfair >= 0.5 else 'ethical', max(unfair, 1 - unfair), None))
    return results


_legalbert_store = None
_store_lock = threading.Lock()


def legalbert_embed(texts, batch_size=32):
    """Mean-pooled encoder output from the fine-tuned model - a space that, unlike
    MiniLM's, has been trained to separate unfair clauses from fair ones."""
    tokeniser, classifier = get_legalbert()
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = tokeniser(texts[start:start + batch_size], padding=True, truncation=True,
                          max_length=256, return_tensors="pt")
        with torch.no_grad():
            hidden = classifier.bert(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        vectors.append(torch.nn.functional.normalize(pooled, dim=-1))
    return torch.cat(vectors).numpy().astype('float32')


def get_legalbert_datastore():
    """The reference corpus embedded in Legal-BERT's space, with its labels."""
    global _legalbert_store
    with _store_lock:   # Flask serves requests on threads; without this both build it
        if _legalbert_store is not None:
            return _legalbert_store
        clauses = ETHICAL_CLAUSES + NON_ETHICAL_CLAUSES
        labels = np.array([0.0] * len(ETHICAL_CLAUSES) + [1.0] * len(NON_ETHICAL_CLAUSES), dtype='float32')
        # embedding 6,600 clauses takes well over a minute, so keep it on disk and
        # rebuild only when the corpus itself changes
        cache = os.path.join(DATA_DIR, "legalbert_vectors.npz")
        # hashlib, not hash(): Python randomises string hashing per process, so a
        # hash()-based fingerprint never matches after a restart
        fingerprint = hashlib.sha256("|".join(clauses).encode()).hexdigest()
        if os.path.exists(cache):
            stored = np.load(cache, allow_pickle=False)
            if str(stored["fingerprint"]) == fingerprint:
                _legalbert_store = (stored["vectors"], labels, clauses)
                return _legalbert_store
        print(f"Embedding {len(clauses)} reference clauses for Legal-BERT k-NN (one-off, then cached)...")
        vectors = legalbert_embed(clauses)
        np.savez(cache, vectors=vectors, fingerprint=np.array(fingerprint))
        _legalbert_store = (vectors, labels, clauses)
        return _legalbert_store


def legalbert_knn_labels(sentences, k=None, weight=None):
    """
    Retrieval-augmented verdict: the model's own probability blended with a
    similarity-weighted vote of its nearest reference clauses. This is what lets
    corpus additions (the privacy clauses) reach a model whose training data
    never contained them.
    """
    # read the globals here, not as default arguments: defaults bind once at import
    # and silently ignore any later change (which made a whole sweep look flat)
    k = KNN_K if k is None else k
    weight = KNN_WEIGHT if weight is None else weight
    vectors, labels, clauses = get_legalbert_datastore()
    queries = legalbert_embed(sentences)
    similarity = queries @ vectors.T
    results = []
    for (label, confidence, category), row in zip(legalbert_labels(sentences), similarity):
        top = np.argpartition(-row, k)[:k]
        weights = np.exp((row[top] - row[top].max()) * 10)   # sharpen: near neighbours dominate
        neighbour_unfair = float((weights * labels[top]).sum() / weights.sum())
        model_unfair = confidence if label == 'non_ethical' else 1 - confidence
        unfair = (1 - weight) * model_unfair + weight * neighbour_unfair
        nearest = clauses[int(top[np.argmax(row[top])])]
        results.append(('non_ethical' if unfair >= 0.5 else 'ethical',
                        max(unfair, 1 - unfair), category if unfair >= 0.5 else None, nearest))
    return results


def build_trace(sentence, matches, label, flipped):
    """
    Why this verdict: the clauses consulted, the vote they cast, and how close the
    opposite answer was. Returned to the popup so a reader can audit any flag.
    """
    weights = defaultdict(float)
    for _, candidate_label, score in matches:
        weights[candidate_label] += max(score, 0.0)
    opposite = next(((c, s) for c, candidate_label, s in matches if candidate_label != label), None)
    best_same = next((s for _, candidate_label, s in matches if candidate_label == label), 0.0)
    return {
        "evidence": [{"clause": clause, "label": candidate_label, "score": round(score, 2),
                      "category": next((str(c) for c in CLAUSE_LABELS.get(clause, ())), None)}
                     for clause, candidate_label, score in matches[:TOP_K]],
        "vote": {side: round(total, 2) for side, total in weights.items()},
        "closest_opposite": {"clause": opposite[0], "score": round(opposite[1], 2)} if opposite else None,
        "margin": round(best_same - (opposite[1] if opposite else 0.0), 2),
        "negation_flip": flipped,
    }


def classify_clauses(sentences, rag=DEFAULT_RAG, classifier=DEFAULT_CLASSIFIER, with_trace=False):
    """
    Algorithm 1, batched. With the defaults this is exactly the report's method:
    the label of the single most similar reference clause, with its cosine
    similarity as the confidence. The other variants change how the retrieved
    clauses are turned into a verdict; `classifier="legalbert"` takes the verdict
    from the trained model instead and keeps retrieval only for the evidence.
    Returns one (label, confidence, matched_clause, category) per sentence;
    category is the kind of unfairness when Legal-BERT named one, otherwise None.
    """
    if not sentences:
        return []
    vectors = embed(sentences)
    candidates = retrieve_candidates(sentences, vectors, rag)
    flips = [False] * len(sentences)   # replaced below on the clause-matching path

    def finish(results):
        if not with_trace:
            return results
        return [result + (build_trace(sentence, matches, result[0], flip),)
                for result, sentence, matches, flip in zip(results, sentences, candidates, flips)]

    if classifier == "legalbert":
        return finish([(label, round(confidence, 2), matches[0][0] if matches else None, category)
                       for (label, confidence, category), matches in zip(legalbert_labels(sentences), candidates)])

    if classifier == "legalbert-knn":
        return finish([(label, round(confidence, 2), nearest, category)
                       for label, confidence, category, nearest in legalbert_knn_labels(sentences)])

    if classifier == "legalbert-privacy":
        # Legal-BERT knows contract fairness but was never trained on privacy practices,
        # and k-NN in its own space cannot supply the missing concept (privacy F1 0.000
        # at the tuned weight). Clause matching can, so it overrules - but only on a
        # clause the privacy corpus labelled, never on contract text.
        verdicts = legalbert_knn_labels(sentences)
        results = []
        for (label, confidence, category, nearest), matches in zip(verdicts, candidates):
            clause, matched_label, score = matches[0]
            privacy_hit = matched_label == 'non_ethical' and any(
                str(c).startswith("Privacy:") for c in CLAUSE_LABELS.get(clause, ()))
            if privacy_hit and label == 'ethical':
                label, confidence, nearest = 'non_ethical', score, clause
                category = next((c for c in CLAUSE_LABELS.get(clause, ()) if str(c).startswith("Privacy:")), None)
            results.append((label, round(confidence, 2), nearest, category))
        return finish(results)

    before_polarity = candidates
    candidates = [apply_polarity(sentence, matches) for sentence, matches in zip(sentences, candidates)]
    flips[:] = [[c[1] for c in before] != [c[1] for c in after]
                for before, after in zip(before_polarity, candidates)]

    results = []
    for vector, matches in zip(vectors, candidates):
        if rag == "nearest":
            clause, label, score = matches[0]
            results.append((label, round(score, 2), clause, None))
            continue
        label, clause = vote(matches)
        results.append((label, round(max(0.0, float(vector @ CLAUSE_VECTORS[clause])), 2), clause, None))

    if rag == "gated":
        # A vote is usually unanimous, so the gate compares the best clause of each
        # label instead: too weak a match, or too close a call, answers "unclear".
        best_ethical = search(ethical_index, ETHICAL_CLAUSES, vectors, 1)
        best_non_ethical = search(non_ethical_index, NON_ETHICAL_CLAUSES, vectors, 1)
        for i, ([(_, ethical_score)], [(_, non_ethical_score)]) in enumerate(zip(best_ethical, best_non_ethical)):
            top = max(ethical_score, non_ethical_score)
            if top < GATE_FLOOR or abs(ethical_score - non_ethical_score) < GATE_MARGIN:
                results[i] = ('unclear', round(top, 2), results[i][2], None)
    return finish(results)

# ============================================
# CHUNKING + GROUNDED GENERATION (Section III.A / III.C)
# ============================================
def clean_page_text(text):
    """
    Keep the lines that read like prose. A page's visible text starts with menus,
    buttons and headings; without this the first 3,000 characters analysed are
    navigation rather than terms.
    """
    kept, seen = [], set()
    for line in text.splitlines():
        line = line.strip()
        # ponytail: length + a space is enough to separate clauses from nav items
        if len(line) < 40 or " " not in line or line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return "\n".join(kept)


def chunk_text(text, size=CHUNK_TOKENS - 2, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of `size` tokens (+[CLS]/[SEP] = CHUNK_TOKENS)."""
    offsets = semantic_model.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    chunks = []
    for start in range(0, len(offsets), size - overlap):
        window = offsets[start:start + size]
        chunks.append(text[window[0][0]:window[-1][1]])
        if start + size >= len(offsets):
            break
    return chunks


# What each clause type means for the reader. The small local models cannot write
# this reliably - flan-t5-base repeats the clause back and Qwen pads with preamble -
# but the clause type is already known from retrieval or Legal-BERT, so the
# explanation can be stated exactly rather than generated.
EXPLANATIONS = {
    "Limitation of liability": "the company will not compensate you if something goes wrong, including a data breach",
    "Unilateral termination": "the company can close your account whenever it likes, without having to justify it",
    "Unilateral change": "the company can change these terms at any time, and carrying on using the service counts as agreeing",
    "Content removal": "the company can delete things you post without warning you first",
    "Contract by using": "simply using the service binds you to these terms, whether or not you read them",
    "Choice of law": "any dispute is judged under a legal system the company picked, which may not be yours",
    "Jurisdiction": "you would have to sue in courts the company chose, which may be far from where you live",
    "Arbitration": "you give up the right to take the company to court, including as part of a group claim",
    "Privacy: shares data with third parties for advertising": "your personal data is passed to other companies so they can advertise to you",
    "Privacy: collects data for advertising": "what you do here is collected to target advertising at you",
    "Privacy: retains data indefinitely": "your data is kept for as long as the company likes, even after you leave",
    "Privacy: ignores Do Not Track": "your browser's Do Not Track setting is ignored",
    "Privacy: no access to your data": "you cannot see or retrieve the data held about you",
    "Privacy: states it does not share": "the company states it does not share your personal data",
    "Privacy: states it does not collect": "the company states it does not collect this data",
    "Privacy: states a retention period": "the company says how long it keeps your data",
    "Privacy: offers a choice about data use": "you are given a way to opt in or out of this data use",
    "Privacy: describes a security measure": "the company describes how it protects your data",
    "Privacy: lets you access or delete your data": "you can see or delete the data held about you",
}


def explain(point, label):
    """One plain sentence for a classified clause, from its type rather than a model."""
    category = point.get("category") or next(
        (c for c in CLAUSE_LABELS.get(point.get("matched_clause"), ()) if c in EXPLANATIONS), None)
    opening = "This clause means" if label == 'non_ethical' else "In your favour:"
    if category in EXPLANATIONS:
        return f"{opening} {EXPLANATIONS[category]}."
    return ("This clause looks similar to terms that work against the user."
            if label == 'non_ethical' else "This clause looks similar to terms that protect the user.")


def generate_warning(excerpt, references, generator=DEFAULT_GENERATOR):
    """
    Plain-language warning for one excerpt, grounded in the retrieved reference
    clauses. references=[] gives the ungrounded baseline used by evaluate.py.
    The excerpt goes last so truncation trims its tail rather than the
    instruction or the grounding context.
    """
    # ponytail: picked from 6 prompt layouts tried by hand; flan-t5-base output is
    # still mostly extractive - evaluate.py generation measures how good it is.
    if generator in ("none", "templates"):
        return ""
    prompt = ""
    if references:
        prompt += "Similar known clauses: " + " ".join(clause for clause, _, _ in references) + "\n\n"
    prompt += f"Explain in simple words to a non-lawyer what the following terms mean for them: {excerpt}"

    generator_tokenizer, generator_model = get_generator(generator)
    if GENERATORS[generator].get("causal"):
        messages = [{"role": "system", "content": "You explain legal terms to ordinary people. "
                                                  "Reply with exactly one short sentence. No preamble, no lists."},
                    {"role": "user", "content": prompt}]
        inputs = generator_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            truncation=True, max_length=768)
        with torch.no_grad():
            outputs = generator_model.generate(**inputs, max_new_tokens=45, do_sample=False,
                                               pad_token_id=generator_tokenizer.eos_token_id)
        reply = generator_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        # keep the first sentence only; small chat models pad with preamble and lists
        reply = reply.strip().lstrip("-*0123456789. ").split("\n")[0]
        return (reply.split(". ")[0].strip() + ".") if ". " in reply else reply.strip()

    inputs = generator_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = generator_model.generate(**inputs, max_length=60, num_beams=2, early_stopping=True)
    return generator_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ============================================
# MAIN ENDPOINT
# ============================================
@app.route('/process', methods=['POST'])
def process_text():
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    incoming_links = data.get('links', [])  # Links from extension
    rag = data.get('rag') or DEFAULT_RAG
    generator = data.get('generator') or DEFAULT_GENERATOR
    classifier = data.get('classifier') or DEFAULT_CLASSIFIER
    if rag not in RAG_VARIANTS or generator not in GENERATORS or classifier not in CLASSIFIERS:
        return jsonify({"error": "Unknown rag, generator or classifier"}), 400

    if len(text) < 50:
        return jsonify({
            "ethical_points": [],
            "non_ethical_points": [],
            "risk_level": "Unknown",
            "summary": ["Please navigate to a page with more content."],
            "processing_time_seconds": 0,
            "highlighted_indices": [],
            "total_text_analyzed_chars": 0,
            "clauses_analyzed": 0,
            "chunks_analyzed": 0,
            "unclear_points": [],
            "links": [],
            "links_count": 0
        }), 200

    # 1. Strip navigation/headings, then apply the input limit to what is left
    text = clean_page_text(text)[:MAX_INPUT_CHARS]

    # 2. Chunk -> embed -> FAISS top-3 reference clauses per chunk
    chunks = chunk_text(text)  # reported as coverage; retrieval now runs per clause

    # 3. Classify the first MAX_SENTENCES sentences with Algorithm 1
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if len(s.strip()) > 10][:MAX_SENTENCES]

    points = {"ethical": [], "non_ethical": [], "unclear": []}
    highlighted_indices = []
    for sentence, (label, confidence, matched_clause, category, trace) in zip(
            sentences, classify_clauses(sentences, rag, classifier, with_trace=True)):
        points[label].append({"text": sentence, "confidence": confidence, "matched_clause": matched_clause,
                              "category": category, "trace": dict(trace, rule=rag, verdict_by=classifier)})
        if label != 'unclear':
            highlighted_indices.append({"text": sentence, "type": label})

    for group in points.values():
        group.sort(key=lambda x: x['confidence'], reverse=True)
    ethical_points, non_ethical_points = points["ethical"], points["non_ethical"]

    # 4. Summarise the clauses that were actually flagged, worst first. Short, clean
    # sentences give far better output than raw page chunks - and are much faster.
    summary = []
    # causal chat models take ~6s per summary, so they get the worst clause only;
    # the fast paths still explain the top three
    limit = 1 if GENERATORS[generator].get("causal") else TOP_K
    headline = (non_ethical_points + ethical_points)[:limit]
    if headline and generator == "templates":
        summary = [explain(point, 'non_ethical' if point in non_ethical_points else 'ethical')
                   for point in headline]
    elif headline and generator != "none":
        headline_references = retrieve_references(embed([p["text"] for p in headline]))
        summary = [generate_warning(point["text"], refs, generator)
                   for point, refs in zip(headline, headline_references)]

    # 5. Calculate Risk Level (count-based thresholds)
    # Only consider points with >= 50% confidence for risk classification.
    risk_confidence_threshold = 0.50
    ethical_count = sum(1 for p in ethical_points if p['confidence'] >= risk_confidence_threshold)
    non_ethical_count = sum(1 for p in non_ethical_points if p['confidence'] >= risk_confidence_threshold)

    if ethical_count == 0 and non_ethical_count == 0:
        risk_level = "Unknown"
    elif ethical_count == 0:
        # Non-ethical points exist with no ethical counterpoints.
        risk_level = "High"
    elif non_ethical_count >= ethical_count * 1.75:
        risk_level = "Medium-High"
    elif non_ethical_count >= ethical_count * 1.5:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    processing_time = round(time.time() - start_time, 2)

    # Extract URLs from the text and combine with extension-provided links
    all_links = list(set(re.findall(r'https?://[^\s\)]+', text)))
    for link in incoming_links:
        if link.get('url') and link['url'] not in all_links:
            all_links.append(link['url'])

    return jsonify({
        "ethical_points": ethical_points,
        "non_ethical_points": non_ethical_points,
        "risk_level": risk_level,
        "summary": summary,
        "processing_time_seconds": processing_time,
        "clauses_analyzed": len(sentences),
        "chunks_analyzed": len(chunks),
        "highlighted_indices": highlighted_indices,
        "total_text_analyzed_chars": len(text),
        "unclear_points": points["unclear"],
        "links": all_links,
        "links_count": len(all_links),
        "used": {"rag": rag, "generator": generator, "classifier": classifier}
    })

@app.route('/options', methods=['GET'])
def options():
    """Everything the popup needs to build its dropdowns, including the one-line reasons."""
    def entries(registry, default, available=lambda key: True):
        return [{"id": key, "name": spec["name"], "why": spec["why"],
                 "default": key == default, "available": available(key)}
                for key, spec in registry.items()]

    return jsonify({
        "rag": entries(RAG_VARIANTS, DEFAULT_RAG),
        "generator": entries(GENERATORS, DEFAULT_GENERATOR),
        "classifier": entries(CLASSIFIERS, DEFAULT_CLASSIFIER,
                              lambda key: not key.startswith("legalbert") or os.path.isdir(LEGALBERT_DIR)),
    })


PORT = int(os.environ.get("TERMSGUARD_PORT", 5001))


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "port": PORT, "models_loaded": True})

if __name__ == '__main__':
    app.run(port=PORT, debug=False, threaded=True)
