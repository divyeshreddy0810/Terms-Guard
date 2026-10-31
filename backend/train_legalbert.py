"""
Fine-tunes Legal-BERT on the eight UNFAIR-ToS categories, so the extension can
compare it against clause matching and name the kind of unfairness it found.
Run with: python train_legalbert.py

A sentence counts as unfair when any category fires, which is what the extension
needs, but training on the eight categories gives the model a finer signal than a
single unfair/fair target. Rare categories get a heavier loss so they are not
drowned out by the ~90% of sentences that are fair.

Trains on the train split plus most of the validation split, early-stops on the
rest by unfair-clause F1, and saves the best epoch to models/legalbert-unfair-tos/.
The test split is never touched here - evaluate.py scores every method on it.

Uses the Mac GPU (MPS) when available, otherwise the CPU.
"""
import os
import random
import sys

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 6-layer legal model: about 4x faster and much lighter than legal-bert-base, which
# matters on an 8GB machine. Override to compare: LEGALBERT_MODEL=nlpaueb/legal-bert-base-uncased
MODEL = os.environ.get("LEGALBERT_MODEL", "nlpaueb/legal-bert-small-uncased")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "legalbert-unfair-tos")
LABELS = ["Limitation of liability", "Unilateral termination", "Unilateral change", "Content removal",
          "Contract by using", "Choice of law", "Jurisdiction", "Arbitration"]
EPOCHS = 4  # the smaller model needs more passes than the base one
BATCH_SIZE = 16
MAX_LENGTH = 256      # 128 truncated 4% of unfair sentences; padding is per batch, so this is nearly free
LEARNING_RATE = 2e-5
HOLDOUT = 0.2         # share of the validation split kept back for early stopping


def collate(tokeniser):
    def collate_batch(rows):
        batch = tokeniser([r["text"] for r in rows], padding=True, truncation=True,
                          max_length=MAX_LENGTH, return_tensors="pt")
        batch["labels"] = torch.tensor([[1.0 if name in r["labels"] else 0.0 for name in LABELS] for r in rows])
        return batch
    return collate_batch


def unfair_f1(model, loader, device):
    """F1 for "any category fires" - the decision the extension actually shows."""
    model.eval()
    true_positive = predicted = actual = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            predictions = (logits.sigmoid().max(-1).values >= 0.5).cpu()
            gold = labels.max(-1).values >= 0.5
            true_positive += int((predictions & gold).sum())
            predicted += int(predictions.sum())
            actual += int(gold.sum())
    model.train()
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / actual if actual else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def main():
    import app  # the dataset loader lives there, so both use exactly the same splits

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    validation = app.load_unfair_tos("val")
    random.Random(0).shuffle(validation)
    split = int(len(validation) * HOLDOUT)
    holdout, extra = validation[:split], validation[split:]
    training = app.load_unfair_tos("train") + extra
    print(f"training on {len(training)} sentences ({len(extra)} from the validation split), "
          f"early stopping on {len(holdout)}")

    tokeniser = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), problem_type="multi_label_classification",
        id2label=dict(enumerate(LABELS)), label2id={name: i for i, name in enumerate(LABELS)}).to(device)

    train_loader = DataLoader(training, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate(tokeniser))
    holdout_loader = DataLoader(holdout, batch_size=64, collate_fn=collate(tokeniser))

    # Rare categories would otherwise be drowned out; sqrt keeps the correction from
    # over-flagging (raw ratios reach 1:125 for arbitration).
    positives = [sum(name in r["labels"] for r in training) for name in LABELS]
    pos_weight = torch.tensor([((len(training) - n) / max(n, 1)) ** 0.5 for n in positives], device=device)
    print("positives per category:", dict(zip(LABELS, positives)))
    loss_function = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    schedule = torch.optim.lr_scheduler.LinearLR(optimiser, 1.0, 0.0, EPOCHS * len(train_loader))

    best = 0.0
    for epoch in range(1, EPOCHS + 1):
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            loss = loss_function(logits, labels)
            loss.backward()
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f}", flush=True)

        score = unfair_f1(model, holdout_loader, device)
        print(f"epoch {epoch}: held-out unfair F1 {score:.4f}", flush=True)
        if score > best:
            best = score
            os.makedirs(OUT_DIR, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokeniser.save_pretrained(OUT_DIR)
            print(f"saved to {OUT_DIR}", flush=True)

    if best == 0.0:
        sys.exit("training produced no usable model")
    print(f"best held-out unfair F1: {best:.4f}")


if __name__ == "__main__":
    main()
