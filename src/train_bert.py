"""Fine-tune DistilBERT on the (deduplicated) game-chat splits, over several seeds.

Run (needs torch + transformers, see requirements-bert.txt; they are NOT in requirements.txt):
    python -m src.train_bert                    # seeds 42,43,44
    python -m src.train_bert --seeds 42 --limit 500 --epochs 1    # quick smoke test

Same data as every other model: train on train.csv, pick the best epoch by
validation loss, report on test.csv. Output goes to results/bert/:
    seed<N>.json          val/test accuracy, macro F1, per-class F1, confusion matrix
    predictions_seed<N>.csv   test predictions (for error analysis)
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import RESULTS_DIR, SPLITS_DIR

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = RESULTS_DIR / "bert"
MAX_LEN = 64
BATCH_SIZE = 32
LR = 5e-5
PATIENCE = 1  # stop after this many epochs without a better validation loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")


@torch.no_grad()
def predict(model, tok, texts, device):
    model.eval()
    logits = []
    texts = list(texts)
    for i in range(0, len(texts), 128):
        b = {k: v.to(device) for k, v in encode(tok, texts[i:i + 128]).items()}
        logits.append(model(**b).logits.float().cpu())
    return torch.cat(logits)


def score(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2]),
            "f1_per_class": f1_score(y, pred, average=None, labels=[0, 1, 2]).tolist(),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def run(seed: int, train, val, test, epochs: int, device) -> dict:
    seed_everything(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(train) / BATCH_SIZE))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    y_tr = torch.tensor(train["label"].to_numpy())
    y_va = torch.tensor(val["label"].to_numpy())
    texts = train["text_norm"].tolist()
    gen = torch.Generator().manual_seed(seed)

    best_loss, best_state, bad, best_epoch = float("inf"), None, 0, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(texts), generator=gen).tolist()
        for i in range(0, len(order), BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            b = {k: v.to(device) for k, v in encode(tok, [texts[j] for j in idx]).items()}
            loss = model(**b, labels=y_tr[idx].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        v_logits = predict(model, tok, val["text_norm"], device)
        v_loss = torch.nn.functional.cross_entropy(v_logits, y_va).item()
        v_acc = (v_logits.argmax(1) == y_va).float().mean().item()
        print(f"  seed {seed} epoch {epoch}: val loss {v_loss:.4f}  val acc {v_acc:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if v_loss < best_loss:
            best_loss, best_epoch, bad = v_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad > PATIENCE:
                break
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)

    v_pred = predict(model, tok, val["text_norm"], device).argmax(1).numpy()
    t_logits = predict(model, tok, test["text_norm"], device)
    t_pred = t_logits.argmax(1).numpy()
    proba = torch.softmax(t_logits, 1).numpy()
    seen = test["text_norm"].isin(set(train["text_norm"])).to_numpy()
    res = {"seed": seed, "model": MODEL_NAME, "best_epoch": best_epoch, "train_seconds": round(secs),
           "val": score(val["label"].to_numpy(), v_pred),
           "test": score(test["label"].to_numpy(), t_pred),
           "test_share_seen_in_train": float(seen.mean())}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"seed{seed}.json").write_text(json.dumps(res, indent=2))
    preds = test[["text", "text_norm", "source", "label"]].copy()
    preds["true"], preds["pred"] = preds["label"], t_pred
    for c in range(3):
        preds[f"p{c}"] = proba[:, c].round(4)
    preds.to_csv(OUT_DIR / f"predictions_seed{seed}.csv", index=False)
    print(f"seed {seed}: test acc {res['test']['accuracy']:.4f}  macro F1 {res['test']['macro_f1']:.4f}",
          flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--limit", type=int, help="use only the first N rows of each split (smoke test)")
    args = ap.parse_args()

    train, val, test = (pd.read_csv(SPLITS_DIR / f"{s}.csv", keep_default_na=False)
                        for s in ("train", "val", "test"))
    if args.limit:
        train, val, test = train[:args.limit], val[:args.limit], test[:args.limit]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  train/val/test={len(train)}/{len(val)}/{len(test)}")
    for seed in map(int, args.seeds.split(",")):
        run(seed, train, val, test, args.epochs, device)


if __name__ == "__main__":
    main()
