"""
fer_common.py — shared module for the RAF-DB FER benchmark.

All notebooks use this file, so that data splitting, label mapping,
prediction saving and metrics are identical for every model.

Usage on Kaggle: run this cell with %%writefile fer_common.py,
then:  import fer_common as fc
"""
import os
import re
import json
import time
import platform

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
# RAF-DB folder numbers 1..7 -> indices 0..6 (original RAF-DB order)
CLASS_NAMES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
NUM_CLASSES = len(CLASS_NAMES)
SPLIT_SEED = 42      # FIXED seed for the train/val split — do not change
VAL_FRACTION = 0.10
UNMATCHED = -1       # generative output that cannot be mapped to a class

OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "./outputs"


# ---------------------------------------------------------------------------
# 2. DATASET
# ---------------------------------------------------------------------------
def find_dataset_root(base="/kaggle/input"):
    """Finds the DATASET directory that contains the train/ and test/ folders."""
    for dirpath, dirnames, _ in os.walk(base, followlinks=True):
        if "train" in dirnames and "test" in dirnames:
            return dirpath
    raise FileNotFoundError("RAF-DB folder (containing train/ and test/) not found.")


def list_split(root, split):
    """split = 'train' or 'test'. Returns a sorted, deterministic DataFrame.

    Columns: image_id (e.g. 'test/4/test_0001_aligned.jpg'), path, label (0..6)
    """
    rows = []
    split_dir = os.path.join(root, split)
    for folder in sorted(os.listdir(split_dir), key=lambda s: (len(s), s)):
        cdir = os.path.join(split_dir, folder)
        if not os.path.isdir(cdir) or not folder.isdigit():
            continue
        label = int(folder) - 1
        for fn in sorted(os.listdir(cdir)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append({"image_id": f"{split}/{folder}/{fn}",
                             "path": os.path.join(cdir, fn),
                             "label": label})
    df = pd.DataFrame(rows)
    assert df["label"].between(0, NUM_CLASSES - 1).all()
    return df


def load_raf_db(root=None, verbose=True):
    """Returns train / val / test DataFrames. Val is a stratified split of train."""
    from sklearn.model_selection import train_test_split
    root = root or find_dataset_root()
    train_full = list_split(root, "train")
    test = list_split(root, "test")
    train, val = train_test_split(train_full, test_size=VAL_FRACTION,
                                  stratify=train_full["label"],
                                  random_state=SPLIT_SEED)
    train = train.sort_values("image_id").reset_index(drop=True)
    val = val.sort_values("image_id").reset_index(drop=True)
    if verbose:
        print(f"RAF-DB root: {root}")
        print(f"train={len(train)}  val={len(val)}  test={len(test)}")
        print(class_distribution({"train": train, "val": val, "test": test}))
    return train, val, test


def subsample_train(train, fraction, seed):
    """Stratified training subset for label-efficiency experiments (>=1 image per class)."""
    if fraction >= 1.0:
        return train.copy()
    parts = []
    for _, g in train.groupby("label"):
        n = max(1, int(round(len(g) * fraction)))
        parts.append(g.sample(n=n, random_state=seed))
    return pd.concat(parts).sort_values("image_id").reset_index(drop=True)


def class_distribution(splits):
    tab = pd.DataFrame({name: df["label"].value_counts().sort_index()
                        for name, df in splits.items()})
    tab.index = [CLASS_NAMES[i] for i in tab.index]
    tab.loc["TOTAL"] = tab.sum()
    return tab


# ---------------------------------------------------------------------------
# 3. LABEL MAPPING (for generative models) — the lexicon was fixed IN ADVANCE
# ---------------------------------------------------------------------------
# Word forms / synonyms per class. Terms are matched at word boundaries (\b),
# so that wrong matches such as 'unhappy' -> 'happy' are avoided.
LEXICON = {
    "surprise": ["surprise", "surprised", "surprising", "shock", "shocked",
                 "astonished", "astonishment", "amazed", "amazement", "startled"],
    "fear":     ["fear", "fearful", "afraid", "scared", "frightened", "terrified",
                 "terror", "panic", "panicked"],
    "disgust":  ["disgust", "disgusted", "disgusting", "revulsion", "repulsed",
                 "grossed out"],
    "happy":    ["happy", "happiness", "joy", "joyful", "smile", "smiling",
                 "laughing", "laughter", "cheerful", "glad", "delighted"],
    "sad":      ["sad", "sadness", "unhappy", "sorrow", "sorrowful", "crying",
                 "tearful", "upset", "depressed", "grief", "melancholy"],
    "angry":    ["angry", "anger", "mad", "furious", "rage", "annoyed",
                 "irritated", "enraged"],
    "neutral":  ["neutral", "calm", "expressionless", "blank", "no emotion",
                 "emotionless", "indifferent"],
}
_NEG = (r"\b(?:not|no|never|isn't|isnt|doesn't|doesnt|without)\s+"
        r"(?:(?:very|really|too|so|that|particularly|exactly|look|looking|seem|seems)\s+)?")

_PATTERNS = {c: [re.compile(r"\b" + re.escape(w) + r"\b") for w in words]
             for c, words in LEXICON.items()}
_NEG_PATTERNS = {c: [re.compile(_NEG + re.escape(w) + r"\b") for w in words]
                 for c, words in LEXICON.items()}


def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def map_strict(text):
    """Strict matching: accepted only if the output is exactly a class name, otherwise UNMATCHED."""
    t = normalize_text(text)
    return CLASS_NAMES.index(t) if t in CLASS_NAMES else UNMATCHED


def map_lexicon(text):
    """Lexicon-based matching. Returns: (label_idx, status)

    status: 'exact' | 'lexicon' | 'ambiguous' | 'unmatched'
    If several classes are mentioned, the FIRST one in the text is used ('ambiguous').
    Negated expressions ('not happy') are ignored.
    """
    t = normalize_text(text)
    if t in CLASS_NAMES:
        return CLASS_NAMES.index(t), "exact"
    hits = {}
    for c in CLASS_NAMES:
        for p, np_ in zip(_PATTERNS[c], _NEG_PATTERNS[c]):
            for m in p.finditer(t):
                negated = any(nm.end() == m.end() for nm in np_.finditer(t))
                if not negated:
                    hits[c] = min(hits.get(c, 10**9), m.start())
    if not hits:
        return UNMATCHED, "unmatched"
    first = min(hits, key=hits.get)
    return CLASS_NAMES.index(first), ("lexicon" if len(hits) == 1 else "ambiguous")


# ---------------------------------------------------------------------------
# 4. SAVING PREDICTIONS
# ---------------------------------------------------------------------------
def save_predictions(df, model_name, extra_meta=None, out_dir=None):
    """df must contain at least the columns: image_id, y_true, y_pred
    (for generative models additionally raw_output; probability scores are optional).
    Files: preds_<model_name>.csv  +  meta_<model_name>.json
    """
    need = {"image_id", "y_true", "y_pred"}
    assert need.issubset(df.columns), f"Missing columns: {need - set(df.columns)}"
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"preds_{model_name}.csv")
    df.to_csv(path, index=False)
    meta = {"model_name": model_name, "n": int(len(df)),
            "class_names": CLASS_NAMES, "env": env_info()}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(out_dir, f"meta_{model_name}.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved: {path}")
    return path


def env_info():
    info = {"python": platform.python_version(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


class Timer:
    """Measures inference time and peak GPU memory (for the cost table)."""
    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
        except ImportError:
            pass
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t0
        self.peak_gpu_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_gb = sum(torch.cuda.max_memory_allocated(i)
                                       for i in range(torch.cuda.device_count())) / 1e9
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# 5. METRICS
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """UNMATCHED (-1) predictions are counted as errors."""
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))
    per_class_recall = np.array([(y_pred[y_true == c] == c).mean() for c in labels])
    return {
        "overall_acc": float((y_true == y_pred).mean()),
        "mean_acc": float(per_class_recall.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0)),
        "unmatched_rate": float((y_pred == UNMATCHED).mean()),
        **{f"recall_{CLASS_NAMES[c]}": float(per_class_recall[c]) for c in labels},
    }


def bootstrap_ci(y_true, y_pred, n_boot=2000, seed=0, alpha=0.05):
    """95% bootstrap confidence interval for overall accuracy."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    n = len(correct)
    accs = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.quantile(accs, alpha / 2)), float(np.quantile(accs, 1 - alpha / 2))


def confusion(y_true, y_pred):
    """Rows: true class, columns: predicted class (+ 'unmatched' column)."""
    cols = CLASS_NAMES + ["unmatched"]
    cm = pd.DataFrame(0, index=CLASS_NAMES, columns=cols)
    for t, p in zip(y_true, y_pred):
        cm.iloc[t, p if p != UNMATCHED else NUM_CLASSES] += 1
    if cm["unmatched"].sum() == 0:
        cm = cm.drop(columns="unmatched")
    return cm


def mcnemar_exact(y_true, pred_a, pred_b):
    """Exact McNemar test for the predictions of two models on the same test images."""
    from scipy.stats import binomtest
    y_true = np.asarray(y_true)
    ca = np.asarray(pred_a) == y_true
    cb = np.asarray(pred_b) == y_true
    b = int((ca & ~cb).sum())   # A correct, B wrong
    c = int((~ca & cb).sum())   # A wrong, B correct
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def report(y_true, y_pred, title=""):
    m = compute_metrics(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    print(f"\n===== {title} =====")
    print(f"Overall acc : {m['overall_acc']*100:.2f}%  (95% CI {lo*100:.2f}–{hi*100:.2f})")
    print(f"Mean acc    : {m['mean_acc']*100:.2f}%")
    print(f"Macro-F1    : {m['macro_f1']*100:.2f}%")
    if m["unmatched_rate"] > 0:
        print(f"Unmatched   : {m['unmatched_rate']*100:.2f}%")
    print(confusion(y_true, y_pred))
    return m


# ---------------------------------------------------------------------------
# 6. SHARED PROMPTS (all generative models use the same instruction)
# ---------------------------------------------------------------------------
MLLM_INSTRUCTION = ("What is the facial expression of the person in this image? "
                    "Choose one of: happy, sad, angry, surprise, fear, disgust, neutral. "
                    "Answer with one word.")
CLIP_ADJ = ["surprised", "fearful", "disgusted", "happy", "sad", "angry", "neutral"]


def strip_prompt(out_ids, in_ids):
    """Removes the prompt from generate() output if it is included (version-independent)."""
    L = in_ids.shape[-1]
    if out_ids.shape[-1] > L and bool((out_ids[0, :L] == in_ids[0]).all()):
        return out_ids[0, L:]
    return out_ids[0]


# ---------------------------------------------------------------------------
# 7. IMAGE LOADING AND SUPERVISED TRAINING
# ---------------------------------------------------------------------------
def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_images(df, size=100):
    """Loads all images into memory as a uint8 array: [N, size, size, 3]."""
    from PIL import Image
    out = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        im = Image.open(p).convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BILINEAR)
        out[i] = np.asarray(im)
    return out


IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
HALF_MEAN, HALF_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def gpu_preprocess(xb, size, mean, std, train, pad=8):
    """xb: uint8 [B,H,W,3] on the GPU. Training: reflect padding + random crop + horizontal flip."""
    import torch
    import torch.nn.functional as F
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    if train:
        B, _, H, W = x.shape
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        ii = torch.randint(0, 2 * pad + 1, (B,)).tolist()
        jj = torch.randint(0, 2 * pad + 1, (B,)).tolist()
        x = torch.stack([xp[b, :, ii[b]:ii[b] + H, jj[b]:jj[b] + W] for b in range(B)])
        flip = torch.rand(B, device=x.device) < 0.5
        x[flip] = x[flip].flip(-1)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _logits(out):
    return out.logits if hasattr(out, "logits") else out


def train_supervised(name, build_model, X_tr, y_tr, X_val, y_val, X_te, y_te, test_ids,
                     seed, input_size, mean, std, epochs, batch_size, lr, weight_decay,
                     warmup_epochs=1, label_smoothing=0.0, select_on_val=True,
                     min_steps=0, max_epochs=200, extra_meta=None):
    """Generic supervised training loop (AdamW + cosine + warm-up + AMP).

    select_on_val=True  : the epoch with the best validation accuracy is selected.
    select_on_val=False : the final epoch is used (label-efficiency experiments).
    Test predictions and probabilities are saved as preds_<name>.csv.
    """
    import math, copy
    import torch
    import torch.nn as nn
    set_seed(seed)
    dev = "cuda"
    model = build_model().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = len(y_tr)
    steps_per_epoch = max(1, math.ceil(n_train / batch_size))
    epochs = min(max_epochs, max(epochs, math.ceil(min_steps / steps_per_epoch)))
    total = epochs * steps_per_epoch
    warm = min(warmup_epochs * steps_per_epoch, total // 5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    Xtr_g = torch.from_numpy(X_tr).to(dev)
    ytr_g = torch.from_numpy(np.asarray(y_tr)).long().to(dev)

    @torch.no_grad()
    def predict(X):
        model.eval()
        probs = []
        Xg = torch.from_numpy(X)
        for i in range(0, len(X), 256):
            xb = gpu_preprocess(Xg[i:i + 256].to(dev), input_size, mean, std, False)
            with torch.autocast("cuda", dtype=torch.float16):
                probs.append(torch.softmax(_logits(model(xb)).float(), -1).cpu())
        return torch.cat(probs).numpy()

    best = (-1.0, None, -1)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=dev)
        tot_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = gpu_preprocess(Xtr_g[idx], input_size, mean, std, True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = crit(_logits(model(xb)), ytr_g[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * len(idx)
        rec = {"epoch": ep + 1, "train_loss": tot_loss / n_train}
        if select_on_val:
            rec["val_acc"] = float((predict(X_val).argmax(1) == np.asarray(y_val)).mean())
            if rec["val_acc"] > best[0]:
                best = (rec["val_acc"], copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()}), ep + 1)
        history.append(rec)
        print(rec)
    train_sec = time.time() - t0
    if select_on_val:
        model.load_state_dict(best[1])
    torch.cuda.synchronize()
    t1 = time.time()
    P = predict(X_te)
    torch.cuda.synchronize()
    infer_sec_per_img = (time.time() - t1) / len(X_te)
    y_pred = P.argmax(1)
    df = pd.DataFrame({"image_id": test_ids, "y_true": y_te, "y_pred": y_pred})
    for c, cname in enumerate(CLASS_NAMES):
        df[f"p_{cname}"] = P[:, c]
    m = compute_metrics(y_te, y_pred)
    meta = {"seed": seed, "n_train": n_train, "epochs": epochs, "batch_size": batch_size,
            "lr": lr, "weight_decay": weight_decay, "input_size": input_size,
            "label_smoothing": label_smoothing, "select_on_val": select_on_val,
            "best_epoch": best[2], "best_val_acc": best[0], "n_params": n_params,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "train_seconds": train_sec, "infer_sec_per_image": infer_sec_per_img,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "history": history, "metrics": m}
    if extra_meta:
        meta.update(extra_meta)
    save_predictions(df, name, meta)
    print(f"[{name}] test acc={m['overall_acc']*100:.2f}  mean acc={m['mean_acc']*100:.2f}  "
          f"macro-F1={m['macro_f1']*100:.2f}")
    del model, Xtr_g, ytr_g
    torch.cuda.empty_cache()
    return m
