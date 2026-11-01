"""Reproduce the three figures of the paper from the prediction files.

Usage:
    python scripts/make_paper_figures.py --preds <folder with preds_*.csv> --out figures/

The prediction files are produced by the notebooks in notebooks/ (01-08).
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fer_common as fc  # noqa: E402

NAMES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
ABBR = ["Sur", "Fea", "Dis", "Hap", "Sad", "Ang", "Neu"]
N_TRAIN_FULL = 11043


def load_predictions(folder):
    groups = {}
    for f in glob.glob(os.path.join(folder, "**", "preds_*.csv"), recursive=True):
        name = re.sub(r"_shard\d+of\d+", "", os.path.basename(f)[6:-4])
        groups.setdefault(name, []).append(f)
    preds = {}
    for name, files in groups.items():
        df = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
        preds[name] = df.drop_duplicates("image_id").sort_values("image_id").reset_index(drop=True)
    return preds


def parse(name):
    m = re.match(r"^(?P<base>.+?)(?:_le_f(?P<pct>\d{3}))?(?:_s(?P<seed>\d+))?$", name)
    return m["base"], (int(m["pct"]) if m["pct"] else None), (int(m["seed"]) if m["seed"] else None)


def acc(df):
    return float((df.y_true.values == df.y_pred.values.astype(int)).mean() * 100)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "legend.fontsize": 8})
    P = load_predictions(args.preds)

    # ---- Figure: label efficiency
    rows = []
    for name, df in P.items():
        base, pct, seed = parse(name)
        if pct is not None:
            rows.append({"model": base, "pct": pct, "seed": seed, "acc": acc(df)})
    le = pd.DataFrame(rows).groupby(["model", "pct"]).acc.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    for m, lab, mk in [("clip_b16_lp", "CLIP ViT-B/16 linear probe", "o"),
                       ("clip_b32_lp", "CLIP ViT-B/32 linear probe", "s"),
                       ("vit_b16", "ViT-B/16 full fine-tuning", "^")]:
        g = le[le.model == m].sort_values("pct")
        ax.errorbar(g.pct / 100 * N_TRAIN_FULL, g["mean"], yerr=g["std"].fillna(0), marker=mk,
                    capsize=2.5, lw=1.3, ms=4.5, label=lab)
    for k, lab, ls, col in [("qwen25vl_7b_score", "Qwen2.5-VL-7B (zero-shot)", (0, (1, 1.5)), "0.1"),
                            ("llava15_7b_score", "LLaVA-1.5-7B (zero-shot)", "--", "0.35"),
                            ("instructblip_v7b_score", "InstructBLIP-7B (zero-shot)", "-.", "0.35"),
                            ("clip_b16_zs_ens", "CLIP ViT-B/16 (zero-shot)", (0, (6, 2, 1, 2, 1, 2)), "0.5")]:
        if k in P:
            ax.axhline(acc(P[k]), color=col, ls=ls, lw=1.1, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("Number of labeled training images (log scale)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(44, 90)
    ax.grid(alpha=0.3, lw=0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_label_efficiency.pdf"))
    plt.close(fig)

    # ---- Figure: predicted class distribution of generative models
    models = [("True distribution", None), ("BLIP (closed question)", "blip_vqa_closed_gen"),
              ("InstructBLIP-7B", "instructblip_v7b_gen"), ("LLaVA-1.5-7B", "llava15_7b_gen"),
              ("Qwen2.5-VL-7B", "qwen25vl_7b_gen")]
    y = P["vit_b16_s0"].y_true.values
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    w, x = 0.16, np.arange(7)
    hatches = ["", "//", "..", "xx", "\\\\"]
    for i, (lab, k) in enumerate(models):
        if k is None:
            v = np.bincount(y, minlength=7) / len(y) * 100
        else:
            p = P[k].y_pred.values.astype(int)
            v = np.array([(p == c).mean() * 100 for c in range(7)])
        ax.bar(x + (i - 2) * w, v, w, label=lab, color=("0.15" if i == 0 else None), hatch=hatches[i],
               edgecolor="white" if i == 0 else "black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("Share of test predictions (%)")
    ax.set_ylim(0, 52)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_pred_distribution.pdf"))
    plt.close(fig)

    # ---- Figure: confusion matrices
    sel = [("ViT-B/16 (fine-tuned)", "vit_b16_s0"), ("CLIP ViT-B/16 (linear probe)", "clip_b16_lp"),
           ("LLaVA-1.5-7B (zero-shot)", "llava15_7b_score"), ("Qwen2.5-VL-7B (zero-shot)", "qwen25vl_7b_score")]
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 6.0))
    for ax, (lab, k) in zip(axs.ravel(), sel):
        d = P[k]
        cm = fc.confusion(d.y_true.values, d.y_pred.values.astype(int)).values[:, :7].astype(float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Greys")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f"{cmn[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cmn[i, j] > 0.5 else "black")
        ax.set_xticks(range(7)); ax.set_xticklabels(ABBR)
        ax.set_yticks(range(7)); ax.set_yticklabels(ABBR)
        ax.set_title(lab); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_confusions.pdf"))
    plt.close(fig)
    print("Figures written to", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="folder containing preds_*.csv files")
    ap.add_argument("--out", default="figures")
    main(ap.parse_args())
