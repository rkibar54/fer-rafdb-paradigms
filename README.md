# Zero-Shot Multimodal Models, Linear Probes, or Fine-Tuning? — FER on RAF-DB

Code accompanying the paper:

> R. Kibar and M. Kotan, "Zero-Shot Multimodal Models, Linear Probes, or Fine-Tuning? A Controlled Comparison of Learning Paradigms for In-the-Wild Facial Expression Recognition on RAF-DB," *Sakarya University Journal of Computer and Information Sciences*, 2026 (under review).

The repository contains all notebooks used to train and evaluate the models, the shared evaluation module, the aggregated result tables, and a script that reproduces the figures of the paper.

## Overview

Five learning paradigms are compared on the RAF-DB basic-emotion subset (7 classes) under a single protocol (fixed train/validation split, same test images, three seeds for trained models, paired McNemar tests):

| Paradigm | Models |
|---|---|
| Training from scratch | Custom CNN |
| Full fine-tuning | ResNet-18 (ImageNet-1K), ViT-B/16 (ImageNet-21K) |
| Linear probe | CLIP ViT-B/32, CLIP ViT-B/16 (frozen encoders) |
| Zero-shot contrastive | CLIP ViT-B/32, CLIP ViT-B/16 |
| Zero-shot generative | BLIP-VQA, InstructBLIP (Vicuna-7B), LLaVA-1.5-7B, Qwen2.5-VL-7B-Instruct |

For generative models, three output-to-label protocols are evaluated on the same outputs: strict matching, lexicon-based matching, and closed-set likelihood scoring. A label-efficiency study trains linear probes and ViT-B/16 on 1–100% of the training data.

### Main results (test accuracy, %)

| Model | Accuracy | Mean acc. | Macro-F1 |
|---|---|---|---|
| CLIP ViT-B/16 linear probe | 86.31 | 77.68 | 78.67 |
| ViT-B/16 fine-tuned (3 seeds) | 85.46 ± 0.34 | 76.54 | 77.02 |
| ResNet-18 fine-tuned (3 seeds) | 83.77 ± 0.20 | 73.83 | 75.20 |
| Custom CNN (3 seeds) | 80.36 ± 0.56 | 68.70 | 70.77 |
| Qwen2.5-VL-7B, zero-shot | 64.57 | 53.16 | 53.41 |
| LLaVA-1.5-7B, zero-shot | 64.05 | 57.27 | 48.30 |

Full tables are in [`results/`](results/).

## Repository structure

```
fer_common.py                shared module: data split, label mapping, metrics, training loop
notebooks/
  01_cnn_baseline.ipynb      custom CNN trained from scratch
  02_resnet18.ipynb          ResNet-18 fine-tuning
  03_vit_b16.ipynb           ViT-B/16 fine-tuning (main run + label-efficiency run)
  04_clip.ipynb              CLIP zero-shot and linear probe (incl. label efficiency)
  05_blip_vqa.ipynb          BLIP-VQA, open and closed question
  06_instructblip.ipynb      InstructBLIP (Vicuna-7B)
  07_llava15.ipynb           LLaVA-1.5-7B
  08_qwen25vl.ipynb          Qwen2.5-VL-7B-Instruct
  99_evaluation.ipynb        aggregates all predictions into tables and figures
scripts/make_paper_figures.py  reproduces the figures of the paper
results/                     aggregated result tables (CSV)
```

Each notebook writes per-image predictions (`preds_<name>.csv`) and run metadata (`meta_<name>.json`, including hyper-parameters, timing and GPU memory). The first cell of every notebook writes `fer_common.py`, so each notebook is self-contained.

## Data

RAF-DB is **not** included in this repository. It must be requested from its owners for non-commercial research use: http://www.whdeng.cn/raf/model1.html

The code expects the aligned images in the folder structure `DATASET/train/<1..7>/` and `DATASET/test/<1..7>/`, where folders 1–7 correspond to surprise, fear, disgust, happiness, sadness, anger, and neutral. The train/validation split (10% validation, stratified, seed 42) is created deterministically by `fer_common.load_raf_db()`.

## Running the experiments

The experiments were run on Kaggle with two NVIDIA T4 GPUs (Python 3.13, PyTorch 2.11, Transformers 4.53.3).

1. Import a notebook into Kaggle (*File → Import Notebook*).
2. Add RAF-DB as an input, enable the internet, and select **GPU T4 x2**.
3. Run with *Save Version → Save & Run All*.
4. For the label-efficiency run of ViT-B/16, run a copy of `03_vit_b16.ipynb` with `LABEL_EFF_MODE = True` and `FRACTIONS = [0.01, 0.05, 0.10, 0.25, 1.0]`.
5. Add the outputs of all notebooks as inputs to `99_evaluation.ipynb` and run it to obtain all tables.

Approximate run times on 2×T4: CNN < 1 h, ResNet-18 ≈ 1 h, ViT-B/16 ≈ 1 h (main) and ≈ 2–3 h (label efficiency), CLIP < 1 h, BLIP < 1 h, InstructBLIP ≈ 1.5 h, Qwen2.5-VL ≈ 5 h, LLaVA-1.5 ≈ 6.5 h. Generative notebooks save partial results every 250 images and can be split into shards (`NUM_SHARDS`) if needed.

To reproduce the figures from the prediction files:

```bash
pip install -r requirements.txt
python scripts/make_paper_figures.py --preds <folder with preds_*.csv> --out figures/
```

## Notes

- Generative models are run in FP16 with greedy decoding (max. 10 new tokens); results are deterministic.
- The lexicon used for label matching is defined in `fer_common.LEXICON` and was fixed before the final runs.
- Per-image prediction files are not included because they contain RAF-DB labels; they can be regenerated with the notebooks.

## Citation

```bibtex
@article{kibar2026fer,
  author  = {Kibar, Refik and Kotan, Muhammed},
  title   = {Zero-Shot Multimodal Models, Linear Probes, or Fine-Tuning? A Controlled Comparison of Learning Paradigms for In-the-Wild Facial Expression Recognition on RAF-DB},
  journal = {Sakarya University Journal of Computer and Information Sciences},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT License (see [`LICENSE`](LICENSE)). RAF-DB is subject to its own license terms.
