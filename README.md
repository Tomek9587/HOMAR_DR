# HOMAR: Hierarchical Ordinal Mamba with Anatomical Routing for Multi-Source Diabetic Retinopathy Grading

Official code for the JBHI submission *"HOMAR: Hierarchical Ordinal Mamba
with Anatomical Routing for Multi-Source Diabetic Retinopathy Grading"*.

HOMAR builds on the RETFound foundation model and adds three modules:

1. **Anatomical routing** — three fixed spatial priors (macular, optic-disc,
   peripheral) over the patch grid plus a learned instance-adaptive gate;
2. **Mamba capsule experts + hierarchical Mamba aggregation** — one expert
   per anatomical stream, fused across three receptive fields with a
   cross-level consensus;
3. **CORAL ordinal head** — trained with a differentiable Soft-QWK surrogate
   and a label-smoothed cross-entropy.

Trained on a four-source joint dataset (APTOS-2019 + Messidor-2 + IDRiD +
DDR, JOINT4, n = 10,456), a single deployed checkpoint reaches
QWK = 0.9001 / 0.7038 / 0.7329 / 0.7130 on the four held-out test splits
and 0.7555 on the combined pool.

## Repository layout

```
scripts/
  HOMAR-retfound-v2.py        Core model (HOMAR), trainer, EMA, TTA, metrics
  HOMAR-retfound-v3.py        Multi-source training entry (JOINT4 and ablations)
  retfound_baseline.py        P1 baseline (RETFound + CORAL, no Mamba head)
  evaluate_external.py        Cross-domain evaluation of a checkpoint
  profile_efficiency.py       Params / FLOPs / FPS profiling
  prepare_combined_dataset.py Build merged ImageFolder for APTOS + Messidor-2
  prepare_ddr.py              DDR split preparation
  prepare_m2_only.py          Messidor-2 split preparation
results/
  single_source/              Final metrics: JOINT4 and each single-source run
  ablation/                   Final metrics: P1, no-hierarchy, no-consensus, no-routing
  cross_domain/               Full cross-domain transfer matrix (JSON)
  cross_domain_ablation/      Cross-domain evaluation of the ablation variants
```

## Environment

- Python 3.10+, PyTorch 2.x, torchvision, scikit-learn, numpy, pandas, tqdm
- One 24 GB GPU (all experiments used a single NVIDIA RTX 4090)

```bash
pip install torch torchvision scikit-learn pandas tqdm
```

You also need the RETFound ViT-L/16 weights (`RETFound_cfp_weights.pth`)
from the official RETFound release: https://github.com/rmaphoh/RETFound_MAE

## Data preparation

Four public datasets are used: APTOS-2019 (Kaggle), Messidor-2, IDRiD
(B-set, disease grading) and DDR. Download them from their official
sources, then:

```bash
python scripts/prepare_ddr.py                 # merge DDR train/valid/test, drop ungradable
python scripts/prepare_m2_only.py             # Messidor-2 ImageFolder layout
python scripts/prepare_combined_dataset.py    # optional merged APTOS+M2 layout
```

APTOS and Messidor-2 use a stratified 70/30 image-level split (seed 42);
IDRiD uses its official 413/103 split; DDR uses the stratified 70/30 split
(6,260/3,759) described in the paper.

## Training

JOINT4 (main model, 100 epochs, two-stage protocol):

```bash
python scripts/HOMAR-retfound-v3.py \
  --aptos_root  /path/to/aptos \
  --m2_root     /path/to/messidor2 \
  --idrid_train /path/to/idrid/train \
  --idrid_test  /path/to/idrid/test \
  --ddr_train   /path/to/ddr/train \
  --ddr_test    /path/to/ddr/test \
  --save_dir runs/homar_v3_joint4 --epochs 100 --patience 999
```

Single-source runs: pass only the corresponding root(s). Ablations add
`--no_routing`, `--no_hierarchy`, `--no_consensus`; the P1 baseline is
trained with `scripts/retfound_baseline.py` using the same arguments.

## Cross-domain evaluation

```bash
python scripts/evaluate_external.py \
  --ckpt runs/homar_v3_joint4/best_model.pth \
  --data_root /path/to/target/ImageFolder
```

writes Acc/QWK/AUROC/AUPR to a JSON file (see `results/cross_domain/`
for the full matrix reported in the paper).

## Pretrained checkpoints

Checkpoints (JOINT4 and the four ablation variants) will be released on
Zenodo; the link will be added here upon publication. All metrics needed
to verify the paper's tables are already included under `results/`.

## License

MIT (see `LICENSE`).
