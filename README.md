# GA-SCL: GA-SCL: Geometry-Conditioned Self-Supervised Contrastive Learning with Geometric Algebra Transformers for 3D Skeleton Action Recognition

> Official PyTorch implementation of **GA-SCL**, a geometry-aware self-supervised contrastive learning framework for 3D skeleton-based action recognition.

---

## News

- **[Coming soon]** Paper under preparation.
- **[Coming soon]** Training and evaluation scripts will be released.
- **[Coming soon]** Pretrained checkpoints and logs will be provided after publication/submission policy confirmation.

---

## Overview

GA-SCL is designed for self-supervised 3D skeleton-based action recognition. The method builds on a strong temporal--spatial contrastive learning baseline and introduces a geometry-aware branch based on Geometric Algebra Transformer (GATr) to enhance skeleton representation learning.

The framework aims to preserve the discriminative temporal--spatial representation while incorporating geometric cues from skeletal structure and motion.

---

## Highlights

- GATr-enhanced skeleton representation learning
- Motion-aware geometric contrastive modeling
- Geometry-guided self-supervised feature learning
- Robust 3D action representation learning

---

## Method

The proposed GA-SCL framework contains the following main components:

1. **Temporal--spatial skeleton encoder** for learning action-discriminative representations.
2. **Geometry-aware GATr branch** for modeling skeletal structure and motion geometry.
3. **Self-supervised contrastive objectives** for temporal, spatial, instance, and geometric representation learning.
4. **Protected downstream evaluation** using the main temporal--spatial feature representation.

A detailed method description will be added after the paper is finalized.

---

## Framework

A framework figure will be added here after final paper formatting.

```text
Input 3D Skeleton Sequence
        |
        |-- Temporal Encoder --> Temporal Feature
        |
        |-- Spatial Encoder  --> Spatial Feature
        |
        |-- GATr Geometry Branch --> Geometry Feature
        |
 Self-Supervised Contrastive Learning
        |
 Downstream Linear Evaluation
```

---

## Requirements

The tested environment will be updated after final experiments.

```bash
python >= 3.8
pytorch >= 1.10
numpy
scikit-learn
matplotlib
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

This repository supports skeleton-based action recognition datasets such as:

- NTU RGB+D 60
- NTU RGB+D 120
- PKU-MMD

Dataset downloading and preprocessing instructions will be added after paper completion.

Expected dataset structure:

```text
data/
  ntu60/
  ntu120/
  pku_mmd/
```

---

## Pretraining

Example command for self-supervised pretraining:

```bash
python pretraining.py \
  --pre-dataset ntu60 \
  --protocol cross_subject \
  --skeleton-representation joint \
  --epochs 451 \
  --batch-size 64 \
  --lr 0.01 \
  --use-gatr-branch
```

The final command will be updated after all experiments are fixed.

---

## Linear Evaluation

Example command for linear evaluation:

```bash
python action_classification.py \
  --pretrained ./checkpoints/path_to_checkpoint.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject \
  --finetune_skeleton_representation joint \
  --epochs 80 \
  --batch-size 1024 \
  --lr 30
```

---

## t-SNE Visualization

Optional t-SNE visualization can be generated after linear evaluation:

```bash
python action_classification.py \
  --pretrained ./checkpoints/path_to_checkpoint.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject \
  --finetune_skeleton_representation joint \
  --run-tsne-after
```

---

## Results

Results will be added after the paper is finalized.

### NTU RGB+D 60

| Method | Protocol | Top-1 Accuracy (%) |
|---|---:|---:|
| SCD-Net baseline | X-Sub | 86.6 |
| GA-SCL | X-Sub | 86.9 |

### NTU RGB+D 120

| Method | Protocol | Top-1 Accuracy (%) |
|---|---:|---:|
| GA-SCL | X-Sub |78.6 |
| GA-SCL | X-Set | 79.8 |

---

## Ablation Study

Ablation results will be added after final experiments.

| Variant | Description | Accuracy (%) |
|---|---|---:|
| Baseline | Temporal--spatial contrastive learning | TBD |
| + GATr | Geometry-aware branch | TBD |
| + Motion Geometry | Motion-aware geometric tokens | TBD |
| Full GA-SCL | Final proposed model | TBD |

---

## Checkpoints

Pretrained checkpoints will be released after paper acceptance or according to the publication policy.

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{usman2026gascl,
  title   = {GA-SCL: Geometry-Aware Self-Supervised Contrastive Learning with Geometric Algebra Transformer for 3D Skeleton-Based Action Recognition},
  author  = {Usman, Muhammad and others},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under preparation}
}
```

The final BibTeX will be updated after publication.

---

## Acknowledgements

This repository builds upon prior research in self-supervised skeleton-based action recognition and geometric representation learning. Detailed acknowledgements will be added in the final release.

---

## License

The license will be added when the repository is made public.

---

## Contact

For questions, please contact:

```text
Muhammad Usman
usmanszu@qq.com
```
