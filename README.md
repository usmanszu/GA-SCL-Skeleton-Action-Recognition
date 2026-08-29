# GA-SCL: Geometry-Conditioned Self-Supervised Contrastive Learning with Geometric Algebra Transformers for 3D Skeleton Action Recognition

> Official PyTorch implementation of **GA-SCL**, a geometry-conditioned self-supervised contrastive learning framework for 3D skeleton action recognition.

---

## News

- **[Coming soon]** Training and evaluation code will be released.
- **[Coming soon]** Pretrained checkpoints and experiment logs will be provided according to the publication and repository release policy.

---

## Overview

**GA-SCL** introduces explicit motion-aware skeletal geometry into
self-supervised temporal--spatial representation learning.

Given a 3D skeleton sequence, GA-SCL constructs projective geometric algebra
(PGA) representations from joint positions, bone vectors, joint
displacements, and bone displacements. These representations are processed by
a **Geometric Algebra Transformer (GATr)** followed by a **Pin-invariant
readout** to obtain joint--window geometric descriptors.

Rather than using geometry as an additional downstream recognition stream,
GA-SCL uses the resulting geometric token bank to condition both temporal and
spatial representations through **zero-gated cross-attention**. A dedicated
geometric self-contrastive objective further structures the geometric
representation during pretraining.

For strict linear evaluation, the complete pretrained encoder is frozen and
classification uses only

\[
[\mathbf{v}_t;\mathbf{v}_s],
\]

without directly appending the standalone geometric representation
\(\mathbf{v}_g\).

---

## Highlights

- **Motion-aware PGA representation** of joint, bone, and displacement information.
- **GATr + Pin-invariant readout** for structured skeletal geometric encoding.
- **Joint--window geometric token bank** preserving localized geometric information.
- **Zero-gated cross-attention** conditioning of both temporal and spatial streams.
- **Geometric self-contrastive learning** for the auxiliary geometric representation.
- **Strict downstream evaluation** using only \([\mathbf{v}_t;\mathbf{v}_s]\).
- **Parameter-matched Cartesian control** to evaluate whether gains arise from additional capacity alone.

---

## Method

GA-SCL consists of four main components:

1. **Temporal--spatial backbone**  
   Learns complementary temporal and spatial skeleton representations.

2. **Motion-aware geometric pathway**  
   Joint positions, bone vectors, joint displacements, and bone displacements
   are represented in PGA space and processed using GATr.

3. **Geometric self-supervision and conditioning**  
   A geometric self-contrastive objective structures the global geometric
   representation, while separate zero-gated cross-attention modules allow
   temporal and spatial tokens to query the joint--window geometric token bank.

4. **Strict downstream evaluation**  
   During linear evaluation, the pretrained feature extractor is frozen and
   the classifier receives only the temporal--spatial representation
   \([\mathbf{v}_t;\mathbf{v}_s]\).

---

## Framework

The final framework figure will be added with the public code release.

```text
                      Input 3D Skeleton Sequence
                                 |
             +-------------------+-------------------+
             |                   |                   |
      Temporal Path        Spatial Path       Motion-Aware Geometry
             |                   |                   |
             |                   |          PGA Representation
             |                   |                   |
             |                   |             GATr + PinInv
             |                   |                   |
             |                   |        Joint--Window Token Bank
             |                   |             + Global v_g
             |                   |                   |
             +------ Zero-Gated Cross-Attention -----+
             |                   |
        Temporal v_t        Spatial v_s
             |                   |
             +---------+---------+
                       |
                  [v_t ; v_s]
                       |
             Strict Linear Evaluation
```

---

## Requirements

The exact tested software environment will be provided with the public
release.

Main dependencies include:

```text
Python
PyTorch
NumPy
scikit-learn
matplotlib
tqdm
THOP
```

Install the released dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The experiments use:

- **NTU RGB+D 60**
- **NTU RGB+D 120**
- **PKU-MMD Phase I**

Dataset preprocessing and directory instructions will be provided with the
code release.

Expected structure:

```text
data/
├── ntu60/
├── ntu120/
└── pku_mmd/
```

---

## Self-Supervised Pretraining

The final GA-SCL configuration uses:

- 4 temporal windows
- 4 multivector channels
- 4 scalar channels
- motion-aware geometric encoding
- GATr hidden multivector width: 8
- GATr hidden scalar width: 32
- 2 GATr blocks
- 4 attention heads
- zero-gated cross-attention
- temporal + spatial conditioning
- geometric self-contrastive weight: 0.03
- geometric-loss warm-up: 100 epochs
- conditioning activation: epoch 150

Example NTU RGB+D 60 X-Sub pretraining command:

```bash
CUDA_VISIBLE_DEVICES=0 python pretraining.py \
  --lr 0.01 \
  --batch-size 64 \
  --encoder-t 0.2 \
  --encoder-k 8192 \
  --encoder-m 0.999 \
  --encoder-dim 128 \
  --schedule 351 \
  --epochs 451 \
  --pre-dataset ntu60 \
  --protocol cross_subject \
  --skeleton-representation joint \
  --use-gatr-branch \
  --gatr-condition-main \
  --gatr-condition-mode xattn \
  --film-start-epoch 150 \
  --gatr-use-motion \
  --gatr-mv-channels 4 \
  --gatr-scalar-channels 4 \
  --gatr-hidden-mv-channels 8 \
  --gatr-hidden-s-channels 32 \
  --gatr-num-windows 4 \
  --gatr-mask-persons \
  --lambda-g 0.03 \
  --g-warmup-epochs 100 \
  --workers 8 \
  --amp
```

---

## Linear Evaluation

GA-SCL follows a **strict frozen linear-evaluation protocol**. The complete
pretrained feature extractor is frozen and only a linear classifier is
optimized.

The downstream representation is

\[
[\mathbf{v}_t;\mathbf{v}_s],
\]

and the standalone geometric feature \(\mathbf{v}_g\) is **not appended** to
the classifier input.

Example NTU RGB+D 60 X-Sub command:

```bash
CUDA_VISIBLE_DEVICES=0 python action_classification.py \
  --pretrained ./checkpoints/ga_scl/checkpoint.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject \
  --finetune_skeleton_representation joint \
  --epochs 80 \
  --batch-size 256 \
  --lr 30 \
  --use-gatr-branch \
  --gatr-eval-mode baseline \
  --gatr-condition-main \
  --gatr-condition-mode xattn \
  --gatr-use-motion \
  --gatr-num-windows 4 \
  --gatr-mask-persons \
  --gatr-mv-channels 4 \
  --gatr-scalar-channels 4 \
  --gatr-hidden-mv-channels 8 \
  --gatr-hidden-s-channels 32
```

---

## Results

### Linear Evaluation

| Method | NTU60 X-Sub | NTU60 X-View | NTU120 X-Sub | NTU120 X-Set | PKU-MMD Phase I |
|---|---:|---:|---:|---:|---:|
| SCD-Net (published) | 86.6 | 91.7 | 76.9 | 80.1 | **91.9** |
| SCD-Net (our reproduction) | 86.3 | n/e | n/e | n/e | n/e |
| **GA-SCL** | **86.9** | **92.1** | **79.2** | **80.2** | 89.8 |

`n/e` denotes protocols not evaluated in our controlled SCD-Net reproduction.

Under the controlled NTU RGB+D 60 X-Sub experiment, GA-SCL improves the
reproduced SCD-Net result from **86.3% to 86.9%**.

---

## 1-NN Evaluation

| Method | NTU60 X-Sub | NTU60 X-View | NTU120 X-Sub | NTU120 X-Set |
|---|---:|---:|---:|---:|
| SCD-Net | 76.2 | 86.8 | 59.8 | 65.7 |
| **GA-SCL** | **76.9** | **87.5** | **63.3** | **66.2** |

---

## Ablation Studies

Key ablations evaluate:

- geometric self-supervision and conditioning;
- static versus motion-aware geometry;
- body-frame canonicalization;
- FiLM versus joint--window cross-attention;
- temporal versus temporal--spatial conditioning;
- temporal-window resolution;
- downstream use of \(\mathbf{v}_g\);
- parameter-matched Cartesian processing.

Selected NTU RGB+D 60 X-Sub results:

| Configuration | Top-1 (%) |
|---|---:|
| SCD-Net (reproduced) | 86.3 |
| Auxiliary geometry only | 86.7 |
| Conditioning only | 86.1 |
| **GA-SCL** | **86.9** |

### Parameter-Matched Cartesian Control

| Processor | Gradient-Updated Params (M) | Top-1 (%) |
|---|---:|---:|
| Cartesian Transformer | 102.8551 | 85.3 |
| **GA-SCL (GATr)** | 102.8542 | **86.9** |

The Cartesian control is effectively parameter matched with GA-SCL. The
comparison therefore indicates that the observed improvement cannot be
explained by increased parameter count alone.

---

## Computational Efficiency

Controlled downstream profiling on NTU RGB+D 60 X-Sub:

| Method | Params (M) | Est. GFLOPs/clip | Throughput (clips/s) | Peak Memory (MB) |
|---|---:|---:|---:|---:|
| SCD-Net (reproduced) | 74.693 | 7.1066 | **1159.15** | **855.82** |
| **GA-SCL** | 81.071 | 7.3676 | 1001.90 | 879.86 |

GA-SCL introduces moderate additional inference complexity in exchange for
improved recognition accuracy.

---

## Visualization

The repository will include scripts for:

- t-SNE visualization;
- confusion-matrix analysis;
- representation analysis.

Example:

```bash
python action_classification.py \
  --pretrained ./checkpoints/ga_scl/checkpoint.pth.tar \
  --finetune-dataset ntu60 \
  --protocol cross_subject \
  --generate-tsne
```

---

## Checkpoints

Pretrained checkpoints will be released according to the publication and
repository release policy.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{usman2026gascl,
  title   = {GA-SCL: Geometry-Conditioned Self-Supervised Contrastive Learning
             with Geometric Algebra Transformers for 3D Skeleton Action Recognition},
  author  = {Usman, Muhammad and others},
  journal = {Neurocomputing},
  year    = {2026},
  note    = {Under review}
}
```

The BibTeX entry will be updated after publication.

---

## Acknowledgements

This implementation builds upon prior work in self-supervised skeleton action
recognition and geometric representation learning. Appropriate acknowledgements
and links to upstream repositories will be provided with the public release.

---

## License

License information will be provided when the repository is publicly released.

---

## Contact

For questions, please contact:

```text
Muhammad Usman
usmanszu@qq.com
```
