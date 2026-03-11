# Toy Dataset Generation

> Part B — Advanced Machine Learning Mid-Semester Exam (Roll No. 230064)
> **Paper:** "Object Detection with Discriminatively Trained Part-Based Models"
> Felzenszwalb, Girshick, McAllester, Ramanan — IEEE TPAMI, 2010

---

## Description

This directory contains the toy dataset used in Part B of the mid-semester exam. The dataset is **artificially generated** using [scikit-learn](https://scikit-learn.org/) to simulate simplified object detection features for experimentation.

The dataset is designed to mirror the core feature structure of the **Deformable Part Model (DPM)** described in the paper. Each sample is composed of three groups of features:

| Feature Group            | Description                                                                                                                                                                   |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Root features**        | Global object representation — analogous to the root filter response in the DPM, capturing coarse, whole-object appearance.                                                   |
| **Part features**        | Local features corresponding to individual object parts — analogous to the part filter responses that capture fine-grained detail.                                            |
| **Deformation features** | Spatial displacement values representing how far each part is from its expected anchor position — analogous to the quadratic deformation penalty in the DPM scoring function. |

This construction mimics the DPM scoring function (Equation 2 of the paper):

```
score = F0 · phi_root + sum_i [ Fi · phi_part_i - di · Phi_d(dx_i, dy_i) ] + b
```

where the detection score combines root filter response, part filter responses, and deformation costs.

---

## Usage

The dataset is **generated dynamically** inside the Jupyter notebooks rather than stored as static files. This ensures full reproducibility and transparency — every step of data creation is visible in the code.

- The dataset is produced using `sklearn.datasets.make_classification` combined with custom deformation feature construction.
- **Random seeds are fixed** (`RANDOM_SEED = 42`) across all notebooks to guarantee identical data on every run.
- Each notebook is **self-contained** and regenerates the dataset from scratch, so there are no cross-notebook file dependencies.

---

## Example Dataset Generation

Below is a minimal example showing how the dataset is generated:

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    n_classes=2,
    random_state=42
)
```

### Feature Breakdown

The generated feature matrix is structured as follows:

```
root_features        = X[:, 0:2]   # First 2 features  — global object appearance
part_features        = X[:, 2:4]   # Next 2 features   — local part appearance
deformation_features = X[:, 4:6]   # Last 2 features   — simulated spatial displacement
```

In the full implementation used in the notebooks, the dataset is extended to **14 features**:

| Index | Feature                | Group                     |
| ----- | ---------------------- | ------------------------- |
| 0–1   | `root_f0`, `root_f1`   | Root (global appearance)  |
| 2–3   | `part1_f0`, `part1_f1` | Part 1 (local appearance) |
| 4–5   | `part2_f0`, `part2_f1` | Part 2 (local appearance) |
| 6–7   | `part3_f0`, `part3_f1` | Part 3 (local appearance) |
| 8–9   | `part1_dx`, `part1_dy` | Part 1 deformation        |
| 10–11 | `part2_dx`, `part2_dy` | Part 2 deformation        |
| 12–13 | `part3_dx`, `part3_dy` | Part 3 deformation        |

Deformation features are drawn from different distributions for positive and negative samples — positives have small displacements (parts near anchors) while negatives have larger, random displacements.

---

## Purpose

This synthetic dataset allows testing the following core ideas from the paper:

- **Rigid vs. part-based models** — Comparing a root-only (rigid template) baseline against a full part-based model to verify that parts improve classification.
- **Effect of deformation penalties** — Measuring how the quadratic deformation cost contributes to discriminative power beyond part appearance alone.
- **Ablation experiments** — Systematically removing part features or deformation features to quantify each component's contribution (corresponding to Section 6, Table 3 of the paper).
- **Failure mode analysis** — Constructing scenarios where the DPM's unimodal deformation assumption is violated (e.g., bimodal part positions) to identify when the model breaks down.

---

## Limitations

This dataset is a **simplified abstraction** compared to the real dataset used in the paper:

| Aspect            | Paper (PASCAL VOC)                               | Toy Dataset                           |
| ----------------- | ------------------------------------------------ | ------------------------------------- |
| Data type         | Real images with complex backgrounds             | Synthetic numerical features          |
| Task              | Object detection (localization + classification) | Binary classification                 |
| Features          | HOG descriptors computed from pixel gradients    | Directly generated numerical features |
| Scale             | ~10,000 images, 20 object categories             | 500 samples, 2 classes                |
| Spatial structure | True 2D spatial layout of parts                  | Simulated displacement values         |
| Evaluation metric | Average Precision (AP) with IoU overlap          | Accuracy, F1-score, AUC-ROC           |

Instead of real images and HOG features, the toy dataset represents object structure using numerical features that capture the **conceptual decomposition** (root vs. parts vs. deformation) of the DPM. This is a deliberate design choice for pedagogical clarity and CPU feasibility.

---

## References

- P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan. _"Object Detection with Discriminatively Trained Part-Based Models."_ IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2010.
- N. Dalal and B. Triggs. _"Histograms of Oriented Gradients for Human Detection."_ CVPR, 2005.
