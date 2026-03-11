# Report Outline — Part B: Object Detection with Discriminatively Trained Part-Based Models

**Paper:** Felzenszwalb, Girshick, McAllester, Ramanan. "Object Detection with Discriminatively Trained Part-Based Models." IEEE TPAMI, 2010.

**Student Roll Number:** 230064  
**Target Length:** 4 pages

---

## 1. Paper Summary (0.75 pages)

### 1.1 Problem Statement

- Object detection in natural images: localizing and classifying objects under variation in pose, viewpoint, and appearance.
- Prior rigid template methods (Dalal & Triggs HOG) cannot handle articulation or viewpoint changes.

### 1.2 Proposed Method: Deformable Part Model (DPM)

- Star-structured model: root filter (global) + part filters (local, high-resolution) + deformation penalty (quadratic springs).
- Scoring function: score = root response + Σ(part responses − deformation costs) + bias.
- Feature pyramid of HOG descriptors at multiple scales.
- Efficient inference using generalized distance transform.

### 1.3 Training: Latent SVM

- Part positions as latent variables (not annotated).
- Coordinate descent: alternate between fixing latent variables and optimizing SVM weights.
- Hard negative mining to focus on difficult background examples.

### 1.4 Key Results

- State-of-the-art on PASCAL VOC 2006, 2007, 2008.
- Mean AP ~28.7% on VOC 2007 (significant improvement over rigid baseline ~22%).

---

## 2. Reproduction Setup (0.75 pages)

### 2.1 Dataset

- Synthetic part-based dataset: 500 samples, 14 features (root + parts + deformation).
- Designed to mirror DPM's conceptual structure on a CPU-friendly scale.
- Limitations vs. PASCAL VOC: no real images, no spatial structure, classification instead of detection.

### 2.2 Implementation

- Baseline: Linear SVM on root features only (Dalal & Triggs analog).
- DPM model: Linear SVM on root + part + quadratic deformation features.
- Hard negative mining: iterative sample reweighting approach.
- All models trained with scikit-learn LinearSVC, C=1.0, seed=42.

### 2.3 Evaluation Protocol

- 80/20 stratified train/test split.
- Metrics: accuracy, precision, recall, F1-score, AUC-ROC, confusion matrix.

---

## 3. Results vs Paper Results (0.75 pages)

### 3.1 Our Results

- Table: Rigid vs. DPM vs. DPM+HNM — accuracy, F1, AUC.
- Confusion matrices for all three models.
- ROC curve comparison.

### 3.2 Paper Results (Section 6, Table 3)

- Root only: ~22% mAP, Root+Parts: ~26% mAP, Full model: ~28.7% mAP.

### 3.3 Analysis of Differences

- Different task (classification vs. detection), metric (accuracy vs. AP), and data complexity.
- Qualitative trend preserved: rigid < DPM < DPM+HNM.
- Higher absolute accuracy due to simpler toy dataset.
- No latent variable optimization in our implementation.

---

## 4. Ablation Findings (0.75 pages)

### 4.1 Ablation 1: Remove Part Filters

- Root-only model: largest performance drop.
- Confirms part filters add significant discriminative power beyond global shape.
- Matches paper's Table 3 findings.

### 4.2 Ablation 2: Remove Deformation Penalty

- Root + parts without spatial constraints: moderate performance drop.
- Confirms deformation penalty provides complementary signal to part appearance.
- Without spatial consistency, model accepts implausible part configurations.

### 4.3 Summary Table

- Full model vs. both ablations: accuracy, F1, confusion matrices side-by-side.

---

## 5. Failure Mode Analysis (0.5 pages)

### 5.1 Failure Scenario

- Extreme bimodal deformation + weak root features.
- Violates Assumption 3 (quadratic/unimodal deformation penalty).

### 5.2 Results

- All model variants perform near chance (~50%) on the failure dataset.
- DPM's quadratic penalty cannot model bimodal part positions.

### 5.3 Suggested Fix

- Mixture-of-Gaussians deformation model: per-part multi-anchor positions.
- Each mode has its own quadratic penalty; model selects best mode per part.

---

## 6. Reflection (0.5 pages)

### 6.1 What Worked

- The toy dataset effectively demonstrated the DPM's core principles.
- Ablation studies clearly quantified component contributions.
- Paper references grounded every design decision.

### 6.2 What I Learned

- Trade-off between model expressiveness and computational efficiency (quadratic penalty enables distance transform but limits deformation modeling).
- Importance of feature design: HOG was state-of-the-art pre-CNN but has fundamental limitations.
- Latent variable training is powerful for avoiding expensive annotations.

### 6.3 Limitations of This Reproduction

- Synthetic features vs. real HOG on images.
- Classification vs. detection (no localization component).
- No mixture components (multi-viewpoint) in our implementation.
- Hard negative mining simplified to sample reweighting.

### 6.4 Historical Context

- DPM won PASCAL VOC "lifetime achievement" prize.
- Represented peak of hand-crafted feature era before deep learning.
- Key ideas (multi-scale, deformable models, hard mining) carried forward into modern detectors (Deformable DETR, OHEM, FPN).

---

## References

1. Felzenszwalb, P.F., Girshick, R.B., McAllester, D., Ramanan, D. "Object Detection with Discriminatively Trained Part-Based Models." IEEE TPAMI, 32(9):1627–1645, 2010.
2. Dalal, N., Triggs, B. "Histograms of Oriented Gradients for Human Detection." CVPR, 2005.
3. Everingham, M., et al. "The PASCAL Visual Object Classes (VOC) Challenge." IJCV, 2010.
