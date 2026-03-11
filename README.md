# 230064 — Advanced Machine Learning Mid-Semester Exam

> **Student:** Aman Kumar (Roll No. 230064)  
> **Paper:** _"Object Detection with Discriminatively Trained Part-Based Models"_ — Felzenszwalb, Girshick, McAllester, Ramanan (IEEE TPAMI, 2010)

---

## Overview

This repository contains my mid-semester examination submission for the Advanced Machine Learning course. The exam involves selecting a qualifying ML research paper, understanding its core ideas, reproducing key results on a toy dataset, and performing deeper analysis through ablation studies and failure mode investigation.

The chosen paper introduces **Deformable Part Models (DPM)** — a framework that combines root (global) and part (local) HOG filters with a quadratic deformation penalty, trained via **Latent SVMs** with hard negative mining, for object detection on PASCAL VOC.

---

## Repository Structure

```
├── README.md
├── partA/                          # Paper selection
│   └── llm usage partA.json        # LLM usage log for Part A
└── partB/                          # Implementation & analysis
    ├── requirements.txt            # Python dependencies
    ├── data/                       # Dataset documentation
    │   └── README.md
    ├── results/                    # Saved outputs
    │   └── model_comparison.csv
    ├── task_1_1.ipynb              # Core contribution / architecture
    ├── task_1_2.ipynb              # Key assumptions
    ├── task_1_3.ipynb              # Baseline and improvement
    ├── task_2_1.ipynb              # Dataset selection & construction
    ├── task_2_2.ipynb              # Reproduce one contribution
    ├── task_2_3.ipynb              # Results and comparison
    ├── task_3_1.ipynb              # Two-component ablation study
    ├── task_3_2.ipynb              # Failure mode analysis
    └── llm_task_*.json             # LLM usage logs per task
```

---

## Part A — Paper Selection

Selection and feasibility analysis of the DPM paper. The paper qualifies because it uses a well-defined ML algorithm (Latent SVM), is reproducible on CPU, and operates on publicly available data (PASCAL VOC). Documented in `partA/llm usage partA.json`.

---

## Part B — Implementation & Analysis

All notebooks use a **synthetic toy dataset** (500 samples, 14 features) that mirrors DPM's feature structure — root features (global appearance), part features (local appearance), and deformation features (spatial displacement). Data is generated dynamically with a fixed random seed (`RANDOM_SEED = 42`) for full reproducibility.

### Task Group 1: Paper Understanding

| Notebook         | Topic                  | Description                                                                                                                     |
| ---------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `task_1_1.ipynb` | Core Contribution      | Full DPM pipeline walkthrough: HOG pyramid → root/part filters → deformation modeling → Latent SVM → hard negative mining → NMS |
| `task_1_2.ipynb` | Key Assumptions        | Four core assumptions (star topology, HOG sufficiency, unimodal deformation, reliable annotations) with violation scenarios     |
| `task_1_3.ipynb` | Baseline & Improvement | Rigid HOG + linear SVM baseline vs. DPM improvements; failure cases for DPM                                                     |

### Task Group 2: Reproduction

| Notebook         | Topic                | Description                                                                                               |
| ---------------- | -------------------- | --------------------------------------------------------------------------------------------------------- |
| `task_2_1.ipynb` | Dataset Construction | Synthetic dataset with class-dependent deformation distributions; feature visualizations                  |
| `task_2_2.ipynb` | Model Training       | Three models: rigid root-only SVM, DPM-style SVM (root + parts + deformation), DPM + hard negative mining |
| `task_2_3.ipynb` | Evaluation           | Confusion matrices, ROC curves, comparison with paper's PASCAL VOC results                                |

### Task Group 3: Analysis

| Notebook         | Topic          | Description                                                                                                                        |
| ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `task_3_1.ipynb` | Ablation Study | Component ablation — remove parts (root-only) and remove deformation (root + parts only) to quantify each component's contribution |
| `task_3_2.ipynb` | Failure Modes  | Adversarial dataset violating the unimodal deformation assumption (bimodal part positions); proposes mixture-of-Gaussians fix      |

---

## Key Results

| Model                      | Accuracy | Precision | Recall | F1 Score |
| -------------------------- | -------- | --------- | ------ | -------- |
| Rigid (Root-Only)          | 0.63     | 0.71      | 0.44   | 0.54     |
| Part-Based (DPM)           | 1.00     | 1.00      | 1.00   | 1.00     |
| DPM + Hard Negative Mining | 1.00     | 1.00      | 1.00   | 1.00     |

The qualitative trend matches the paper: rigid baseline < DPM < DPM + HNM. The toy dataset yields higher absolute numbers because it is deliberately designed to be separable with part and deformation features, while the rigid model struggles with only root features.

In the failure mode analysis (Task 3.2), all models drop to near-chance accuracy (~50%) when the quadratic deformation assumption is violated.

---

## Setup

```bash
cd partB
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open any notebook in Jupyter or VS Code and run all cells.

---

## LLM Usage

LLM interactions are documented per task:

- **Part A:** `partA/llm usage partA.json` (ChatGPT / GPT-5.2)
- **Part B:** `partB/llm_task_*.json` (GitHub Copilot / Claude)
