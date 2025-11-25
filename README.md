# ***GanCtrl***: An AI-Based Pilot Study for Modeling Synthetic Controls from Treatment Data in Preclinical Toxicology

**GanCtrl** (GAN-based synthetic control) is a conditional VAE–GAN model that translates treatment-derived clinical-pathology profiles into their time-matched control equivalents—synthetic controls using the Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System) rat ***in vivo*** repeat-dose data across 38 clinical pathology measurements. 

GanCtrl is trained on high-dose treatment samples and couples a context-conditioned encoder with an attention-aware, biologically informed decoder acting as the generator, together with a discriminator to generate physiologically coherent virtual controls.

---

## Table of Contents

- [Introduction](#introduction)
- [Code Scripts](#code-scripts)
  - [GanCtrl Model Development, Training & Predictions](#ganctrl-model-development-training--predictions)
  - [Statistical Agreement](#statistical-analysis)
  - [Toxicity Assessment](#toxicity-assessment)
  - [Biological Significance](#biological-significance)
- [Data Files](#data-files)
- [Installation](#installation)
- [License](#license)
  
---

## Introduction

This repository contains the full experimental code for **GanCtrl**, a one-sided conditional VAE–GAN framework that:

- Learns to map **high-dose treatment** clinical pathology profiles to their **time-matched control** equivalents.
- Uses **body weight**, **timepoint**, **replicate identity**, and **study-specific clusters** as conditioning variables.
- Enforces **variance-aware Gaussian NLL**, **adversarial loss**, **TBIL range constraints loss**, **biological correlation preservation loss**, and **batch-level mean matching** to preserve organ-level patterns and realistic dispersion.
  
The implementation is designed for reproducibility (fixed seeds, deterministic TensorFlow ops):

- **Model definition & training**
- **Inference / prediction of synthetic controls**
- **Downstream agreement statistics and toxicological interpretation**
  
---

## Code Scripts

### **[GanCtrl Model Development, Training & Predictions](./training)**

This contains the core code for developing and training the **GanCtrl** framework. It also includes the code for generating synthetic controls. 

**Files**:
- [`ganctrl_training.py`](./training/ganctrl_training.py) - End-to-end **training script** for the one-sided treatment→control CVAE–GAN
- [`train_test_samples.py`](./training/train_test_samples.py) - **Inference / prediction script** to generate synthetic controls from trained checkpoints
  
> **Note:** The actual filenames in your repo may differ slightly (e.g. `*_cv2.py`, `*_cv5.py`). This README assumes descriptive names; please update them if needed.

---

### **Statistical Agreement**

Downstream scripts for evaluating model performance i.e., how well do synthetic controls agree with real controls. The performance of GanCtrl was assessed on the test set using two metrices: cosine similarity and root mean squared error (RMSE). Three baselines were derived from time-matched real controls:

- **Inter-laboratory Baseline** -  This group represents cross-study measurements (same vehicle, different laboratory)
- **Intra-laboratory Baseline** - This group represents within-study measurements (same vehicle, same laboratory)
- **Replicate Control** - Measurements were calculated between biological replicates within a treatment, providing a benchmark for real biological performance
- **GanCtrl** - Measurements between each synthetic profile and its corresponding real control profile

 **Files**:
  - [`interlab_cosine.py`](./baseline/interlab_cosine.py) - Script for calculating the cosine similarity for the inter-lab baseline.
  - [`intralab_cosine.py`](./baseline/intralab_cosine.py) - Script for calculating the cosine similarity for the intra-lab baseline.
  - [`replicate_control_cosine.py`](./baseline/replicate_control_cosine.py) - For calculating the cosine similarity for the replicate control baseline.
  - [`cosine.py`](./evaluation/cosine.py) - For calculating the cosine similarity for GanCtrl group (synthetic control profiles).
  - [`interlab_rmse.py`](./baseline/interlab_rmse.py) - Script for calculating the RMSE for the inter-lab baseline.
  - [`intralab_rmse.py`](./baseline/intralab_rmse.py) - Script for calculating the RMSE for the intra-lab baseline.
  - [`rmse.py`](./evaluation/rmse.py) - For calculating the rmse for GanCtrl group (synthetic control profiles).
  - [`replicate_control_rmse.py`](./baseline/replicate_control_rmse.py) - For calculating the RMSE for the replicate control baseline.
    
---

### **[Toxicity Assessment]

Scripts focusing on whether synthetic controls can substitute real controls for **toxicological decision-making**.

- `analysis/toxicity_assessment.py` / `analysis/toxicity_assessment.ipynb`  
  Uses real vs synthetic controls to recompute standard toxicity calls, e.g.:

- **Elevation flags** for clinical chemistry / hematology markers:
  - Threshold-based or ULN-based rules.
- Classification of **treatment groups** as:
  - Non-toxic / mild / moderate / severe for given organ systems.
- Concordance metrics between:
  - Decisions made using **real controls** and
  - Decisions made using **synthetic controls** (e.g. % agreement per marker / timepoint / compound).

This part of the pipeline is intended to answer:  
> *“If I replace the concurrent control arm with GanCtrl-generated virtual controls, do I arrive at the same toxicological conclusions?”*

