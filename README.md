# ***GanCtrl***: An AI-Based Pilot Study for Modeling Synthetic Controls from Treatment Data in Preclinical Toxicology

**GanCtrl** (GAN-based synthetic control) is a conditional VAE–GAN model that translates treatment-derived clinical-pathology profiles into their time-matched control equivalents—synthetic controls using the Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System) rat ***in vivo*** repeat-dose data across 38 clinical pathology measurements. 

GanCtrl is trained on high-dose treatment samples and couples a context-conditioned encoder with an attention-aware, biologically informed decoder acting as the generator, together with a discriminator to generate physiologically coherent virtual controls.

---

## Table of Contents

- [Introduction](#introduction)
- [Code Scripts](#code-scripts)
  - [GanCtrl Model Development, Training & Predictions](#ganctrl-model-development-training--predictions)
  - [Statistical Agreement](#statistical-agreement)
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

### **Toxicity Assessment**

Scripts focusing on whether synthetic controls can substitute real controls for **toxicological decision-making** i.e., it's ability to replicate toxicity outcomes. We do this by implementing concordance analysis comparing sample-level abnormal/normal classifications derived using real-control references versus synthetic-control references. 
We first perform concordance analysis on the **training set** to calibrate a z-score decision threshold, and then fix this threshold and apply it to the **held-out test set**. The resulting GanCtrl-based calls are benchmarked against **Inter-lab baselines** (cross-study comparisons), and **Intra-lab baselines** (within-study comparisons), to contextualize the performance of synthetic controls relative to biological and technical variability.

- **z-Score Calculation**: For each compound–time group and measurement, every treatment sample was standardized against each control source using a control-referenced z-score:  
  $z = \dfrac{x_{\text{treatment}} - \mu_{\text{control}}}{\sigma_{\text{control}}}$

- **Concordance Accuracy**: Concordance accuracy was defined by cross-tabulating synthetic-control-referenced calls against real-referenced calls at the sample level into true positives (TP, abnormal under both), true negatives (TN, normal under both), false positives (FP, abnormal under GanCtrl but normal under real), and false negatives (FN, abnormal under real but normal under GanCtrl), and computed as:  
  $\text{Concordance Accuracy} = \dfrac{TP + TN}{TP + TN + FP + FN}$

**Files**:
- [`interlab_concordance.py`](./baseline/interlab_concordance.py) - Script for concordance analysis for cross-study comparisons.
- [`intralab_concordance.py`](./baseline/intralab_concordance.py) -Script for concordance analysis in within-study comparisons.
- [`train_concordnace.py`](./evaluation/train_concordance.py) - Concordance analysis in the training set for threshold calibration.
- [`test_concordance.py`](./evaluation/test_concordance.py) - Concordance analysis in the test set for evaluating the ability of synthetic controls in toxicity assessment.

This part of the pipeline is intended to answer:  
> *“If I replace the concurrent control arm with GanCtrl-generated synthetic controls, do I arrive at the same toxicological conclusions?”*

---

### **Biological Significance**

Scripts exploring whether GanCtrl preserves **biologically meaningful relationships** among measurements (biomarkers). This module includes two complementary analyses—(a) **measurement-elevation consistency** and (b) **coordinated multi-measurement responses**.

- **Measurement-elevation consistency** (single-measurement level):  
  - Checks whether elevations observed in treated samples with real controls (e.g., ALT, CRE, TBIL) are consistently reproduced when using synthetic controls instead.
  - Compares the direction and magnitude of shift when treatment compared to controls across compound-time groups.

- **Coordinated multi-measurement responses** (co-elevation):  
  - Checks whether biomarker co-regulation patterns are preserved when real controls are substituted with synthetic controls. 
    - ALT–AST for hepatpocellular injury,
    - BUN–CRE for renal function, etc.
      
These analyses help validate that the model does more than fit marginal distributions; it also respects **multi-marker biological structure** and preserves biologically plausible co-response patterns.

**Files**:
- [`biomarker_elevation.py`](./evaluation/biomarker_elevation.py) - Script for analyzing elevation calls in treated samples when compared with real and synthetic controls.
- [`co_elevation.py`](./evaluation/co-elevation.py) -Script for assessing the coordinated multi-biomarker response patterns such as co-elevation trends between measurements.


---

## Data Files

This repository expects preprocessed CSVs derived from Open TG-GATEs. The raw TG-GATEs data are *not* distributed here; you must obtain them separately and reproduce the preprocessing if needed from the official Open TG-GATEs repository:  
[Open TG-GATEs download page](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html).

Typical input files:

| Filepath                                                        | Description                                                                                          |
|-----------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| [`repeat_train_control.csv`](data/repeat_train_control.csv)     | Training data with control profiles consisting clinical-pathology measurements and metadata.        |
| [`repeat_test_control.csv`](data/repeat_test_control.csv)       | Test data with time-matched control profiles.                                                       |
| [`repeat_train_treatment.csv`](data/repeat_train_treatment.csv) | Training data with treatment profiles, including high-dose arms used for modeling.                  |
| [`repeat_test_treatment.csv`](data/repeat_test_treatment.csv)   | Test data with treatment profiles.                                                                  |
| [`body_wt.csv`](data/body_wt.csv)                               | Longitudinal body weight measurements with `PROGRESS_TIME`, used to derive latest BODY_WEIGHT per animal. |

Key column expectations (used in the scripts):

- **Metadata / keys**:
  - `COMPOUND_NAME`, `DOSE_LEVEL`, `SACRIFICE_PERIOD`, `EXP_ID`, `GROUP_ID`, `INDIVIDUAL_ID`, `ID`
- **Outcome features** (38 clinical pathology profiles)
  - Clinical chemistry & hematology markers (ALT, AST, ALP, LDH, TBIL, BUN, CRE, WBC, etc.).
  - Out of 38, nine are liver-specific (ALP, ALT, AST, GTP, LDH, TBIL, DBIL, RALB, TP) and seven are kidney-specific (BUN, CRE, Ca, Cl, Na, IP, K) as they known for their established relevance to hepatotoxicity and nephrotoxicity.

Model outputs:

- `g_model1_*.h5` — Generator checkpoints  
- `d_model1_*.h5` — Discriminator checkpoints  
- `composite_model1_*.h5` — Composite checkpoints  
- `predictions_encoded/` — Encoded-space predictions (if enabled)  
- `predictions_decoded/` — Fully decoded predictions (means + samples)  

Adjust paths and prefixes as needed for your environment.

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/GanCtrl.git
   cd GanCtrl
