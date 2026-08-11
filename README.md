# ***GanCtrl***: A Generative AI Approach to Derive Study-Aligned Synthetic Controls for Reducing Concurrent Control Animal Use

*GanCtrl* (GAN-based synthetic control) is a conditional VAE–GAN framework that translates treatment-derived clinical-pathology profiles into their time-matched control equivalents—synthetic controls using the Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System) rat *in vivo* repeat-dose data across 38 clinical pathology measurements. 

GanCtrl is trained on high-dose treatment samples and couples a context-conditioned encoder with an attention-aware, biologically informed decoder acting as the generator, together with a discriminator to generate physiologically coherent virtual controls.

---

## Table of Contents

- [Introduction](#introduction)
- [Code Scripts](#code-scripts)
  - [GanCtrl Model Development, Training & Predictions](#ganctrl-model-development-training--predictions)
  - [GanCtrl versus Real control](#ganctrl-versus-real-control)
  - [Toxicity Assessment](#toxicity-assessment)
  - [Literature Validation](#literature-validation)
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

### **GanCtrl versus Real control**

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

We also evaluated whether replacing real controls with synthetic controls preserves biological inferences for literature-anchored hepatotoxicity and nephrotoxicity responses. Treatment-associated measurement elevations were identified using one-sided Welch t-tests with Benjamini-Hochberg FDR correction. Measurement-level calls were combined to assess coordinated responses for the ALT–AST, ALP–TBIL, ALP–GGT/GTP, and BUN–CRE pairs and the ALT–AST–LDH triad. Agreement between biological conclusions obtained using real versus synthetic controls was evaluated using three metrics:

- Recall - Fraction of treatments identified as co-elevated using real controls that were also identified as co-elevated using synthetic controls.
- Specificity - Fraction of treatments not identified as co-elevated using real controls that remained not co-elevated using synthetic controls.
- Balanced Accuracy - Mean of recall and specificity.

**Files**:

<script_name>.py - Performs statistical testing and evaluates agreement of co-elevation calls between real- and synthetic-control analyses.    
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

## Data Files

This repository expects preprocessed CSVs derived from Open TG-GATEs. The raw TG-GATEs data are *not* distributed here; you must obtain them separately and reproduce the preprocessing if needed from the official Open TG-GATEs repository:  
[Open TG-GATEs download page](https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html).

### Preprocessed inputs (provided via Zenodo)

Because the preprocessed input files are **large**, we distribute them separately on Zenodo:  
[Zenodo deposit (inputs + molecular descriptors)](https://doi.org/10.5281/zenodo.17883691)

The Zenodo deposit includes the training and test splits of treatment/control CSVs used for training and evaluation, as well as the molecular descriptor features required by the model.

### Generated predictions (synthetic controls)

Below are the **generated prediction files** (decoded synthetic control outputs) provided for direct downstream analysis and evaluation:

| File | Description |
|---|---|
| [`generated_liver_train.csv`](data/generated_liver_train.csv) | Synthetic control predictions from the **liver**-specific GanCtrl checkpoint for the training set. |
| [`generated_liver_test.csv`](data/generated_liver_test.csv) | Synthetic control predictions from the **liver**-specific GanCtrl checkpoint for the test set. |
| [`generated_kidney_train.csv`](data/generated_kidney_train.csv) | Synthetic control predictions from the **kidney**-specific GanCtrl checkpoint for the training set. |
| [`generated_kidney_test.csv`](data/generated_kidney_test.csv) | Synthetic control predictions from the **kidney**-specific GanCtrl checkpoint for the test set. |
| [`generated_predictions_merged_train.csv`](data/generated_predictions_merged_train.csv) | Merged liver + kidney synthetic control predictions for the training set (38 clinical pathology measures), formatted for statistical agreement analyses (cosine similarity, RMSE, etc.). |
| [`generated_predictions_merged_test.csv`](data/generated_predictions_merged_test.csv) | Merged liver + kidney synthetic control predictions for the test set (38 clinical pathology measures), formatted for statistical agreement analyses (cosine similarity, RMSE, etc.). |


Key column expectations (used in the scripts):

- **Metadata / keys**:
  - `COMPOUND_NAME`, `DOSE_LEVEL`, `SACRIFICE_PERIOD`, `INDIVIDUAL_ID`
- **Outcome features** (38 clinical pathology profiles)
  - Clinical chemistry & hematology markers (ALT, AST, ALP, LDH, TBIL, BUN, CRE, WBC, etc.).
  - Out of 38, seven are liver-associated (ALP, ALT, AST, GTP, LDH, TBIL, DBIL) and seven are kidney-associated (BUN, CRE, Ca, Cl, Na, IP, K), reflecting established relevance to hepatotoxicity and nephrotoxicity.

Model outputs:

- `g_model1_*.h5` — Generator checkpoints  
- `d_model1_*.h5` — Discriminator checkpoints  
- `composite_model1_*.h5` — Composite checkpoints  
- `predictions_encoded/` — Encoded-space predictions (if enabled)  
- `predictions_decoded/` — Fully decoded predictions (means + samples)  

Adjust paths and prefixes as needed for your environment.

> *If sampling is enabled during inference, additional files may be produced such as `generated_samples_s{K}_*.csv` under a `samples/` subfolder.*

---

## Installation


### Prerequisites

Before using this repository, ensure you have the following installed:
- Python (version 3.11.7)
- Tensorflow-GPU (version 2.4.1)
- R (version 4.4.1)
- Bioconductor (version 3.19)
- Other packages specified in the code scripts

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
