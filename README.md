# ***GanCtrl***: A Generative AI Approach to Derive Study-Aligned Synthetic Controls for Reducing Concurrent Control Animal Use

<p align="center">
  <strong>Generating study-aligned synthetic control clinical-pathology profiles from treatment animals using a conditional VAE-GAN framework</strong>
</p>

<p align="center">

[![Data DOI](https://img.shields.io/badge/Data_DOI-10.5281%2Fzenodo.17883691-blue)](https://doi.org/10.5281/zenodo.17883691)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.11.7-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.4.1-orange)
![R](https://img.shields.io/badge/R-4.4.1-blue)

</p>

<p align="center">
  <a href="#quick-start"><strong>Quick Start</strong></a> •
  <a href="#repository-structure"><strong>Repository Structure</strong></a> •
  <a href="#reproducing-the-ganctrl-analysis"><strong>Reproduce Analysis</strong></a> •
  <a href="#data"><strong>Data</strong></a> •
  <a href="#citation"><strong>Citation</strong></a>
</p>

---

## Overview

***GanCtrl*** (**GAN-based synthetic control**) is a **conditional VAE-GAN** framework that translates treatment-derived clinical-pathology profiles into their **time-matched control equivalents**, generating study-aligned synthetic controls.

The framework was developed using the **Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System)** rat *in vivo* repeat-dose dataset across **38 clinical pathology measurements**.

GanCtrl is trained on high-dose treatment samples and combines:

- A **context-conditioned encoder**
- An **attention-aware, biologically informed decoder** acting as the generator
- An **adversarial discriminator**
- Study- and animal-level conditioning information
- Biologically motivated loss components designed to preserve physiologically realistic control profiles

The overall goal is to evaluate whether **GanCtrl-generated synthetic controls can reproduce real concurrent-control behavior sufficiently well to preserve downstream toxicological conclusions**.

> **Paper:** Published in *Toxicological Sciences*  
> **Paper DOI:** `PASTE_CORRECT_PAPER_DOI_HERE`  
> **GanCtrl data:** [Zenodo DOI: 10.5281/zenodo.17883691](https://doi.org/10.5281/zenodo.17883691)

---

## GanCtrl Study Design

<p align="center">
  <img src="plots/study_design.png" alt="GanCtrl study design and architecture" width="900">
</p>

<p align="center">
  <em>Overview of the GanCtrl framework for translating treatment-derived clinical-pathology profiles into study-aligned synthetic controls.</em>
</p>

---

## GanCtrl at a Glance

| Component | Description |
|---|---|
| **Framework** | Conditional VAE-GAN |
| **Input** | High-dose treatment clinical-pathology profiles |
| **Output** | Time-matched synthetic control profiles |
| **Dataset** | Open TG-GATEs rat *in vivo* repeat-dose studies |
| **Clinical pathology measurements** | 38 |
| **Conditioning information** | Body weight, timepoint, replicate identity, and study-specific clusters |
| **Primary agreement metrics** | Cosine similarity and RMSE |
| **Biological evaluation** | Liver- and kidney-associated co-elevation patterns |
| **Toxicity evaluation** | Real-control vs synthetic-control concordance |
| **Historical-control benchmark** | VCG and VCG-LR |
| **Primary organs evaluated** | Liver and kidney |

---

## Table of Contents

- [Overview](#overview)
- [GanCtrl Study Design](#ganctrl-study-design)
- [GanCtrl at a Glance](#ganctrl-at-a-glance)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Reproducing the GanCtrl Analysis](#reproducing-the-ganctrl-analysis)
  - [Step 1: Obtain the Data](#step-1-obtain-the-data)
  - [Step 2: Train GanCtrl](#step-2-train-ganctrl)
  - [Step 3: Generate Synthetic Controls](#step-3-generate-synthetic-controls)
  - [Step 4: Evaluate Synthetic vs Real Controls](#step-4-evaluate-synthetic-vs-real-controls)
  - [Step 5: Evaluate Biological Co-elevation](#step-5-evaluate-biological-co-elevation)
  - [Step 6: Evaluate Toxicity Concordance](#step-6-evaluate-toxicity-concordance)
  - [Step 7: Compare Against VCG Approaches](#step-7-compare-against-vcg-approaches)
- [GanCtrl Model](#ganctrl-model)
- [Code Reference](#code-reference)
  - [Model Development, Training and Prediction](#model-development-training-and-prediction)
  - [Synthetic vs Real Control Evaluation](#synthetic-vs-real-control-evaluation)
  - [Biological Co-elevation Analysis](#biological-co-elevation-analysis)
  - [Toxicity Assessment](#toxicity-assessment)
  - [VCG Benchmark](#vcg-benchmark)
- [Data](#data)
- [Generated Synthetic Controls](#generated-synthetic-controls)
- [Clinical Pathology Measurements](#clinical-pathology-measurements)
- [Model Outputs](#model-outputs)
- [Installation](#installation)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)

---

# Quick Start

This section provides the overall workflow for reproducing the GanCtrl analyses.

## 1. Clone the Repository

```bash
git clone https://github.com/CHANDMX20/GanCtrl.git
cd GanCtrl
```

---

## 2. Obtain the Input Data

The preprocessed Open TG-GATEs training and test inputs used by GanCtrl are distributed through Zenodo:

**GanCtrl input data:**  
https://doi.org/10.5281/zenodo.17883691

The Zenodo deposit contains:

- Training treatment profiles
- Training control profiles
- Held-out test treatment profiles
- Held-out test control profiles
- Study/sample metadata
- Molecular descriptor features required by GanCtrl

See the [Data](#data) section for additional information.

---

## 3. Train GanCtrl

The primary model-training script is:

```bash
python training/ganctrl_training.py
```

This script implements the treatment-to-control conditional VAE-GAN training workflow.

> **Note:** Input/output paths used by the training script should be configured for your local environment before execution.

---

## 4. Generate Synthetic Controls

After training, generate synthetic-control profiles using:

```bash
python training/train_test_samples.py
```

The inference workflow generates decoded synthetic-control profiles from trained GanCtrl checkpoints.

---

## 5. Evaluate Synthetic vs Real Controls

### Cosine Similarity

```bash
python evaluation/cosine.py
```

### Root Mean Squared Error

```bash
python evaluation/rmse.py
```

These analyses quantify the agreement between GanCtrl-generated synthetic controls and corresponding real controls.

---

## 6. Run Downstream Toxicological Analyses

### Biological co-elevation analysis

```bash
python evaluation/co-elevation.py
```

### Training-set concordance threshold calibration

```bash
python evaluation/train_concordance.py
```

### Held-out test-set concordance

```bash
python evaluation/test_concordance.py
```

---

## 7. Run Historical-Control VCG Benchmarks

```bash
python vcg/vcg_baseline.py
```

```bash
python vcg/vcg-lr_baseline.py
```

Additional baseline scripts are described in the [Code Reference](#code-reference).

---

# Repository Structure

The repository is organized around model development, baseline construction, downstream evaluation, and generated data.

```text
GanCtrl/
│
├── training/
│   ├── ganctrl_training.py
│   └── train_test_samples.py
│
├── evaluation/
│   ├── cosine.py
│   ├── rmse.py
│   ├── co-elevation.py
│   ├── train_concordance.py
│   └── test_concordance.py
│
├── baseline/
│   ├── interlab_cosine.py
│   ├── intralab_cosine.py
│   ├── replicate_control_cosine.py
│   ├── interlab_rmse.py
│   ├── intralab_rmse.py
│   └── replicate_control_rmse.py
│
├── vcg/
│   ├── vcg_baseline.py
│   └── vcg-lr_baseline.py
│
├── data/
│   ├── generated_liver_train.csv
│   ├── generated_liver_test.csv
│   ├── generated_kidney_train.csv
│   ├── generated_kidney_test.csv
│   ├── generated_predictions_merged_train.csv
│   └── generated_predictions_merged_test.csv
│
├── plots/
│   └── study_design.png
│
├── README.md
└── LICENSE
```

> The structure above highlights the primary files used in the GanCtrl workflow. Additional supporting files may also be present within individual directories.

---

# Reproducing the GanCtrl Analysis

The complete analysis can be viewed as the following sequence:

```text
                    Open TG-GATEs
                          │
                          ▼
                Preprocessed Inputs
                          │
                          ▼
                    Zenodo Data
                          │
                          ▼
                   GanCtrl Training
                          │
                          ▼
             Generate Synthetic Controls
                          │
             ┌────────────┴────────────┐
             │                         │
             ▼                         ▼
      Agreement Metrics        Biological Evaluation
     Cosine Similarity/RMSE        Co-elevation
             │                         │
             └────────────┬────────────┘
                          ▼
                Toxicity Concordance
                          │
                          ▼
                 VCG / VCG-LR Benchmark
```

---

## Step 1: Obtain the Data

The raw Open TG-GATEs data are not redistributed through this repository.

Raw data can be obtained from the official Open TG-GATEs database:

https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html

The **preprocessed files used directly by GanCtrl** are available through Zenodo:

https://doi.org/10.5281/zenodo.17883691

---

## Step 2: Train GanCtrl

The primary training implementation is:

[`training/ganctrl_training.py`](./training/ganctrl_training.py)

The model learns a one-sided mapping from:

```text
High-dose treatment profile
              │
              ▼
           GanCtrl
              │
              ▼
Time-matched synthetic control profile
```

The implementation uses fixed random seeds and deterministic TensorFlow operations to support reproducibility.

---

## Step 3: Generate Synthetic Controls

Synthetic-control inference is performed using:

[`training/train_test_samples.py`](./training/train_test_samples.py)

The script uses trained GanCtrl checkpoints to generate synthetic-control profiles for downstream analysis.

Generated liver-, kidney-, and merged prediction files are included in the [`data/`](./data) directory for direct evaluation.

---

## Step 4: Evaluate Synthetic vs Real Controls

GanCtrl performance is assessed by comparing generated synthetic controls with corresponding real controls using two primary agreement metrics:

- **Cosine similarity**
- **Root mean squared error (RMSE)**

GanCtrl performance is interpreted relative to three real-control benchmarks.

### Inter-laboratory Baseline

Cross-study measurements using:

- The same vehicle
- Different laboratories

### Intra-laboratory Baseline

Within-study measurements using:

- The same vehicle
- The same laboratory

### Replicate Control Baseline

Agreement between biological replicates within a treatment, providing a benchmark for real biological performance.

### GanCtrl

Agreement between each generated synthetic profile and its corresponding real control profile.

---

## Step 5: Evaluate Biological Co-elevation

GanCtrl is also evaluated according to whether replacing real controls with synthetic controls preserves biological inference for literature-anchored **hepatotoxicity** and **nephrotoxicity** responses.

Treatment-associated measurement elevations are identified using:

- One-sided Welch t-tests
- Benjamini-Hochberg false discovery rate correction

Measurement-level calls are combined to evaluate coordinated responses for:

- **ALT–AST**
- **ALP–TBIL**
- **ALP–GGT/GTP**
- **BUN–CRE**
- **ALT–AST–LDH**

Agreement between conclusions obtained using real versus synthetic controls is evaluated using:

| Metric | Definition |
|---|---|
| **Recall** | Fraction of treatments identified as co-elevated using real controls that were also identified as co-elevated using synthetic controls |
| **Specificity** | Fraction of treatments not identified as co-elevated using real controls that remained not co-elevated using synthetic controls |
| **Balanced Accuracy** | Mean of recall and specificity |

The analysis is implemented in:

[`evaluation/co-elevation.py`](./evaluation/co-elevation.py)

---

## Step 6: Evaluate Toxicity Concordance

GanCtrl is evaluated according to whether synthetic controls can substitute real controls for **toxicological decision-making**.

Concordance analysis compares sample-level **abnormal/normal classifications** obtained using:

```text
Real concurrent controls
           versus
GanCtrl synthetic controls
```

The analysis consists of two stages.

### Training Set

The training set is used to calibrate the z-score decision threshold.

### Held-out Test Set

The calibrated threshold is fixed and applied to the independent test set.

For each compound-time group and clinical pathology measurement, treatment samples are standardized relative to the corresponding control source:

\[
z =
\frac{x_{\text{treatment}}-\mu_{\text{control}}}
{\sigma_{\text{control}}}
\]

Synthetic-control-referenced classifications are compared with real-control-referenced classifications using:

- True positives (**TP**)
- True negatives (**TN**)
- False positives (**FP**)
- False negatives (**FN**)

Concordance accuracy is calculated as:

\[
\text{Concordance Accuracy}
=
\frac{TP+TN}{TP+TN+FP+FN}
\]

The relevant scripts are:

- [`evaluation/train_concordance.py`](./evaluation/train_concordance.py)
- [`evaluation/test_concordance.py`](./evaluation/test_concordance.py)

The analysis is designed to address the practical question:

> **If the concurrent control arm is replaced with GanCtrl-generated synthetic controls, are the same toxicological conclusions reached?**

---

## Step 7: Compare Against VCG Approaches

GanCtrl is benchmarked against virtual control group (**VCG**) approaches constructed from historical control data.

Historical controls are derived from training-set concurrent controls and matched to each test compound-time group using study metadata.

Controls originating from the same compound are excluded to prevent information leakage.

Two VCG configurations are evaluated.

### VCG

Historical controls matched according to:

- Sacrifice time
- Vehicle
- Laboratory

### VCG-LR

A laboratory-relaxed VCG matched according to:

- Sacrifice time
- Vehicle

but allowing controls to originate from different laboratories.

This approximates a mixed-laboratory historical-control setting.

For every test group:

- The VCG contains the same number of animals as the corresponding real concurrent control group.
- Sampling is repeated **100 times**.
- Independent random draws are made from the eligible historical-control pool.

Relevant scripts:

- [`vcg/vcg_baseline.py`](./vcg/vcg_baseline.py)
- [`vcg/vcg-lr_baseline.py`](./vcg/vcg-lr_baseline.py)

---

# GanCtrl Model

GanCtrl is a one-sided **conditional VAE-GAN** trained to translate high-dose treatment profiles into time-matched synthetic-control profiles.

### Conditioning Variables

GanCtrl incorporates:

- **Body weight**
- **Timepoint**
- **Replicate identity**
- **Study-specific clusters**

### Biologically Motivated Training Objectives

The model incorporates:

- **Variance-aware Gaussian negative log likelihood**
- **Adversarial loss**
- **TBIL range-constraint loss**
- **Biological correlation-preservation loss**
- **Batch-level mean matching**

These components are designed to preserve organ-level biological patterns while maintaining realistic variation in the generated control profiles.

<details>
<summary><strong>Why use these additional biological constraints?</strong></summary>

<br>

The goal of GanCtrl is not simply to minimize point-wise prediction error.

Synthetic controls should also maintain biologically plausible relationships among clinical pathology measurements and preserve realistic variability.

Accordingly, the model combines reconstruction- and distribution-oriented objectives with biological constraints designed to encourage physiologically coherent synthetic profiles.

</details>

---

# Code Reference

## Model Development, Training and Prediction

| File | Purpose |
|---|---|
| [`training/ganctrl_training.py`](./training/ganctrl_training.py) | End-to-end model-development and training script for the one-sided treatment-to-control conditional VAE-GAN |
| [`training/train_test_samples.py`](./training/train_test_samples.py) | Inference and prediction script used to generate synthetic controls from trained checkpoints |

---

## Synthetic vs Real Control Evaluation

### Cosine Similarity

| Analysis | Script | Purpose |
|---|---|---|
| **Inter-laboratory baseline** | [`baseline/interlab_cosine.py`](./baseline/interlab_cosine.py) | Calculates cross-study cosine similarity using controls from different laboratories |
| **Intra-laboratory baseline** | [`baseline/intralab_cosine.py`](./baseline/intralab_cosine.py) | Calculates within-study cosine similarity using controls from the same laboratory |
| **Replicate-control baseline** | [`baseline/replicate_control_cosine.py`](./baseline/replicate_control_cosine.py) | Calculates cosine similarity among biological replicates |
| **GanCtrl** | [`evaluation/cosine.py`](./evaluation/cosine.py) | Calculates cosine similarity between synthetic and corresponding real-control profiles |

### Root Mean Squared Error

| Analysis | Script | Purpose |
|---|---|---|
| **Inter-laboratory baseline** | [`baseline/interlab_rmse.py`](./baseline/interlab_rmse.py) | Calculates cross-study RMSE |
| **Intra-laboratory baseline** | [`baseline/intralab_rmse.py`](./baseline/intralab_rmse.py) | Calculates within-study RMSE |
| **Replicate-control baseline** | [`baseline/replicate_control_rmse.py`](./baseline/replicate_control_rmse.py) | Calculates replicate-control RMSE |
| **GanCtrl** | [`evaluation/rmse.py`](./evaluation/rmse.py) | Calculates RMSE between synthetic and corresponding real-control profiles |

---

## Biological Co-elevation Analysis

| File | Purpose |
|---|---|
| [`evaluation/co-elevation.py`](./evaluation/co-elevation.py) | Performs statistical testing and evaluates agreement of co-elevation calls obtained using real and synthetic controls |

The analysis evaluates coordinated changes involving:

```text
Liver:
ALT–AST
ALP–TBIL
ALP–GGT/GTP
ALT–AST–LDH

Kidney:
BUN–CRE
```

<details>
<summary><strong>Evaluation metrics</strong></summary>

<br>

### Recall

Fraction of treatments identified as co-elevated using real controls that are also identified as co-elevated using synthetic controls.

### Specificity

Fraction of treatments not identified as co-elevated using real controls that remain not co-elevated using synthetic controls.

### Balanced Accuracy

\[
\text{Balanced Accuracy}
=
\frac{\text{Recall}+\text{Specificity}}{2}
\]

</details>

---

## Toxicity Assessment

The concordance pipeline evaluates whether synthetic-control references reproduce toxicity classifications obtained using real concurrent controls.

| File | Purpose |
|---|---|
| [`evaluation/train_concordance.py`](./evaluation/train_concordance.py) | Performs training-set concordance analysis used for threshold calibration |
| [`evaluation/test_concordance.py`](./evaluation/test_concordance.py) | Applies the fixed threshold to the held-out test set and evaluates synthetic-control toxicity concordance |

---

## VCG Benchmark

| File | Purpose |
|---|---|
| [`vcg/vcg_baseline.py`](./vcg/vcg_baseline.py) | Generates VCGs by matching historical controls on sacrifice time, vehicle, and laboratory |
| [`vcg/vcg-lr_baseline.py`](./vcg/vcg-lr_baseline.py) | Generates laboratory-relaxed VCGs by matching sacrifice time and vehicle while allowing controls from different laboratories |

---

# Data

## Raw Open TG-GATEs Data

GanCtrl was developed using rat *in vivo* repeat-dose clinical pathology data derived from **Open TG-GATEs**.

The raw Open TG-GATEs data are **not distributed through this repository**.

The original data can be obtained from:

**Open TG-GATEs download page:**  
https://dbarchive.biosciencedbc.jp/en/open-tggates/download.html

---

## Preprocessed GanCtrl Inputs

Because the preprocessed input files are large, the exact training and test input datasets used by GanCtrl are distributed through Zenodo:

### [Download GanCtrl Input Data from Zenodo](https://doi.org/10.5281/zenodo.17883691)

The Zenodo deposit contains the treatment and control CSVs used for training and evaluation together with the molecular-descriptor features required by the model.

The principal files include:

```text
repeat_train_treatment_2d.csv
repeat_train_control_2d.csv
repeat_test_treatment_2d.csv
repeat_test_control_2d.csv
```

Conceptually:

```text
                    Open TG-GATEs
                          │
                          ▼
                Preprocessed Inputs
                          │
                          ▼
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
      Training Inputs              Test Inputs
   Treatment + Control         Treatment + Control
            │                           │
            └─────────────┬─────────────┘
                          ▼
                       GanCtrl
                          │
                          ▼
               Synthetic Controls
```

---

# Generated Synthetic Controls

Generated decoded synthetic-control predictions are included in this repository to allow downstream analyses to be reproduced **without retraining GanCtrl**.

| File | Description |
|---|---|
| [`generated_liver_train.csv`](data/generated_liver_train.csv) | Synthetic-control predictions generated by the liver-specific GanCtrl checkpoint for the training set |
| [`generated_liver_test.csv`](data/generated_liver_test.csv) | Synthetic-control predictions generated by the liver-specific GanCtrl checkpoint for the held-out test set |
| [`generated_kidney_train.csv`](data/generated_kidney_train.csv) | Synthetic-control predictions generated by the kidney-specific GanCtrl checkpoint for the training set |
| [`generated_kidney_test.csv`](data/generated_kidney_test.csv) | Synthetic-control predictions generated by the kidney-specific GanCtrl checkpoint for the held-out test set |
| [`generated_predictions_merged_train.csv`](data/generated_predictions_merged_train.csv) | Merged liver + kidney synthetic controls for the training set containing the 38 clinical pathology measurements |
| [`generated_predictions_merged_test.csv`](data/generated_predictions_merged_test.csv) | Merged liver + kidney synthetic controls for the test set containing the 38 clinical pathology measurements |

The merged prediction files are formatted for downstream statistical agreement analyses including:

- Cosine similarity
- RMSE
- Biological co-elevation
- Toxicity concordance

---

# Clinical Pathology Measurements

Scripts rely on the following key metadata fields:

### Metadata / Identifiers

```text
COMPOUND_NAME
DOSE_LEVEL
SACRIFICE_PERIOD
INDIVIDUAL_ID
```

### Outcome Features

GanCtrl evaluates **38 clinical pathology measurements**, spanning clinical chemistry and hematology endpoints.

Examples include:

```text
ALT
AST
ALP
LDH
TBIL
BUN
CRE
WBC
...
```

### Liver-associated Measurements

Seven measurements are treated as liver-associated:

| Abbreviation | Measurement |
|---|---|
| ALP | Alkaline phosphatase |
| ALT | Alanine aminotransferase |
| AST | Aspartate aminotransferase |
| GTP/GGT | Gamma-glutamyl transferase |
| LDH | Lactate dehydrogenase |
| TBIL | Total bilirubin |
| DBIL | Direct bilirubin |

### Kidney-associated Measurements

Seven measurements are treated as kidney-associated:

| Abbreviation | Measurement |
|---|---|
| BUN | Blood urea nitrogen |
| CRE | Creatinine |
| Ca | Calcium |
| Cl | Chloride |
| Na | Sodium |
| IP | Inorganic phosphorus |
| K | Potassium |

These measurements reflect established relevance to hepatotoxicity and nephrotoxicity and are used in the downstream biological analyses.

---

# Model Outputs

Depending on the training/inference configuration, GanCtrl can produce:

```text
g_model1_*.h5
d_model1_*.h5
composite_model1_*.h5
predictions_encoded/
predictions_decoded/
```

where:

| Output | Description |
|---|---|
| `g_model1_*.h5` | Generator checkpoints |
| `d_model1_*.h5` | Discriminator checkpoints |
| `composite_model1_*.h5` | Composite-model checkpoints |
| `predictions_encoded/` | Encoded-space predictions, if enabled |
| `predictions_decoded/` | Fully decoded prediction outputs |

If stochastic sampling is enabled during inference, additional files may be generated, for example:

```text
samples/generated_samples_s{K}_*.csv
```

Paths and output prefixes may need to be adjusted for the local environment.

---

# Installation

## Prerequisites

The GanCtrl analysis was developed using:

| Software | Version |
|---|---:|
| **Python** | 3.11.7 |
| **TensorFlow-GPU** | 2.4.1 |
| **R** | 4.4.1 |
| **Bioconductor** | 3.19 |

Additional Python and R packages required by individual analyses are imported within the corresponding scripts.

---

## Clone the Repository

```bash
git clone https://github.com/CHANDMX20/GanCtrl.git
cd GanCtrl
```

---

## Data Setup

Download the preprocessed GanCtrl inputs from:

https://doi.org/10.5281/zenodo.17883691

Configure the input paths referenced by the relevant training or evaluation scripts before execution.

---

# Reproducibility

The GanCtrl implementation is designed to support reproducibility through:

- Fixed random seeds
- Deterministic TensorFlow operations
- Explicit training and held-out test splits
- Independent synthetic-control prediction files
- Publicly available preprocessed inputs through Zenodo
- Generated synthetic-control outputs included for direct downstream evaluation
- Separate scripts for model training, prediction, agreement evaluation, biological analysis, toxicity concordance, and historical-control benchmarking

Two levels of reproducibility are therefore possible.

### Full Model Reproduction

```text
Zenodo inputs
      ↓
Train GanCtrl
      ↓
Generate synthetic controls
      ↓
Run downstream analyses
```

### Downstream Analysis Reproduction

Users who do not wish to retrain the model can begin directly with the generated prediction files:

```text
data/generated_predictions_merged_*.csv
      ↓
Cosine / RMSE
      ↓
Biological co-elevation
      ↓
Toxicity concordance
      ↓
VCG comparison
```

---

# Citation

If you use **GanCtrl** in your research, please cite the associated manuscript and dataset.

## GanCtrl Paper

> **TODO:** Replace the citation below with the final verified GanCtrl paper citation and DOI.

```text
Chandra M, et al.
GanCtrl: A Generative AI Approach to Derive Study-Aligned
Synthetic Controls for Reducing Concurrent Control Animal Use.
Toxicological Sciences.
DOI: PASTE_CORRECT_PAPER_DOI_HERE
```

## GanCtrl Dataset

**Chandra, M.**  
*GanCtrl: Synthetic Control Predictions for Liver and Kidney Clinical-Pathology Profiles (Open TG-GATEs).*  
Zenodo.  
https://doi.org/10.5281/zenodo.17883691

---

# License

This project is licensed under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details.

---

<p align="center">
  <strong>GanCtrl</strong><br>
  Generative AI for study-aligned synthetic controls in preclinical toxicology
</p>
