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

### **[Preprocessing](./preprocessing)**

This folder contains the code and scripts for data preprocessing, specifically focusing on **RMA (Robust Multi-array Average) normalization** and **annotation** of gene expression data. These preprocessing steps are critical for preparing data before training the **AIVIVE** model.

**Files**:


