# ***GanCtrl***: An AI-Based Pilot Study for Modeling Virtual Controls from Treatment Data in Preclinical Toxicology

**GanCtrl** (GAN-based virtual control) is a conditional VAE–GAN model that translates treatment-derived clinical-pathology profiles into their time-matched control equivalents—virtual controls using the Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System) rat ***in vivo*** repeat-dose data. GanCtrl is trained on high-dose treatment samples and couples a context-conditioned encoder with an attention-aware, biologically informed mixture-of-experts **(Bio-MoE)** decoder acting as the generator, together with a discriminator to generate physiologically coherent virtual controls.

---

## Table of Contents

- [Introduction](#introduction)
- [Code Scripts](#code-scripts)
  - [GanCtrl Model Development, Training & Predictions](#aivive-model-development-training--predictions)
  - [DEG Analysis](#deg-analysis)
  - [KEGG Pathway Analysis](#kegg-pathway-analysis)
  - [AOP-Gene Expression Analysis](#aop-gene-expression-analysis)
  - [Necrosis Prediction Model](#necrosis-prediction-model)
  - [Model Evaluation](#model-evaluation)
- [Data Files](#data-files)
- [Installation](#installation)
- [License](#license)
  
---
