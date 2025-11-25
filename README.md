# ***GanCtrl***: An AI-Based Pilot Study for Modeling Synthetic Controls from Treatment Data in Preclinical Toxicology

**GanCtrl** (GAN-based synthetic control) is a conditional VAE–GAN model that translates treatment-derived clinical-pathology profiles into their time-matched control equivalents—synthetic controls using the Open TG-GATEs (Toxicogenomics Project-Genomics Assisted Toxicity Evaluation System) rat ***in vivo*** repeat-dose data across 38 clinical pathology measurements. GanCtrl is trained on high-dose treatment samples and couples a context-conditioned encoder with an attention-aware, biologically informed decoder acting as the generator, together with a discriminator to generate physiologically coherent virtual controls.

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
