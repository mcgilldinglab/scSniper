# 🎯 scSniper: Single-cell Deep Neural Network-basd Identification of Prominent Biomarkers
*A novel method for biomarker discovery that leverages deep neural network sensitivity analysis and an attention mechanism to select pivotal multi-omics biomarkers without relying on differential expression.*

This repository contains the official implementation of the paper:
> __scSniper: Single-cell Deep Neural Network-basd Identification of Prominent Biomarkers__
> [Mingyang Li],[Yanshuo Chen],[Jun Ding]
> 
## Overview
<p align="center">
  <img src=images/scSniperLogo.png alt="GitHub Logo" width="500" height="300">
</p>
scSniper introduces a novel approach to biomarker discovery, leveraging deep neural network sensitivity analysis to pinpoint key gene biomarkers beyond traditional differential expression methods. It stands out by utilizing a mimetic attention mechanism, which allows for the integration of multi-omic data, highlighting critical biomarkers across genomics, proteomics, and metabolomics. This mechanism prioritizes important multi-omic features, enabling a comprehensive analysis that other single-omic focused tools may miss.The tool innovatively merges a disease classifier and an autoencoder, producing joint cell embeddings that represent disease-specific multi-omic profiles, improving single-cell resolution clustering and biomarker identification accuracy. scSniper's method represents a significant leap in biomarker discovery, offering a more integrative and precise approach to understanding complex diseases.

## Setup

## Training

## Evaluation

## To Do
- [ ] Add distributed training (more than one GPU)
- [ ] Add option to use different optimizers

## Acknowledgement
This work was funded in part by grants awarded to [JD]. We gratefully acknowledge the support from the Canadian Institutes of Health Research (CIHR) under Grant Nos. PJT-180505; the Funds de recherche du Québec - Santé (FRQS) under Grant Nos. 295298 and 295299; the Natural Sciences and Engineering Research Council of Canada (NSERC) under Grant No. RGPIN2022-04399; and the Meakins-Christie Chair in Respiratory Research.
## Citation
