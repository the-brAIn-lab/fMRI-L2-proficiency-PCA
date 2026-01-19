# fMRI-L2-proficiency-PCA
Group-level PCA of bilingual fMRI network activity to explain individual differences in L2 proficiency.

## Goal
The goal of this analysis is to determine whether **shared brain network representations**, learned across all participants, can explain **individual differences in second-language (L2) proficiency**.

Rather than focusing on subject-specific decompositions, we use a **group-level dimensionality reduction** approach to learn a **common representational space**, and then examine how individuals differ in their engagement of these shared components across language conditions.

This approach is motivated by work showing that high-level cognition is supported by **information-rich but compressible** brain activity patterns, and that group-level PCA can reveal low-dimensional representations that generalize across individuals and tasks (Owen & Manning, 2024).

## What this repo contains
- Code to build group-level data matrices across participants and conditions
- Group PCA fitting and component inspection
- Subject-wise projection into the shared PCA space
- Statistical modeling of PCA-derived features vs. L2 proficiency measures
- Figure/table generation for reporting

## Data
This repository does **not** store raw neuroimaging data.
Place paths and dataset pointers in a local config file (see `src/config.py`) or environment variables.

See `data/README.md` for expected inputs and how to point the pipeline to your dataset.

## Quick start
### 1) Create environment
Using conda:
```bash
conda env create -f environment.yml
conda activate l2pca
