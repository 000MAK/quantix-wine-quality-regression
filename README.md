# Wine Quality Regression

Regression baselines with chronological-free holdout evaluation, MAE, RMSE and R².

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-offline%20unit%20tests-2ea44f?style=flat)
![Data](https://img.shields.io/badge/data-fetched%20at%20runtime-6f42c1?style=flat)

## Why this project

This repository is part of Mehmet Ali Kaya's applied statistics, actuarial-risk research and machine-learning portfolio. It emphasizes explicit assumptions, repeatable scripts, deterministic seeds, data provenance and honest evaluation rather than inflated performance claims.

**Focus:** Applied statistics · regression.

## Data and provenance

UCI Wine Quality red and white CSV files fetched from UCI at run time; target is the ordinal sensory quality score.

- Source: [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Attribution / terms: Cortez, P. et al. (2009), Wine Quality, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C56S3T. Dataset terms are defined by UCI (CC BY 4.0).
- Data are not mirrored in this repository. Running the example requires network access and may download data or model artifacts into ignored local directories.

## Setup

Requires Python 3.11+ and an internet connection for the first full run. CPU-only execution is supported, though neural-network experiments may take a while.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

## Outputs

The script writes metrics and reports under `outputs/`; model projects also save the learned state or estimator there. Local data, outputs and model artifacts are ignored by Git by default. Most arguments can be inspected with:

```bash
python main.py --help
```

## Quality and scope

The CI workflow runs Python compilation and offline unit tests. Unit tests deliberately do not download datasets or train models. Full end-to-end runs should be performed separately in an internet-enabled environment. Performance results are not included as fixed claims because they depend on the data snapshot, dependencies, hardware and run settings.

**Code license:** MIT; see [`LICENSE`](LICENSE). The code license does **not** change third-party dataset, image, audio or model terms. Review the source page before reuse, redistribution or commercial use. No customer or confidential data are used.
