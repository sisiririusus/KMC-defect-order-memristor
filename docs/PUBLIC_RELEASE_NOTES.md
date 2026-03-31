# Public-release notes

This repository snapshot was prepared from the working KMC project tree and then simplified for GitHub sharing.

## What was changed
- Notebook outputs were stripped to keep the repository lightweight.
- The most important notebooks were renamed with short English titles and grouped under `notebooks/`.
- Repository-relative paths were added to the sensitivity, scaling, supplementary, and review-response notebooks.
- A curated subset of result tables and publication figures was copied into `results/`.
- Temporary caches (`__pycache__`, `.nbi`, `.nbc`) and very large raw intermediate tables were not included.

## What was intentionally left out
- Large raw CSV files used only for internal reruns.
- Duplicated legacy notebooks from `old/` and `new/`.
- Local machine-specific output paths.

## Suggested GitHub description
Kinetic Monte Carlo workflow for defect-order-controlled filament stochasticity in memristive switching, including phase scans, dynamic-feedback validation, sensitivity analysis, and drive-normalized topology scaling.
