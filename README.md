# KMC memristor project — public release

This repository is a GitHub-ready snapshot of the kinetic Monte Carlo workflow used for the manuscript **“Defect-order-controlled filament stochasticity in memristive switching revealed by kinetic Monte Carlo simulations.”**

The public version is organized around a compact notebook workflow, curated result tables, and publication-ready figure assets.

## Repository layout

```text
notebooks/                  cleaned notebooks with English titles
results/                    curated CSV tables and figure assets
src/review_task_modules/    helper Python modules used by the review-task notebook
docs/                       notebook map, result manifest, and release notes
```

## Notebook workflow

1. `00_kmc_baseline_simulator_v5_1.ipynb`
   Baseline 2D KMC simulator and core morphology/statistics utilities.
2. `01_phase1_order_parameter_scan.ipynb`
   Initial order-parameter scan at fixed field and temperature.
3. `02_phase2_field_temperature_scan.ipynb`
   Coupled `m-E` and `m-T` scans for formation probability, delay, and topology.
4. `03_dynamic_feedback_validation.ipynb`
   Static-versus-dynamic comparison with time-dependent order feedback.
5. `04_sensitivity_analysis.ipynb`
   Static and dynamic one-at-a-time sensitivity analysis.
6. `05_drive_normalized_scaling.ipynb`
   Threshold extraction, `E50` normalization, and reduced-descriptor analysis.
7. `06_supplementary_figures_s1_s8.ipynb`
   CSV-first supplementary-figure regeneration notebook.
8. `07_review_response_12_tasks.ipynb`
   Dense rerun and reviewer-response task notebook.

A detailed original-to-public filename map is provided in `docs/NOTEBOOK_MAP.md`.

## Included result files

The `results/` directory includes a curated set of:
- phase-scan summary tables
- dynamic-feedback comparison summaries
- sensitivity-analysis tables
- drive-normalized scaling tables
- reviewer-response summary tables
- publication-ready main and supplementary figure assets

A machine-readable manifest is available at `docs/results_manifest.csv`.

## Running the notebooks

The cleaned notebooks assume a repository-local layout. In the path-aware notebooks, the repository root is detected automatically when `README.md`, `notebooks/`, and `results/` are present.

Recommended environment:
- Python 3.11
- Jupyter Notebook or JupyterLab
- packages listed in `requirements.txt`

## Notes for GitHub upload

- Notebook outputs were stripped intentionally; regenerated figures should be written into `results/` or a local scratch directory.
- The repository does **not** include all heavy raw intermediate files from the working project tree.
- Before making the repository public, add your preferred `LICENSE` and update author / citation metadata as needed.
