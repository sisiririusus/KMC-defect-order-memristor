# Notebook map

| Public notebook | Original source | Purpose |
|---|---|---|
| `notebooks/00_kmc_baseline_simulator_v5_1.ipynb` | `KMC_2D_5.1.ipynb` | Baseline 2D KMC memristor / filament simulator. Includes the core lattice model, event engine, morphology metrics, and diagnostic plots. |
| `notebooks/01_phase1_order_parameter_scan.ipynb` | `stage1/KMC_2D_phase1_mscan_executed.ipynb` | Scans the initial defect-order parameter m0 at fixed field and temperature, then exports barrier, switching, and morphology summaries. |
| `notebooks/02_phase2_field_temperature_scan.ipynb` | `stage1/KMC_2D_phase2_mE_mT_scan_v2_firstperc_median.ipynb` | Runs coupled m-E and m-T scans to quantify formation probability, switching delay, and filament topology across the operating window. |
| `notebooks/03_dynamic_feedback_validation.ipynb` | `stage1/KMC_2D_dynamic_m_auto_update_validation_6pt.ipynb` | Compares static and dynamic order-parameter updates and evaluates how m(t), sigma_E(t), and topology respond near and away from threshold. |
| `notebooks/04_sensitivity_analysis.ipynb` | `sensitivity/KMC_sensitivity_static_dynamic_stage1_paperfig_rev3.ipynb` | Runs one-at-a-time sensitivity analysis around the baseline parameter set for both the static and dynamic KMC workflows. |
| `notebooks/05_drive_normalized_scaling.ipynb` | `delta_data_analy/KMC_drive_normalized_scaling_v2_seaborn_topomain.ipynb` | Fits logistic formation curves, extracts E50 and wE, and evaluates the reduced descriptors Delta, E/E50, E-hat, and Psi. |
| `notebooks/06_supplementary_figures_s1_s8.ipynb` | `supplementary/KMC_all_supplementary_S1_S8_onefile.ipynb` | CSV-first notebook for rebuilding selected supplementary figures directly from exported tables. |
| `notebooks/07_review_response_12_tasks.ipynb` | `12_task/KMC_review_response_12_tasks_runner.ipynb` | End-to-end notebook for rerunning the 12 reviewer-response tasks and exporting curated figures and summary tables. |
