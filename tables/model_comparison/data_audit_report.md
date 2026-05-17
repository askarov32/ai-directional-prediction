# Data Audit Report

- Summary file used: `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d_4materials_balanced/summary_2d.csv`
- Material source used: `/Users/askarovi/Documents/New project/backend/data/media/catalog.json`

## 1. Input files found
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials_balanced.jsonl`
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/inputs/model_comparison_inputs_2d_4materials.jsonl`
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/inputs/model_comparison_inputs_2d.jsonl`

## 2. Summary/result files found
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d_4materials_balanced/summary_2d.csv`
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d_4materials/summary_2d.csv`
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d/summary_2d.csv`
- `/Users/askarovi/Documents/New project/artifacts/data_experiments/results_2d_4materials_balanced/summary.csv`

## 3. Material parameter files found
- `/Users/askarovi/Documents/New project/backend/data/media/catalog.json`

## 4. Prediction summary columns
- Column count: `35`
- Columns: `case_id, model, status, service_mode, fallback_used, requested_domain_type, effective_domain_type, domain_adaptation, material, medium_id, temperature_c, pressure_mpa, time_ms, frequency_hz, source_x, source_y, source_z, probe_x, probe_y, probe_z, direction_x, direction_y, direction_z, azimuth_deg, elevation_deg, magnitude, travel_time_ms_pred, max_displacement, max_temperature_perturbation, wave_type, model_version, error_code, error_message, http_status, input_azimuth_deg`

## 5. Material table columns
- Column count: `18`
- Columns: `material, medium_id, name, category, rho_kg_m3, Vp_m_s, Vs_m_s, E_Pa, K_Pa, mu_Pa, k_W_mK, Cp_J_kgK, alpha_1_K, porosity_percent, E_GPa, K_GPa, mu_GPa, alpha_1e6_K`

## 6. Models present
- Models: `pinn, mgn, fno, transformer`

## 7. Materials present
- Materials: `basalt, sandstone, granite, limestone`

## 8. 2D consistency verifiability
- Verifiable columns present: `requested_domain_type, effective_domain_type, source_z, probe_z, direction_z, elevation_deg`
- 2D consistency can be partially verified: `yes`

## 9. Speed / latency availability
- Speed columns found: `none`
- Speed analysis available: `no`

## 10. Ground truth / reference availability
- Reference columns found in summary: `none`
- Reference-like files found: `none`
- Accuracy metrics available from current summary: `no`
