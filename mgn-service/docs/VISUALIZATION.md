# VISUALIZATION

```bash
python scripts/run_prediction.py --config configs/inference.yaml --dataset_id basalt_comsol_real --checkpoint outputs/checkpoints_finetuned/best_model.pt
python scripts/visualize_results.py --dataset_id basalt_comsol_real --prediction outputs/predictions/prediction.pt --slice_axis z
```

Главные файлы:

- `outputs/wave_arrows/wave_arrows_report.html`
- `outputs/wave_arrows/animations/quiver_temperature_change_velocity.gif`
- `outputs/wave_arrows/figures/wave_front_radius_temperature_change.png`
