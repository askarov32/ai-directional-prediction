# VALIDATION

```bash
python scripts/validate_model.py --config configs/base.yaml --dataset_id basalt_comsol_real --checkpoint outputs/checkpoints_finetuned/best_model.pt --split test --slice_axis z
```

Строятся one-step и rollout метрики: RMSE, MAE, relative RMSE per field, error over time, derived physical error, rollout stability. Главный отчёт: `outputs/validation/validation_report.html`.
