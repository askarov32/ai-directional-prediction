# TRAINING

```bash
python scripts/prepare_dataset.py --config configs/base.yaml --dataset_id sandstone_comsol_real
python scripts/train_base_model.py --config configs/base.yaml --dataset_ids sandstone_comsol_real
```

Модель получает на вход `node_static + scenario_features + state_t`, edge features `dx, dy, dz, distance`, target по умолчанию `delta = state_t+1 - state_t`.
