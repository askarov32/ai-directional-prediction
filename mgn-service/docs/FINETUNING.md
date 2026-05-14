# FINETUNING

```bash
python scripts/finetune_model.py --config configs/finetune.yaml --dataset_id basalt_comsol_real --checkpoint outputs/checkpoints/best_model.pt --mode full
```

Режимы: `full`, `decoder_only`, `processor_decoder`.

Перед fine-tune нужно подготовить новый датасет через `prepare_dataset.py`. `field_names`, `node_in_dim`, `edge_in_dim`, `out_dim` должны быть совместимы с checkpoint.
