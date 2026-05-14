# Improved Model Plots

This note explains how to regenerate the scientifically stricter comparison plots and report from `summary.csv`.

## Generate figures only

```bash
python3 scripts/generate_model_comparison_charts.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir artifacts/data_experiments/charts \
  --include-fallback false
```

## Generate markdown report plus figures

```bash
python3 scripts/generate_model_report.py \
  --input artifacts/data_experiments/results/summary.csv \
  --output-dir reports \
  --include-fallback false \
  --save-png true
```

## Main outputs

- `reports/model_comparison_report.md`
- `reports/figures/*.png`

## Scientific rules now enforced

- Fallback responses are excluded from scientific plots by default.
- Circular statistics are used for azimuth disagreement.
- Extreme displacement and temperature scales are split into normal and diagnostic plots.
- FNO 3D adaptation issues are surfaced through warnings and report text instead of being silently hidden.
