# FNO Diagnostic Report

- Rows: `40`
- Fallback count: `0`
- Non-finite output count: `0`
- Scale outlier count: `80`
- Mean max displacement: `66372.845703125`
- Mean max temperature perturbation: `119227.3119140625`

Interpretation:
FNO output values are treated as scale-unstable prototype predictions when they exceed the range of other models by several orders of magnitude.

Warnings:
- FNO produces scale outliers relative to the other models; treat these results as scale-unstable prototype outputs rather than validated physical displacements or temperatures.
