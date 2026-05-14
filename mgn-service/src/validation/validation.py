"""Full validation for Conditional MeshGraphNet on real COMSOL ground truth.

Validation levels:
1) One-step validation: state_t -> state_{t+1} on the selected split.
2) Autoregressive rollout validation: start from the first sample of the split
   and compare predicted trajectory against the COMSOL trajectory.
3) Derived physical validation: displacement magnitude, velocity magnitude,
   von Mises stress, temperature change, risk zones, wave-front radius.
4) Research visual reports: error-over-time plots and COMSOL vs prediction slices.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.normalizer import FeatureNormalizer
from src.data.pipeline import load_processed_dataset
from src.training.checkpoint_manager import load_checkpoint
from src.training.train import build_model, setup_device


EPS = 1e-12


def _safe_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _field_index(field_names: List[str], candidates: Iterable[str], contains: bool = True) -> int | None:
    lower = [f.strip().lower() for f in field_names]
    cand = [c.strip().lower() for c in candidates]
    for c in cand:
        for i, f in enumerate(lower):
            if f == c:
                return i
    if contains:
        for c in cand:
            for i, f in enumerate(lower):
                if c in f:
                    return i
    return None


def _field_indices_exact(field_names: List[str], names: Iterable[str]) -> List[int]:
    lower = [f.strip().lower() for f in field_names]
    out = []
    for name in names:
        name_l = name.strip().lower()
        if name_l in lower:
            out.append(lower.index(name_l))
    return out


def _infer_group(field_name: str) -> str:
    f = field_name.lower()
    if f in {"t", "temp", "temperature"} or "temperature" in f:
        return "temperature"
    if f in {"u", "v", "w"}:
        return "displacement"
    if f in {"ut", "vt", "wt"} or "velocity" in f:
        return "velocity"
    if "mises" in f or f.startswith("s") or ".s" in f or "stress" in f:
        return "stress"
    if f.startswith("e") or ".e" in f or "strain" in f:
        return "strain"
    return "other"


def _denormalize_states(states_norm: np.ndarray, field_names: List[str], normalization: Dict) -> np.ndarray:
    dyn_norm = FeatureNormalizer.from_dict(normalization.get("dynamic", {}))
    return dyn_norm.denormalize_array(field_names, states_norm)


def _sample_true_next_norm(sample_x: torch.Tensor, sample_y: torch.Tensor, S: int, target_mode: str) -> torch.Tensor:
    current = sample_x[:, S:]
    if target_mode == "delta":
        return current + sample_y
    return sample_y


def _prediction_next_norm(model, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, S: int, target_mode: str) -> torch.Tensor:
    pred = model(x, edge_index, edge_attr)
    current = x[:, S:]
    if target_mode == "delta":
        return current + pred
    return pred


def _build_true_split_trajectory_norm(samples: List[Tuple[torch.Tensor, torch.Tensor]], S: int, target_mode: str) -> np.ndarray:
    if not samples:
        raise ValueError("Selected split has no samples.")
    states = [samples[0][0][:, S:].detach().cpu().numpy()]
    for x, y in samples:
        nxt = _sample_true_next_norm(x, y, S, target_mode)
        states.append(nxt.detach().cpu().numpy())
    return np.asarray(states, dtype=np.float32)


def _build_one_step_arrays(
    model,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    S: int,
    target_mode: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    true_next = []
    pred_next = []
    model.eval()
    with torch.no_grad():
        for x_cpu, y_cpu in samples:
            x = x_cpu.to(device)
            y = y_cpu.to(device)
            nxt_pred = _prediction_next_norm(model, x, edge_index, edge_attr, S, target_mode)
            nxt_true = _sample_true_next_norm(x, y, S, target_mode)
            pred_next.append(nxt_pred.detach().cpu().numpy())
            true_next.append(nxt_true.detach().cpu().numpy())
    return np.asarray(true_next, dtype=np.float32), np.asarray(pred_next, dtype=np.float32)


def _rollout_arrays(
    model,
    samples: List[Tuple[torch.Tensor, torch.Tensor]],
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    S: int,
    target_mode: str,
    device: torch.device,
    max_steps: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Selected split has no samples.")
    steps = len(samples) if max_steps is None else min(len(samples), int(max_steps))
    start_x = samples[0][0].clone().to(device)
    x = start_x
    pred_states = [x[:, S:].detach().cpu().numpy()]

    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            nxt = _prediction_next_norm(model, x, edge_index, edge_attr, S, target_mode)
            pred_states.append(nxt.detach().cpu().numpy())
            x = torch.cat([x[:, :S], nxt], dim=1)

    true_states = _build_true_split_trajectory_norm(samples[:steps], S, target_mode)
    return true_states.astype(np.float32), np.asarray(pred_states, dtype=np.float32)


def compute_derived_fields(trajectory: np.ndarray, field_names: List[str], coords: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Return derived physical fields [T,N]. Works with COMSOL names such as solid.sx and solid.mises."""
    fn = [f.strip().lower() for f in field_names]
    derived: Dict[str, np.ndarray] = {}

    # temperature and temperature change
    tidx = _field_index(field_names, ["t", "temp", "temperature"], contains=True)
    if tidx is not None:
        temp = trajectory[:, :, tidx]
        derived["temperature"] = temp
        derived["temperature_change"] = temp - temp[:1]
        derived["temperature_time_gradient"] = np.diff(temp, axis=0, prepend=temp[:1])

    # displacement magnitude
    disp_idx = _field_indices_exact(field_names, ["u", "v", "w"])
    if len(disp_idx) >= 2:
        derived["displacement_magnitude"] = np.linalg.norm(trajectory[:, :, disp_idx], axis=2)

    # velocity magnitude
    vel_idx = _field_indices_exact(field_names, ["ut", "vt", "wt"])
    if len(vel_idx) >= 2:
        derived["velocity_magnitude"] = np.linalg.norm(trajectory[:, :, vel_idx], axis=2)

    # von Mises: prefer direct COMSOL solid.mises if exported
    mises_idx = _field_index(field_names, ["mises", "solid.mises", "von_mises", "von_mises_stress"], contains=True)
    if mises_idx is not None:
        vm = np.abs(trajectory[:, :, mises_idx])
        derived["von_mises_stress"] = vm
    else:
        # Try stress tensor components.
        candidates = [
            ["s11", "s22", "s33", "s12", "s13", "s23"],
            ["sxx", "syy", "szz", "sxy", "sxz", "syz"],
            ["solid.sx", "solid.sy", "solid.sz", "solid.sxy", "solid.sxz", "solid.syz"],
            ["sx", "sy", "sz", "sxy", "sxz", "syz"],
        ]
        for names in candidates:
            idx = _field_indices_exact(field_names, names)
            if len(idx) == 6:
                s = trajectory[:, :, idx]
                s11, s22, s33, s12, s13, s23 = [s[:, :, i] for i in range(6)]
                vm = np.sqrt(
                    np.maximum(
                        0.0,
                        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2 + 6.0 * (s12 ** 2 + s13 ** 2 + s23 ** 2)),
                    )
                )
                derived["von_mises_stress"] = vm
                break

    if "von_mises_stress" in derived:
        vm = derived["von_mises_stress"]
        vmax = np.maximum(np.nanmax(vm, axis=1, keepdims=True), EPS)
        derived["risk_flag_80pct"] = (vm / vmax >= 0.8).astype(np.float32)
        derived["risk_flag_90pct"] = (vm / vmax >= 0.9).astype(np.float32)

    # strain magnitude if components exist.
    strain_candidates = [
        ["exx", "eyy", "ezz", "exy", "exz", "eyz"],
        ["solid.ex", "solid.ey", "solid.ez", "solid.exy", "solid.exz", "solid.eyz"],
        ["ex", "ey", "ez", "exy", "exz", "eyz"],
    ]
    for names in strain_candidates:
        idx = _field_indices_exact(field_names, names)
        if len(idx) >= 3:
            derived["strain_magnitude"] = np.linalg.norm(trajectory[:, :, idx], axis=2)
            break

    # Source-relative radius of the dominant response, useful for wave-front tracking.
    if coords is not None and coords.size:
        pass
    return derived


def _metric_dict(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - true
    rmse = float(np.sqrt(np.nanmean(err ** 2)))
    mae = float(np.nanmean(np.abs(err)))
    max_abs = float(np.nanmax(np.abs(err)))
    denom = float(np.sqrt(np.nanmean(true ** 2)))
    rel = float(rmse / max(denom, EPS))
    return {"rmse": rmse, "mae": mae, "max_abs": max_abs, "relative_rmse": rel}


def _field_metrics_table(
    field_names: List[str],
    field_units: Dict[str, str],
    one_true: np.ndarray,
    one_pred: np.ndarray,
    roll_true: np.ndarray,
    roll_pred: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(field_names):
        one = _metric_dict(one_true[:, :, i], one_pred[:, :, i])
        # skip identical initial state for the mean rollout score where possible
        rt = roll_true[1:, :, i] if roll_true.shape[0] > 1 else roll_true[:, :, i]
        rp = roll_pred[1:, :, i] if roll_pred.shape[0] > 1 else roll_pred[:, :, i]
        rmean = _metric_dict(rt, rp)
        rfinal = _metric_dict(roll_true[-1, :, i], roll_pred[-1, :, i])
        rows.append(
            {
                "field": f,
                "group": _infer_group(f),
                "unit": field_units.get(f, ""),
                "one_step_rmse": one["rmse"],
                "one_step_mae": one["mae"],
                "one_step_relative_rmse": one["relative_rmse"],
                "rollout_mean_rmse": rmean["rmse"],
                "rollout_mean_mae": rmean["mae"],
                "rollout_relative_rmse": rmean["relative_rmse"],
                "rollout_final_rmse": rfinal["rmse"],
                "rollout_final_mae": rfinal["mae"],
                "rollout_final_relative_rmse": rfinal["relative_rmse"],
                "rollout_final_max_abs": rfinal["max_abs"],
            }
        )
    return pd.DataFrame(rows)


def _error_over_time_table(field_names: List[str], times: List[float], roll_true: np.ndarray, roll_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    T = roll_true.shape[0]
    for t in range(T):
        time_val = times[t] if t < len(times) else t
        for i, f in enumerate(field_names):
            m = _metric_dict(roll_true[t, :, i], roll_pred[t, :, i])
            rows.append({"step": t, "time": time_val, "field": f, **m})
    return pd.DataFrame(rows)


def _derived_error_table(times: List[float], true_d: Dict[str, np.ndarray], pred_d: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    common = sorted(set(true_d) & set(pred_d))
    for key in common:
        a = true_d[key]
        b = pred_d[key]
        if a.ndim != 2 or b.ndim != 2:
            continue
        T = min(a.shape[0], b.shape[0])
        for t in range(T):
            time_val = times[t] if t < len(times) else t
            m = _metric_dict(a[t], b[t])
            rows.append({"step": t, "time": time_val, "quantity": key, **m})
    return pd.DataFrame(rows)


def _physical_summary(times: List[float], true_d: Dict[str, np.ndarray], pred_d: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key in sorted(set(true_d) & set(pred_d)):
        a = true_d[key]
        b = pred_d[key]
        if a.ndim != 2 or b.ndim != 2:
            continue
        m = _metric_dict(a[1:] if a.shape[0] > 1 else a, b[1:] if b.shape[0] > 1 else b)
        rows.append(
            {
                "quantity": key,
                "true_initial_min": float(np.nanmin(a[0])),
                "true_initial_max": float(np.nanmax(a[0])),
                "pred_initial_min": float(np.nanmin(b[0])),
                "pred_initial_max": float(np.nanmax(b[0])),
                "true_final_min": float(np.nanmin(a[-1])),
                "true_final_max": float(np.nanmax(a[-1])),
                "pred_final_min": float(np.nanmin(b[-1])),
                "pred_final_max": float(np.nanmax(b[-1])),
                "mean_rmse": m["rmse"],
                "mean_mae": m["mae"],
                "relative_rmse": m["relative_rmse"],
            }
        )
    return pd.DataFrame(rows)


def _group_metrics(field_metrics: pd.DataFrame) -> pd.DataFrame:
    if field_metrics.empty:
        return field_metrics
    cols = [
        "one_step_rmse",
        "one_step_mae",
        "one_step_relative_rmse",
        "rollout_mean_rmse",
        "rollout_mean_mae",
        "rollout_relative_rmse",
        "rollout_final_rmse",
        "rollout_final_relative_rmse",
    ]
    return field_metrics.groupby("group", as_index=False)[cols].mean(numeric_only=True)


def _scenario_center(metadata: Dict, coords: np.ndarray) -> np.ndarray:
    src = metadata.get("scenario", {}).get("source", {})
    center = src.get("center", None)
    if center is None:
        return np.nanmean(coords[:, :3], axis=0)
    arr = np.asarray(center, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size))
    return arr[:3]


def _wave_front_radius(values: np.ndarray, coords: np.ndarray, center: np.ndarray, percentile: float = 95.0, rel_threshold: float = 0.2) -> np.ndarray:
    """Estimate front radius from positive absolute response values [T,N]."""
    xyz = coords[:, :3]
    if xyz.shape[1] < 3:
        xyz = np.column_stack([xyz, np.zeros(len(xyz))])
    radius = np.linalg.norm(xyz - center.reshape(1, 3), axis=1)
    out = []
    vals = np.abs(values)
    for t in range(vals.shape[0]):
        v = vals[t]
        vmax = float(np.nanmax(v))
        if not np.isfinite(vmax) or vmax <= EPS:
            out.append(0.0)
            continue
        mask = v >= rel_threshold * vmax
        if not np.any(mask):
            out.append(0.0)
        else:
            out.append(float(np.nanpercentile(radius[mask], percentile)))
    return np.asarray(out, dtype=np.float64)


def _save_line_plot(df: pd.DataFrame, x: str, y: str, group: str, title: str, save_path: Path, max_groups: int = 12):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        groups = list(df[group].dropna().unique())[:max_groups]
        for g in groups:
            sub = df[df[group] == g]
            ax.plot(sub[x], sub[y], label=str(g))
        ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _save_bar_plot(df: pd.DataFrame, x: str, y: str, title: str, save_path: Path, top_n: int = 25):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    d = df.sort_values(y, ascending=False).head(top_n) if not df.empty else df
    fig, ax = plt.subplots(figsize=(14, 7))
    if d.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        ax.bar(np.arange(len(d)), d[y].values)
        ax.set_xticks(np.arange(len(d)))
        ax.set_xticklabels(d[x].astype(str).values, rotation=75, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _select_slice(coords: np.ndarray, axis: str = "z", value: float | None = None, tol: float | None = None) -> Tuple[np.ndarray, float, float]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    ai = axis_map.get(axis.lower(), 2)
    xyz = coords[:, :3]
    if xyz.shape[1] < 3:
        xyz = np.column_stack([xyz, np.zeros(len(xyz))])
    vals = xyz[:, ai]
    if value is None:
        value = float(np.nanmedian(vals))
    if tol is None:
        span = float(np.nanmax(vals) - np.nanmin(vals))
        tol = max(span * 0.035, 1e-9)
    mask = np.abs(vals - value) <= tol
    if int(mask.sum()) < 20:
        # Fallback: choose closest nodes to requested slice.
        order = np.argsort(np.abs(vals - value))
        keep = order[: min(max(20, len(vals) // 8), len(vals))]
        mask = np.zeros(len(vals), dtype=bool)
        mask[keep] = True
    return mask, float(value), float(tol)


def _project_coords(coords: np.ndarray, axis: str) -> Tuple[np.ndarray, np.ndarray, str, str]:
    if axis == "z":
        return coords[:, 0], coords[:, 1], "X", "Y"
    if axis == "y":
        return coords[:, 0], coords[:, 2], "X", "Z"
    return coords[:, 1], coords[:, 2], "Y", "Z"


def _save_comparison_slice(
    coords: np.ndarray,
    true_values: np.ndarray,
    pred_values: np.ndarray,
    quantity: str,
    step: int,
    axis: str,
    save_path: Path,
    slice_value: float | None = None,
    slice_tol: float | None = None,
):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mask, actual_value, actual_tol = _select_slice(coords, axis, slice_value, slice_tol)
    c = coords[mask]
    tv = true_values[step, mask]
    pv = pred_values[step, mask]
    ev = np.abs(pv - tv)
    x, y, xlabel, ylabel = _project_coords(c, axis)

    vmin = float(np.nanmin([np.nanmin(tv), np.nanmin(pv)]))
    vmax = float(np.nanmax([np.nanmax(tv), np.nanmax(pv)]))
    if abs(vmax - vmin) < EPS:
        vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    sc0 = axes[0].scatter(x, y, c=tv, s=12, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"COMSOL / true: {quantity}")
    fig.colorbar(sc0, ax=axes[0], shrink=0.82)

    sc1 = axes[1].scatter(x, y, c=pv, s=12, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"MGN prediction: {quantity}")
    fig.colorbar(sc1, ax=axes[1], shrink=0.82)

    sc2 = axes[2].scatter(x, y, c=ev, s=12)
    axes[2].set_title("absolute error")
    fig.colorbar(sc2, ax=axes[2], shrink=0.82)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2)
    fig.suptitle(f"{quantity} | step={step} | slice {axis}={actual_value:.4g} ± {actual_tol:.2g}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    plt.close(fig)


def _make_html_report(out_dir: Path, summary: Dict, field_metrics: pd.DataFrame, group_metrics: pd.DataFrame, physical_summary: pd.DataFrame):
    report = out_dir / "validation_report.html"
    figs = sorted((out_dir / "figures").glob("*.png"))
    rel = lambda p: str(p.relative_to(out_dir)).replace("\\", "/")

    def df_html(df: pd.DataFrame, n: int = 20) -> str:
        if df.empty:
            return "<p>No data.</p>"
        return df.head(n).to_html(index=False, float_format=lambda x: f"{x:.6g}")

    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>MeshGraphNet validation report</title>",
        "<style>body{font-family:Arial, sans-serif; margin:32px;} img{max-width:100%; border:1px solid #ddd; margin:12px 0;} table{border-collapse:collapse;} td,th{border:1px solid #ddd; padding:4px 7px;} code{background:#f4f4f4; padding:2px 4px;}</style>",
        "</head><body>",
        "<h1>MeshGraphNet validation report</h1>",
        f"<p><b>Dataset:</b> {summary.get('dataset_id')} &nbsp; <b>Split:</b> {summary.get('split')} &nbsp; <b>Checkpoint:</b> <code>{summary.get('checkpoint')}</code></p>",
        "<h2>Key scores</h2>",
        "<ul>",
        f"<li>One-step mean RMSE: <b>{summary.get('one_step_mean_rmse', float('nan')):.6g}</b></li>",
        f"<li>Rollout mean RMSE: <b>{summary.get('rollout_mean_rmse', float('nan')):.6g}</b></li>",
        f"<li>Rollout final RMSE: <b>{summary.get('rollout_final_rmse', float('nan')):.6g}</b></li>",
        f"<li>Temperature wave-front MAE: <b>{summary.get('temperature_front_radius_mae', float('nan')):.6g}</b></li>",
        "</ul>",
        "<h2>Group metrics</h2>",
        df_html(group_metrics, 20),
        "<h2>Worst fields by rollout final relative RMSE</h2>",
        df_html(field_metrics.sort_values('rollout_final_relative_rmse', ascending=False), 15) if not field_metrics.empty else "<p>No data.</p>",
        "<h2>Physical summary</h2>",
        df_html(physical_summary, 20),
        "<h2>Figures</h2>",
    ]
    for fig in figs:
        html.append(f"<h3>{fig.name}</h3><img src='{rel(fig)}'>")
    html.append("</body></html>")
    report.write_text("\n".join(html), encoding="utf-8")


def _write_readme(out_dir: Path):
    text = """Что смотреть в первую очередь
============================

1) validation_report.html
   Главный HTML-отчёт: ключевые метрики, худшие поля, графики и сравнение COMSOL vs prediction.

2) tables/metrics_per_field.csv
   Ошибка по каждому физическому полю: temperature, u/v/w, stress, strain и т.д.

3) tables/error_over_time.csv
   Накопление ошибки по времени во время autoregressive rollout.

4) figures/rollout_rmse_selected_fields.png
   Самый важный график: показывает, когда модель начинает расходиться с COMSOL.

5) figures/comparison_*png
   Срезы COMSOL / MGN / absolute error для понятных физических величин.

Как читать результаты
=====================

- one_step_* показывает качество локального перехода state_t -> state_t+1.
- rollout_* показывает качество настоящего прогноза, когда модель сама использует свои прошлые предсказания.
- Если one_step хороший, а rollout быстро портится — модель локально обучилась, но нестабильна при длинном прогнозе.
- Для обычного пользователя лучше смотреть temperature_change, displacement_magnitude, velocity_magnitude и von_mises_stress.
- Для исследователя дополнительно смотреть raw-поля stress/strain и error_over_time.csv.
"""
    (out_dir / "README_WHAT_TO_CHECK_FIRST.txt").write_text(text, encoding="utf-8")


def run_full_validation(
    config: Dict,
    dataset_id: str,
    checkpoint: str,
    split: str = "test",
    max_rollout_steps: int | None = None,
    output_dir: str | Path = "outputs/validation",
    slice_axis: str = "z",
    slice_value: float | None = None,
    slice_tol: float | None = None,
) -> Dict:
    data_cfg = config.get("data", {})
    registry_dir = data_cfg.get("registry_dir", "datasets")
    training_cfg = config.get("training", {})
    device = setup_device(training_cfg.get("device", "auto"))

    ds = load_processed_dataset(dataset_id, registry_dir)
    graph = ds["graph"]
    data = ds["data"]
    metadata = ds["metadata"]
    normalization = ds["normalization"]

    if split not in data:
        raise ValueError(f"Unknown split={split}. Available: {list(data.keys())}")
    samples = data[split]
    if not samples:
        raise ValueError(f"Split '{split}' is empty.")

    field_names = metadata["field_names"]
    field_units = metadata.get("field_units", {})
    F = len(field_names)
    S = int(metadata["node_in_dim"]) - F
    target_mode = metadata.get("target_mode", data_cfg.get("target_mode", "delta"))

    model = build_model(config, metadata["node_in_dim"], metadata["edge_in_dim"], F).to(device)
    load_checkpoint(checkpoint, model, map_location=device, strict=True)
    model.eval()

    edge_index = graph["edge_index"].to(device)
    edge_attr = graph["edge_attr"].to(device)
    coords = graph["coords"].detach().cpu().numpy()

    # 1) one-step validation
    one_true_norm, one_pred_norm = _build_one_step_arrays(model, samples, edge_index, edge_attr, S, target_mode, device)
    one_true_raw = _denormalize_states(one_true_norm, field_names, normalization)
    one_pred_raw = _denormalize_states(one_pred_norm, field_names, normalization)

    # 2) rollout validation
    roll_true_norm, roll_pred_norm = _rollout_arrays(model, samples, edge_index, edge_attr, S, target_mode, device, max_rollout_steps)
    roll_true_raw = _denormalize_states(roll_true_norm, field_names, normalization)
    roll_pred_raw = _denormalize_states(roll_pred_norm, field_names, normalization)

    all_times = metadata.get("times", [])
    # Split samples are sequential; approximate split start based on train/val/test sizes.
    n_train = int(metadata.get("n_train", 0))
    n_val = int(metadata.get("n_val", 0))
    split_start = 0 if split == "train" else n_train if split == "val" else n_train + n_val
    times = all_times[split_start : split_start + roll_true_raw.shape[0]] if all_times else list(range(roll_true_raw.shape[0]))
    if len(times) < roll_true_raw.shape[0]:
        times = list(times) + list(range(len(times), roll_true_raw.shape[0]))

    # 3) derived fields and metrics
    true_d = compute_derived_fields(roll_true_raw, field_names, coords)
    pred_d = compute_derived_fields(roll_pred_raw, field_names, coords)

    field_metrics = _field_metrics_table(field_names, field_units, one_true_raw, one_pred_raw, roll_true_raw, roll_pred_raw)
    error_time = _error_over_time_table(field_names, times, roll_true_raw, roll_pred_raw)
    derived_error = _derived_error_table(times, true_d, pred_d)
    physical_summary = _physical_summary(times, true_d, pred_d)
    group_metrics = _group_metrics(field_metrics)

    out_dir = Path(output_dir)
    tables_dir = out_dir / "tables"
    figs_dir = out_dir / "figures"
    arrays_dir = out_dir / "arrays"
    for d in [out_dir, tables_dir, figs_dir, arrays_dir]:
        d.mkdir(parents=True, exist_ok=True)

    field_metrics.to_csv(tables_dir / "metrics_per_field.csv", index=False)
    group_metrics.to_csv(tables_dir / "group_metrics.csv", index=False)
    error_time.to_csv(tables_dir / "error_over_time.csv", index=False)
    derived_error.to_csv(tables_dir / "derived_error_over_time.csv", index=False)
    physical_summary.to_csv(tables_dir / "physical_summary.csv", index=False)

    np.savez_compressed(
        arrays_dir / "validation_rollout_arrays.npz",
        coords=coords,
        times=np.asarray(times, dtype=np.float64),
        true=roll_true_raw.astype(np.float32),
        pred=roll_pred_raw.astype(np.float32),
        field_names=np.asarray(field_names, dtype=object),
    )

    # 4) figures
    _save_bar_plot(field_metrics, "field", "rollout_final_relative_rmse", "Worst fields by final relative rollout RMSE", figs_dir / "worst_fields_final_relative_rmse.png")
    _save_bar_plot(field_metrics, "field", "one_step_relative_rmse", "Worst fields by one-step relative RMSE", figs_dir / "worst_fields_one_step_relative_rmse.png")

    selected_fields = []
    for cand in ["t", "u", "v", "w", "ut", "vt", "wt", "solid.mises", "solid.sx", "solid.sy", "solid.sz"]:
        if cand in field_names:
            selected_fields.append(cand)
    if not selected_fields:
        selected_fields = field_names[: min(8, len(field_names))]
    et_sel = error_time[error_time["field"].isin(selected_fields)]
    _save_line_plot(et_sel, "step", "rmse", "field", "Rollout RMSE over time: selected raw fields", figs_dir / "rollout_rmse_selected_fields.png")
    _save_line_plot(et_sel, "step", "relative_rmse", "field", "Rollout relative RMSE over time: selected raw fields", figs_dir / "rollout_relative_rmse_selected_fields.png")

    if not derived_error.empty:
        main_derived = [q for q in ["temperature_change", "displacement_magnitude", "velocity_magnitude", "von_mises_stress", "strain_magnitude"] if q in set(derived_error["quantity"])]
        de_sel = derived_error[derived_error["quantity"].isin(main_derived)] if main_derived else derived_error
        _save_line_plot(de_sel, "step", "rmse", "quantity", "Derived quantity RMSE over time", figs_dir / "derived_rmse_over_time.png")
        _save_line_plot(de_sel, "step", "relative_rmse", "quantity", "Derived quantity relative RMSE over time", figs_dir / "derived_relative_rmse_over_time.png")

    # Wave-front radius from temperature_change, if available.
    center = _scenario_center(metadata, coords)
    front_mae = float("nan")
    if "temperature_change" in true_d and "temperature_change" in pred_d:
        fr_true = _wave_front_radius(true_d["temperature_change"], coords, center)
        fr_pred = _wave_front_radius(pred_d["temperature_change"], coords, center)
        front_mae = float(np.nanmean(np.abs(fr_pred - fr_true)))
        front_df = pd.DataFrame({"step": np.arange(len(fr_true)), "time": times[: len(fr_true)], "true_front_radius": fr_true, "pred_front_radius": fr_pred, "abs_error": np.abs(fr_pred - fr_true)})
        front_df.to_csv(tables_dir / "wave_front_radius.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(front_df["step"], front_df["true_front_radius"], label="COMSOL true")
        ax.plot(front_df["step"], front_df["pred_front_radius"], label="MGN prediction")
        ax.set_title("Temperature-change wave-front radius")
        ax.set_xlabel("step")
        ax.set_ylabel("radius")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / "wave_front_radius.png", dpi=160)
        plt.close(fig)

    # Comparison slices for readable quantities.
    steps_to_plot = sorted(set([0, max(0, roll_true_raw.shape[0] // 2), roll_true_raw.shape[0] - 1]))
    comparison_quantities: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    temp_idx = _field_index(field_names, ["t", "temperature", "temp"], contains=True)
    if temp_idx is not None:
        comparison_quantities["temperature"] = (roll_true_raw[:, :, temp_idx], roll_pred_raw[:, :, temp_idx])
    for key in ["temperature_change", "displacement_magnitude", "velocity_magnitude", "von_mises_stress", "strain_magnitude"]:
        if key in true_d and key in pred_d:
            comparison_quantities[key] = (true_d[key], pred_d[key])
    for q, (tv, pv) in comparison_quantities.items():
        for st in steps_to_plot:
            _save_comparison_slice(coords, tv, pv, q, st, slice_axis, figs_dir / f"comparison_{_safe_name(q)}_step{st:03d}.png", slice_value, slice_tol)

    # 5) summary and report
    summary = {
        "dataset_id": dataset_id,
        "split": split,
        "checkpoint": str(checkpoint),
        "target_mode": target_mode,
        "n_nodes": int(metadata.get("n_nodes", coords.shape[0])),
        "n_edges": int(metadata.get("n_edges", graph["edge_index"].shape[1])),
        "n_fields": int(F),
        "one_step_mean_rmse": float(field_metrics["one_step_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "one_step_mean_relative_rmse": float(field_metrics["one_step_relative_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "rollout_mean_rmse": float(field_metrics["rollout_mean_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "rollout_mean_relative_rmse": float(field_metrics["rollout_relative_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "rollout_final_rmse": float(field_metrics["rollout_final_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "rollout_final_relative_rmse": float(field_metrics["rollout_final_relative_rmse"].mean()) if not field_metrics.empty else float("nan"),
        "temperature_front_radius_mae": front_mae,
        "output_dir": str(out_dir),
    }
    with (out_dir / "validation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _make_html_report(out_dir, summary, field_metrics, group_metrics, physical_summary)
    _write_readme(out_dir)
    return summary
