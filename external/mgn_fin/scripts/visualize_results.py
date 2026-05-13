from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Make project root importable when launched as: python scripts/visualize_results.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.visualization.vector_fields import (
    animate_quiver_slice,
    compute_derived_fields_strong,
    get_scalar_values,
    plot_quiver_slice,
    plot_research_time_series,
    plot_wave_front_radius,
    sanitize_name,
)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_prediction(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"prediction file not found: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("prediction.pt must contain a dict")

    trajectory = _to_numpy(obj.get("trajectory"))
    coords = _to_numpy(obj.get("coords"))
    field_names = list(obj.get("field_names", []))
    times = obj.get("times", None)
    if times is not None:
        times = [float(t) for t in times]
    derived_raw = obj.get("derived", {}) or {}
    derived = {k: _to_numpy(v) for k, v in derived_raw.items()}

    if trajectory.ndim != 3:
        raise ValueError(f"trajectory must be [T,N,F], got shape {trajectory.shape}")
    if coords.ndim != 2:
        raise ValueError(f"coords must be [N,2/3], got shape {coords.shape}")
    if not field_names:
        raise ValueError("field_names is empty in prediction.pt")
    return trajectory, coords, field_names, times, derived


def write_html_report(output_dir: Path, artifacts: Dict[str, List[str]], summary: Dict):
    html_path = output_dir / "wave_arrows_report.html"
    rel = lambda p: Path(p).resolve().relative_to(output_dir.resolve()).as_posix() if str(p).startswith(str(output_dir.resolve())) else str(p)

    def img_block(title: str, paths: List[str]):
        if not paths:
            return ""
        out = [f"<h2>{title}</h2>"]
        for p in paths:
            rp = rel(str(Path(p).resolve()))
            name = Path(p).name
            if Path(p).suffix.lower() in {".gif", ".png", ".jpg", ".jpeg"}:
                out.append(f"<figure><figcaption>{name}</figcaption><img src='{rp}' loading='lazy'></figure>")
            else:
                out.append(f"<p><a href='{rp}'>{name}</a></p>")
        return "\n".join(out)

    html = f"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Thermoelastic wave arrows report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; line-height: 1.45; background: #f7f7f7; color: #222; }}
h1, h2 {{ color: #111; }}
.card {{ background: white; padding: 18px 22px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,.08); margin-bottom: 22px; }}
img {{ max-width: 100%; border-radius: 8px; border: 1px solid #ddd; background: white; }}
figure {{ margin: 18px 0; }}
figcaption {{ font-weight: bold; margin-bottom: 6px; }}
code, pre {{ background: #eee; padding: 2px 5px; border-radius: 4px; }}
ul {{ margin-top: 6px; }}
</style>
</head>
<body>
<div class="card">
<h1>Визуализация распространения термоупругих волн стрелками</h1>
<p>Фон показывает скалярное поле: обычно <code>temperature_change</code>, <code>von_mises_stress</code> или <code>risk_flag</code>.</p>
<p>Стрелки показывают направление волны: <code>velocity</code> использует <code>ut/vt/wt</code>, <code>displacement</code> использует <code>u/v/w</code>.</p>
<p><b>Важно:</b> длина стрелок масштабирована для видимости. Цвет стрелок показывает истинную величину вектора.</p>
</div>
<div class="card">
<h2>Краткая сводка</h2>
<pre>{json.dumps(summary, indent=2, ensure_ascii=False)}</pre>
</div>
<div class="card">
<h2>Что открыть первым</h2>
<ul>
<li><code>animations/quiver_temperature_change_velocity.gif</code> — основной просмотр распространения волны.</li>
<li><code>animations/quiver_von_mises_stress_displacement.gif</code> — напряжения + направление деформации.</li>
<li><code>figures/wave_front_radius.png</code> — движется ли фронт наружу.</li>
<li><code>figures/timeseries_velocity_magnitude.png</code> — активность механической волны во времени.</li>
</ul>
</div>
<div class="card">
{img_block("Анимации со стрелками", artifacts.get("animations", []))}
</div>
<div class="card">
{img_block("Картинки-срезы", artifacts.get("figures", []))}
</div>
<div class="card">
{img_block("Графики по времени", artifacts.get("timeseries", []))}
</div>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def parse_args():
    p = argparse.ArgumentParser(description="Research visualization: wave propagation as arrows on 2D slices")
    p.add_argument("--config", default=None, help="Optional config path. Not required for visualization.")
    p.add_argument("--dataset_id", default=None, help="Dataset ID, only used in report naming/context.")
    p.add_argument("--prediction", default="outputs/predictions/prediction.pt")
    p.add_argument("--output_dir", default="outputs/wave_arrows")
    p.add_argument("--slice_axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--slice_value", type=float, default=None)
    p.add_argument("--slice_tol", type=float, default=None)
    p.add_argument("--min_slice_points", type=int, default=300)
    p.add_argument("--max_arrows", type=int, default=320)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=120)
    p.add_argument("--make_mp4", action="store_true", help="Also save mp4 animations. Requires ffmpeg.")
    p.add_argument("--no_animations", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    figures_dir = out / "figures"
    animations_dir = out / "animations"
    out.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    animations_dir.mkdir(parents=True, exist_ok=True)

    trajectory, coords, field_names, times, derived_from_file = load_prediction(args.prediction)
    derived = compute_derived_fields_strong(trajectory, field_names)
    # Keep any file-derived fields that were not recomputed.
    for k, v in derived_from_file.items():
        derived.setdefault(k, v)

    artifacts: Dict[str, List[str]] = {"figures": [], "animations": [], "timeseries": []}

    print("=" * 70)
    print("Vector wave visualization")
    print(f"Prediction: {args.prediction}")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Coords shape: {coords.shape}")
    print(f"Fields: {field_names}")
    print(f"Derived: {list(derived)}")
    print("=" * 70)

    # Key timesteps for static slices.
    T = trajectory.shape[0]
    key_steps = sorted(set([0, max(0, T // 4), max(0, T // 2), max(0, 3 * T // 4), T - 1]))

    # Most important views: scalar background + vector arrows.
    views = [
        ("temperature_change", "velocity", "main wave: thermal change + velocity arrows"),
        ("temperature_change", "displacement", "thermal change + displacement arrows"),
        ("von_mises_stress", "displacement", "stress + displacement arrows"),
        ("risk_flag", "velocity", "risk zones + velocity arrows"),
    ]

    for scalar_name, vector_mode, title in views:
        if scalar_name not in derived:
            print(f"[skip] scalar not available: {scalar_name}")
            continue
        try:
            for step in key_steps:
                p = figures_dir / f"quiver_{sanitize_name(scalar_name)}_{vector_mode}_t{step}.png"
                artifacts["figures"].append(
                    plot_quiver_slice(
                        coords=coords,
                        trajectory=trajectory,
                        field_names=field_names,
                        timestep=step,
                        vector_mode=vector_mode,
                        scalar_name=scalar_name,
                        derived=derived,
                        slice_axis=args.slice_axis,
                        slice_value=args.slice_value,
                        tol=args.slice_tol,
                        min_slice_points=args.min_slice_points,
                        max_arrows=args.max_arrows,
                        save_path=p,
                        title=f"{title} | step={step}",
                    )
                )
            if not args.no_animations:
                gif_path = animations_dir / f"quiver_{sanitize_name(scalar_name)}_{vector_mode}.gif"
                artifacts["animations"].append(
                    animate_quiver_slice(
                        coords=coords,
                        trajectory=trajectory,
                        field_names=field_names,
                        vector_mode=vector_mode,
                        scalar_name=scalar_name,
                        derived=derived,
                        slice_axis=args.slice_axis,
                        slice_value=args.slice_value,
                        tol=args.slice_tol,
                        min_slice_points=args.min_slice_points,
                        max_arrows=args.max_arrows,
                        save_path=gif_path,
                        fps=args.fps,
                        max_frames=args.max_frames,
                    )
                )
                if args.make_mp4:
                    mp4_path = animations_dir / f"quiver_{sanitize_name(scalar_name)}_{vector_mode}.mp4"
                    try:
                        artifacts["animations"].append(
                            animate_quiver_slice(
                                coords=coords,
                                trajectory=trajectory,
                                field_names=field_names,
                                vector_mode=vector_mode,
                                scalar_name=scalar_name,
                                derived=derived,
                                slice_axis=args.slice_axis,
                                slice_value=args.slice_value,
                                tol=args.slice_tol,
                                min_slice_points=args.min_slice_points,
                                max_arrows=args.max_arrows,
                                save_path=mp4_path,
                                fps=args.fps,
                                max_frames=args.max_frames,
                            )
                        )
                    except Exception as e:
                        print(f"[warning] mp4 was not created: {e}")
        except Exception as e:
            print(f"[warning] failed view scalar={scalar_name} vector={vector_mode}: {e}")

    # Time series and front radius.
    artifacts["timeseries"].extend(plot_research_time_series(trajectory, field_names, derived, times, figures_dir))
    if "temperature_change" in derived:
        artifacts["timeseries"].append(
            plot_wave_front_radius(
                coords=coords,
                scalar_over_time=derived["temperature_change"],
                times=times,
                threshold_fraction=0.20,
                save_path=figures_dir / "wave_front_radius_temperature_change.png",
            )
        )
    elif "velocity_magnitude" in derived:
        artifacts["timeseries"].append(
            plot_wave_front_radius(
                coords=coords,
                scalar_over_time=derived["velocity_magnitude"],
                times=times,
                threshold_fraction=0.20,
                save_path=figures_dir / "wave_front_radius_velocity_magnitude.png",
            )
        )

    summary = {
        "dataset_id": args.dataset_id,
        "prediction": str(args.prediction),
        "trajectory_shape": list(trajectory.shape),
        "coords_shape": list(coords.shape),
        "field_names": field_names,
        "derived_fields": list(derived),
        "slice_axis": args.slice_axis,
        "max_arrows": args.max_arrows,
        "outputs": artifacts,
    }
    (out / "wave_arrows_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    html = write_html_report(out, artifacts, summary)

    readme = out / "README_OPEN_FIRST.txt"
    readme.write_text(
        "ОТКРЫТЬ СНАЧАЛА:\n"
        f"1) {html}\n"
        "2) animations/quiver_temperature_change_velocity.gif\n"
        "3) figures/wave_front_radius_temperature_change.png\n\n"
        "Смысл визуализации:\n"
        "- фон = scalar field, обычно temperature_change / von_mises_stress / risk_flag;\n"
        "- стрелки = направление velocity или displacement;\n"
        "- длина стрелок масштабирована для видимости;\n"
        "- цвет стрелок = истинная величина вектора.\n",
        encoding="utf-8",
    )

    print("\n✅ Wave-arrow visualization complete")
    print(f"Open report: {html}")
    print(f"Open folder: {out}")


if __name__ == "__main__":
    main()
