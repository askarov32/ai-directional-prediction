"""Vector-field visualizations for thermoelastic wave propagation.

This module is designed for COMSOL/MeshGraphNet rollouts where dynamic fields
include displacement components (u, v, w) and/or velocity components (ut, vt, wt).
It produces readable 2D slice views with arrows showing wave direction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


# ---------------------------------------------------------------------------
# Field utilities
# ---------------------------------------------------------------------------

def _to_numpy(x):
    """Convert torch tensors / lists to numpy arrays without importing torch."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sanitize_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(".", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("^", "")
    )


def normalize_field_name(name: str) -> str:
    return str(name).strip().lower()


def find_field_idx(field_names: List[str], candidates: Iterable[str]) -> Optional[int]:
    """Find field index using exact and robust suffix matches.

    Examples matched:
    - "t", "temperature", "solid.T"
    - "u", "solid.u"
    - "solid.sx" for candidate "sx"
    """
    fn = [normalize_field_name(f) for f in field_names]
    cand_norm = [normalize_field_name(c) for c in candidates]

    # Exact match first.
    for cand in cand_norm:
        for i, f in enumerate(fn):
            if f == cand:
                return i

    # Suffix match: solid.sx should match sx.
    for cand in cand_norm:
        for i, f in enumerate(fn):
            if f.endswith("." + cand) or f.endswith("_" + cand):
                return i

    # Contains match for descriptive names.
    for cand in cand_norm:
        for i, f in enumerate(fn):
            if cand in f:
                return i
    return None


def get_vector_components(
    trajectory: np.ndarray,
    field_names: List[str],
    mode: str = "velocity",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, str, str]]:
    """Return vector components [T, N] for velocity or displacement."""
    mode = mode.lower().strip()
    if mode == "velocity":
        ix = find_field_idx(field_names, ["ut", "vx", "velocity_x", "v_x"])
        iy = find_field_idx(field_names, ["vt", "vy", "velocity_y", "v_y"])
        iz = find_field_idx(field_names, ["wt", "vz", "velocity_z", "v_z"])
        names = ("ut", "vt", "wt")
    elif mode == "displacement":
        ix = find_field_idx(field_names, ["u", "ux", "disp_x", "displacement_x"])
        iy = find_field_idx(field_names, ["v", "uy", "disp_y", "displacement_y"])
        iz = find_field_idx(field_names, ["w", "uz", "disp_z", "displacement_z"])
        names = ("u", "v", "w")
    else:
        raise ValueError("mode must be 'velocity' or 'displacement'")

    if ix is None or iy is None:
        raise ValueError(
            f"Не найдены компоненты {mode}. Нужны минимум X/Y компоненты. "
            f"Доступные поля: {field_names}"
        )
    vx = trajectory[:, :, ix]
    vy = trajectory[:, :, iy]
    vz = trajectory[:, :, iz] if iz is not None else np.zeros_like(vx)
    return vx, vy, vz, names


def compute_derived_fields_strong(trajectory: np.ndarray, field_names: List[str]) -> Dict[str, np.ndarray]:
    """Compute derived fields useful for wave interpretation.

    This is more robust than a minimal implementation because it supports COMSOL
    names such as solid.sx, solid.sy, solid.sz, solid.sxy, solid.sxz, solid.syz,
    and solid.mises.
    """
    derived: Dict[str, np.ndarray] = {}

    # Temperature.
    t_idx = find_field_idx(field_names, ["t", "temperature", "temp"])
    if t_idx is not None:
        temp = trajectory[:, :, t_idx]
        derived["temperature"] = temp
        derived["temperature_change"] = temp - temp[:1]
        derived["temperature_time_gradient"] = np.diff(temp, axis=0, prepend=temp[:1])

    # Displacement magnitude.
    try:
        ux, uy, uz, _ = get_vector_components(trajectory, field_names, mode="displacement")
        derived["displacement_magnitude"] = np.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    except Exception:
        pass

    # Velocity magnitude.
    try:
        vx, vy, vz, _ = get_vector_components(trajectory, field_names, mode="velocity")
        derived["velocity_magnitude"] = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    except Exception:
        pass

    # Use solid.mises directly if available.
    mises_idx = find_field_idx(field_names, ["solid.mises", "mises", "von_mises", "von_mises_stress"])
    if mises_idx is not None:
        vm = trajectory[:, :, mises_idx]
        derived["von_mises_stress"] = vm
        vmax = np.maximum(np.nanmax(np.abs(vm), axis=1, keepdims=True), 1e-12)
        derived["risk_flag"] = (np.abs(vm) / vmax >= 0.80).astype(np.float32)
        return derived

    # Otherwise compute von Mises from stress tensor components.
    sx = find_field_idx(field_names, ["solid.sx", "sx", "sxx", "s11"])
    sy = find_field_idx(field_names, ["solid.sy", "sy", "syy", "s22"])
    sz = find_field_idx(field_names, ["solid.sz", "sz", "szz", "s33"])
    sxy = find_field_idx(field_names, ["solid.sxy", "sxy", "s12"])
    sxz = find_field_idx(field_names, ["solid.sxz", "sxz", "s13"])
    syz = find_field_idx(field_names, ["solid.syz", "syz", "s23"])
    if None not in (sx, sy, sz, sxy, sxz, syz):
        sxx = trajectory[:, :, sx]
        syy = trajectory[:, :, sy]
        szz = trajectory[:, :, sz]
        sxy_v = trajectory[:, :, sxy]
        sxz_v = trajectory[:, :, sxz]
        syz_v = trajectory[:, :, syz]
        vm = np.sqrt(
            0.5 * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (sxy_v ** 2 + sxz_v ** 2 + syz_v ** 2)
            )
        )
        derived["von_mises_stress"] = vm
        vmax = np.maximum(np.nanmax(vm, axis=1, keepdims=True), 1e-12)
        derived["risk_flag"] = (vm / vmax >= 0.80).astype(np.float32)

    return derived


def get_scalar_values(
    trajectory: np.ndarray,
    field_names: List[str],
    derived: Dict[str, np.ndarray],
    scalar_name: str,
) -> np.ndarray:
    """Return scalar field [T, N] from raw fields or derived dict."""
    scalar_name = scalar_name.strip()
    if scalar_name in derived:
        return _to_numpy(derived[scalar_name])
    idx = find_field_idx(field_names, [scalar_name])
    if idx is None:
        raise ValueError(f"Не найдено scalar field '{scalar_name}'. Доступно: {field_names} + {list(derived)}")
    return trajectory[:, :, idx]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def ensure_3d_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])
    return coords[:, :3]


def auto_slice_mask(
    coords: np.ndarray,
    axis: str = "z",
    slice_value: Optional[float] = None,
    tol: Optional[float] = None,
    min_points: int = 300,
) -> Tuple[np.ndarray, float, float]:
    """Select a readable slice/band.

    If a thin median slice has too few points, automatically expands the band
    until min_points are included. This is important for unstructured COMSOL meshes.
    """
    coords = ensure_3d_coords(coords)
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("slice_axis must be x, y, or z")
    ai = axis_map[axis]
    vals = coords[:, ai]
    if slice_value is None:
        slice_value = float(np.median(vals))

    rng = float(np.nanmax(vals) - np.nanmin(vals))
    if rng <= 0:
        return np.ones(len(coords), dtype=bool), slice_value, max(1e-12, rng)

    if tol is None:
        # Start with 2% range, then expand if there are too few points.
        tol_candidate = max(rng * 0.02, 1e-12)
        for frac in [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50]:
            tol_candidate = max(rng * frac, 1e-12)
            mask = np.abs(vals - slice_value) <= tol_candidate
            if int(mask.sum()) >= min_points:
                return mask, slice_value, tol_candidate
        return np.abs(vals - slice_value) <= tol_candidate, slice_value, tol_candidate

    mask = np.abs(vals - slice_value) <= tol
    return mask, slice_value, tol


def project_slice(
    coords: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    vz: np.ndarray,
    axis: str = "z",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Project 3D coordinates and vectors to a 2D plotting plane."""
    coords = ensure_3d_coords(coords)
    if axis == "z":
        return coords[:, 0], coords[:, 1], vx, vy, "X", "Y"
    if axis == "y":
        return coords[:, 0], coords[:, 2], vx, vz, "X", "Z"
    if axis == "x":
        return coords[:, 1], coords[:, 2], vy, vz, "Y", "Z"
    raise ValueError("axis must be x, y, or z")


def grid_subsample_arrows(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    max_arrows: int = 350,
    prefer_strong: bool = True,
) -> np.ndarray:
    """Choose readable arrow indices via 2D grid binning.

    It prevents unreadable overplotting on dense point clouds. In each grid cell,
    chooses either the strongest vector or the first available point.
    """
    n = len(x)
    if n <= max_arrows:
        return np.arange(n)
    if max_arrows <= 0:
        return np.array([], dtype=int)

    grid_n = int(np.ceil(np.sqrt(max_arrows)))
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    if abs(xmax - xmin) < 1e-15 or abs(ymax - ymin) < 1e-15:
        step = max(1, n // max_arrows)
        return np.arange(0, n, step)[:max_arrows]

    xi = np.clip(((x - xmin) / (xmax - xmin) * grid_n).astype(int), 0, grid_n - 1)
    yi = np.clip(((y - ymin) / (ymax - ymin) * grid_n).astype(int), 0, grid_n - 1)
    mag = np.sqrt(u ** 2 + v ** 2)

    chosen: Dict[Tuple[int, int], int] = {}
    for idx, cell in enumerate(zip(xi, yi)):
        if cell not in chosen:
            chosen[cell] = idx
        elif prefer_strong and mag[idx] > mag[chosen[cell]]:
            chosen[cell] = idx

    out = np.array(list(chosen.values()), dtype=int)
    if len(out) > max_arrows:
        order = np.argsort(mag[out])[::-1]
        out = out[order[:max_arrows]]
    return np.sort(out)


def visible_arrow_components(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    arrow_length_fraction: float = 0.045,
    clip_percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale physical vectors to visible plot lengths while preserving direction.

    Physical thermoelastic displacements/velocities can be tiny, so raw quiver
    lengths may be invisible. This function maps p95 magnitude to a fixed
    fraction of plot size. Color still represents the true magnitude.
    """
    mag = np.sqrt(u ** 2 + v ** 2)
    finite_mag = mag[np.isfinite(mag)]
    if finite_mag.size == 0 or float(np.nanmax(finite_mag)) <= 0:
        return np.zeros_like(u), np.zeros_like(v), mag

    ref = float(np.nanpercentile(finite_mag, clip_percentile))
    if ref <= 1e-30:
        ref = float(np.nanmax(finite_mag)) + 1e-30

    xr = float(np.nanmax(x) - np.nanmin(x))
    yr = float(np.nanmax(y) - np.nanmin(y))
    domain = max(np.sqrt(xr * xr + yr * yr), 1e-12)
    scale = domain * arrow_length_fraction / ref

    u_plot = u * scale
    v_plot = v * scale

    # Cap extremely long arrows.
    plot_mag = np.sqrt(u_plot ** 2 + v_plot ** 2)
    max_len = domain * arrow_length_fraction * 2.0
    factor = np.minimum(1.0, max_len / (plot_mag + 1e-30))
    return u_plot * factor, v_plot * factor, mag


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_quiver_slice(
    coords: np.ndarray,
    trajectory: np.ndarray,
    field_names: List[str],
    timestep: int,
    vector_mode: str = "velocity",
    scalar_name: str = "temperature_change",
    derived: Optional[Dict[str, np.ndarray]] = None,
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
    tol: Optional[float] = None,
    min_slice_points: int = 300,
    max_arrows: int = 350,
    arrow_length_fraction: float = 0.045,
    save_path: str | Path | None = None,
    title: Optional[str] = None,
) -> str:
    """Save one 2D slice with scalar background and vector arrows."""
    coords = ensure_3d_coords(coords)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    derived = derived or compute_derived_fields_strong(trajectory, field_names)

    T = trajectory.shape[0]
    t = int(timestep if timestep >= 0 else T - 1)
    t = max(0, min(T - 1, t))

    scalar_all = get_scalar_values(trajectory, field_names, derived, scalar_name)
    scalar = scalar_all[t]
    vx_all, vy_all, vz_all, _ = get_vector_components(trajectory, field_names, vector_mode)
    vx, vy, vz = vx_all[t], vy_all[t], vz_all[t]

    mask, actual_slice, actual_tol = auto_slice_mask(coords, slice_axis, slice_value, tol, min_slice_points)
    c_s = coords[mask]
    scalar_s = scalar[mask]
    vx_s, vy_s, vz_s = vx[mask], vy[mask], vz[mask]

    x2d, y2d, u2d, v2d, xlabel, ylabel = project_slice(c_s, vx_s, vy_s, vz_s, slice_axis)
    idx = grid_subsample_arrows(x2d, y2d, u2d, v2d, max_arrows=max_arrows)
    xq, yq, uq, vq = x2d[idx], y2d[idx], u2d[idx], v2d[idx]
    uq_vis, vq_vis, mag_true = visible_arrow_components(
        xq, yq, uq, vq, arrow_length_fraction=arrow_length_fraction
    )

    fig, ax = plt.subplots(figsize=(11, 8.5))
    sc = ax.scatter(x2d, y2d, c=scalar_s, s=12, alpha=0.88, cmap="plasma")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(scalar_name)

    if len(xq) > 0 and np.nanmax(np.abs(mag_true)) > 0:
        q = ax.quiver(
            xq,
            yq,
            uq_vis,
            vq_vis,
            mag_true,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="viridis",
            width=0.0035,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=4.0,
        )
        cbar2 = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.10)
        cbar2.set_label(f"true {vector_mode} magnitude")
    else:
        ax.text(
            0.5,
            0.02,
            "Vectors are zero or too small at this timestep",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(
        title
        or f"{scalar_name} + {vector_mode} arrows | step={t} | "
        f"slice {slice_axis}={actual_slice:.5g} ± {actual_tol:.3g} | arrows={len(xq)}"
    )
    ax.grid(True, alpha=0.25)

    if save_path is None:
        save_path = Path("outputs") / "wave_arrows" / "figures" / f"quiver_{sanitize_name(scalar_name)}_{vector_mode}_t{t}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def animate_quiver_slice(
    coords: np.ndarray,
    trajectory: np.ndarray,
    field_names: List[str],
    vector_mode: str = "velocity",
    scalar_name: str = "temperature_change",
    derived: Optional[Dict[str, np.ndarray]] = None,
    slice_axis: str = "z",
    slice_value: Optional[float] = None,
    tol: Optional[float] = None,
    min_slice_points: int = 300,
    max_arrows: int = 350,
    arrow_length_fraction: float = 0.045,
    save_path: str | Path = "outputs/wave_arrows/animations/quiver_temperature_velocity.gif",
    fps: int = 8,
    max_frames: int = 160,
) -> str:
    """Save GIF/MP4 animation of scalar background + vector arrows on a slice."""
    coords = ensure_3d_coords(coords)
    trajectory = np.asarray(trajectory, dtype=np.float64)
    derived = derived or compute_derived_fields_strong(trajectory, field_names)
    T = min(int(max_frames), trajectory.shape[0])

    scalar_all = get_scalar_values(trajectory, field_names, derived, scalar_name)[:T]
    vx_all, vy_all, vz_all, _ = get_vector_components(trajectory, field_names, vector_mode)
    vx_all, vy_all, vz_all = vx_all[:T], vy_all[:T], vz_all[:T]

    mask, actual_slice, actual_tol = auto_slice_mask(coords, slice_axis, slice_value, tol, min_slice_points)
    c_s = coords[mask]
    scalar_s = scalar_all[:, mask]
    vx_s, vy_s, vz_s = vx_all[:, mask], vy_all[:, mask], vz_all[:, mask]

    x2d, y2d, _, _, xlabel, ylabel = project_slice(c_s, vx_s[0], vy_s[0], vz_s[0], slice_axis)

    # Choose stable arrow positions based on maximum magnitude over time.
    _, _, u_first, v_first, _, _ = project_slice(c_s, vx_s[0], vy_s[0], vz_s[0], slice_axis)
    max_mag = np.zeros(len(c_s), dtype=np.float64)
    for frame in range(T):
        _, _, uf, vf, _, _ = project_slice(c_s, vx_s[frame], vy_s[frame], vz_s[frame], slice_axis)
        max_mag = np.maximum(max_mag, np.sqrt(uf ** 2 + vf ** 2))
    idx = grid_subsample_arrows(x2d, y2d, max_mag, np.zeros_like(max_mag), max_arrows=max_arrows)

    vmin = float(np.nanpercentile(scalar_s, 1))
    vmax = float(np.nanpercentile(scalar_s, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
        vmin = float(np.nanmin(scalar_s))
        vmax = float(np.nanmax(scalar_s))
        if abs(vmax - vmin) < 1e-12:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(11, 8.5))
    sc = ax.scatter(x2d, y2d, c=scalar_s[0], s=12, alpha=0.88, cmap="plasma", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(scalar_name)

    xq = x2d[idx]
    yq = y2d[idx]
    _, _, u0, v0, _, _ = project_slice(c_s, vx_s[0], vy_s[0], vz_s[0], slice_axis)
    uq_vis, vq_vis, mag_true = visible_arrow_components(
        xq, yq, u0[idx], v0[idx], arrow_length_fraction=arrow_length_fraction
    )
    q = ax.quiver(
        xq,
        yq,
        uq_vis,
        vq_vis,
        mag_true,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        cmap="viridis",
        width=0.0035,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
    )
    cbar2 = fig.colorbar(q, ax=ax, fraction=0.046, pad=0.10)
    cbar2.set_label(f"true {vector_mode} magnitude")

    title = ax.set_title("")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    def update(frame: int):
        sc.set_array(scalar_s[frame])
        _, _, uf, vf, _, _ = project_slice(c_s, vx_s[frame], vy_s[frame], vz_s[frame], slice_axis)
        uq_vis_f, vq_vis_f, mag_f = visible_arrow_components(
            xq, yq, uf[idx], vf[idx], arrow_length_fraction=arrow_length_fraction
        )
        q.set_UVC(uq_vis_f, vq_vis_f, mag_f)
        title.set_text(
            f"{scalar_name} + {vector_mode} arrows | step={frame}/{T - 1} | "
            f"slice {slice_axis}={actual_slice:.5g} ± {actual_tol:.3g} | arrows={len(idx)}"
        )
        return sc, q, title

    anim = FuncAnimation(fig, update, frames=T, interval=1000 / max(fps, 1), blit=False)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if save_path.suffix.lower() == ".mp4":
            writer = FFMpegWriter(fps=fps)
        else:
            writer = PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer)
    finally:
        plt.close(fig)
    return str(save_path)


def plot_wave_front_radius(
    coords: np.ndarray,
    scalar_over_time: np.ndarray,
    times: Optional[List[float]] = None,
    threshold_fraction: float = 0.20,
    source_center: Optional[np.ndarray] = None,
    save_path: str | Path = "outputs/wave_arrows/figures/wave_front_radius.png",
) -> str:
    """Plot a simple radius of active wave front over time.

    Uses nodes where abs(scalar) exceeds threshold_fraction * max(abs(scalar_t)).
    This works best with temperature_change or velocity_magnitude.
    """
    coords = ensure_3d_coords(coords)
    values = np.asarray(scalar_over_time)
    if source_center is None:
        # Use strongest initial disturbance as center when possible.
        idx = int(np.nanargmax(np.abs(values[0]))) if np.nanmax(np.abs(values[0])) > 0 else 0
        source_center = coords[idx]
    source_center = np.asarray(source_center).reshape(1, 3)
    radius_nodes = np.linalg.norm(coords - source_center, axis=1)

    radii = []
    for t in range(values.shape[0]):
        v = np.abs(values[t])
        vmax = float(np.nanmax(v))
        if vmax <= 0 or not np.isfinite(vmax):
            radii.append(0.0)
            continue
        active = v >= threshold_fraction * vmax
        radii.append(float(np.nanpercentile(radius_nodes[active], 95)) if np.any(active) else 0.0)

    if times is None or len(times) != len(radii):
        x = np.arange(len(radii))
        xlabel = "step"
    else:
        x = times
        xlabel = "time"

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, radii, marker="o", markersize=2)
    ax.set_title("Estimated wave-front radius over time")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("front radius")
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def plot_research_time_series(
    trajectory: np.ndarray,
    field_names: List[str],
    derived: Dict[str, np.ndarray],
    times: Optional[List[float]],
    output_dir: str | Path,
) -> List[str]:
    """Save compact time-series plots for the most interpretable wave fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    names = [
        "temperature_change",
        "displacement_magnitude",
        "velocity_magnitude",
        "von_mises_stress",
        "risk_flag",
    ]
    x = np.asarray(times) if times is not None and len(times) == trajectory.shape[0] else np.arange(trajectory.shape[0])
    xlabel = "time" if times is not None and len(times) == trajectory.shape[0] else "step"

    for name in names:
        if name not in derived:
            continue
        v = np.asarray(derived[name])
        if v.ndim != 2:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(x, np.nanmean(v, axis=1), label="mean")
        ax.plot(x, np.nanmax(v, axis=1), label="max")
        ax.plot(x, np.nanpercentile(v, 95, axis=1), label="p95")
        ax.set_title(f"{name}: mean / p95 / max over time")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = output_dir / f"timeseries_{sanitize_name(name)}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p))
    return paths
