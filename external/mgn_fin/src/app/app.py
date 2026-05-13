"""Commercial Streamlit UI for MeshGraphNet Thermoelastic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_registry import list_dataset_ids, load_scenario, register_dataset
from src.data.pipeline import run_pipeline
from src.data.scenario import default_scenario, save_yaml
from src.training.train import load_config

st.set_page_config(page_title="MeshGraphNet Thermoelastic Commercial", page_icon="🌋", layout="wide")
st.title("🌋 MeshGraphNet Commercial: термоупругие волны в геологических средах")
st.caption("Real COMSOL → Conditional MeshGraphNet → Rollout → 3D/VTK/Animation")

CONFIG_PATH = PROJECT_ROOT / "configs" / "base.yaml"


def run_cli(args):
    p = subprocess.run([sys.executable] + args, cwd=PROJECT_ROOT, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr


try:
    cfg = load_config(CONFIG_PATH)
except Exception:
    cfg = {}
registry_dir = cfg.get("data", {}).get("registry_dir", "datasets")
ids = list_dataset_ids(PROJECT_ROOT / registry_dir)
processed_ids = list_dataset_ids(PROJECT_ROOT / registry_dir, require_processed=True)

tabs = st.tabs([
    "📊 Dashboard", "📁 Dataset Registry", "✅ Validation", "🧠 Training", "🔁 Fine-tuning", "🔮 Prediction", "🎞 Visualization", "📐 COMSOL vs AI"
])

with tabs[0]:
    st.header("Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered datasets", len(ids))
    c2.metric("Processed datasets", len(processed_ids))
    c3.metric("Checkpoints", len(list((PROJECT_ROOT / "outputs/checkpoints").glob("*.pt"))))
    c4.metric("Predictions", len(list((PROJECT_ROOT / "outputs/predictions").glob("*.pt"))))
    st.write("**Datasets:**", processed_ids or ids or "No datasets yet")
    hist = PROJECT_ROOT / "outputs/logs/train_history.json"
    if hist.exists():
        df = pd.read_json(hist)
        st.line_chart(df.set_index("epoch")[["train_loss", "val_loss"]])

with tabs[1]:
    st.header("Register real COMSOL dataset")
    dataset_id = st.text_input("Dataset ID", "sandstone_773K")
    raw_dir = st.text_input("Raw COMSOL directory", f"datasets/{dataset_id}/raw")
    mesh_file = st.text_input("Mesh file (.mphtxt)", f"datasets/{dataset_id}/raw/sandstone.mphtxt")
    col1, col2, col3 = st.columns(3)
    with col1:
        rock_type = st.text_input("Rock type", "sandstone")
        physics_type = st.text_input("Physics type", "thermoelastic_wave")
        source_type = st.text_input("Source type", "heated_rod")
    with col2:
        initial_temperature = st.number_input("Source temperature K", value=773.15)
        background_temperature = st.number_input("Background temperature K", value=293.15)
        source_radius = st.number_input("Source radius m", value=0.01, format="%.6f")
    with col3:
        young = st.number_input("Young modulus Pa", value=3.0e10, format="%.4e")
        nu = st.number_input("Poisson ratio", value=0.25)
        density = st.number_input("Density kg/m3", value=2200.0)
    if st.button("Register dataset", type="primary"):
        sc = default_scenario(dataset_id)
        sc["rock_type"] = rock_type
        sc["physics"]["type"] = physics_type
        sc["source"].update({
            "type": source_type,
            "initial_temperature": float(initial_temperature),
            "background_temperature": float(background_temperature),
            "radius": float(source_radius),
        })
        sc["material"].update({"young_modulus": float(young), "poisson_ratio": float(nu), "density": float(density)})
        try:
            d = register_dataset(dataset_id, PROJECT_ROOT / raw_dir, PROJECT_ROOT / mesh_file, PROJECT_ROOT / registry_dir, sc)
            st.success(f"Registered: {d}")
        except Exception as e:
            st.error(str(e))

with tabs[2]:
    st.header("Dataset processing / validation")
    ds = st.selectbox("Dataset", ids or [""], key="prep_ds")
    if st.button("Prepare selected dataset", type="primary") and ds:
        try:
            meta = run_pipeline(ds, cfg.get("data", {}), PROJECT_ROOT / registry_dir)
            st.success("Dataset prepared")
            st.json(meta)
        except Exception as e:
            st.error(str(e))
    if ds:
        md = PROJECT_ROOT / registry_dir / ds / "processed" / "metadata.json"
        prev = PROJECT_ROOT / registry_dir / ds / "processed" / "preview.csv"
        if md.exists():
            st.subheader("Metadata")
            st.json(json.loads(md.read_text(encoding="utf-8")))
        if prev.exists():
            st.subheader("Field preview")
            st.dataframe(pd.read_csv(prev), use_container_width=True)

with tabs[3]:
    st.header("Base model training")
    train_ids = st.multiselect("Datasets", processed_ids, default=processed_ids)
    epochs = st.number_input("Epochs", value=int(cfg.get("training", {}).get("epochs", 200)), min_value=1)
    lr = st.number_input("Learning rate", value=float(cfg.get("training", {}).get("lr", 3e-4)), format="%.6f")
    if st.button("Start training", type="primary"):
        args = ["scripts/train_base_model.py", "--config", "configs/base.yaml", "--epochs", str(epochs), "--lr", str(lr)]
        if train_ids:
            args += ["--dataset_ids"] + train_ids
        code, out, err = run_cli(args)
        st.code(out + err)
        st.success("Training finished") if code == 0 else st.error("Training failed")

with tabs[4]:
    st.header("Fine-tuning on new rock/scenario")
    ft_ids = st.multiselect("Fine-tune datasets", processed_ids, default=processed_ids[:1])
    ckpts = [str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "outputs/checkpoints").glob("*.pt")]
    ckpt = st.selectbox("Base checkpoint", ckpts or ["outputs/checkpoints/best_model.pt"])
    ft_epochs = st.number_input("Fine-tune epochs", value=80, min_value=1)
    ft_lr = st.number_input("Fine-tune LR", value=5e-5, format="%.7f")
    if st.button("Start fine-tuning", type="primary"):
        args = ["scripts/finetune_model.py", "--config", "configs/finetune.yaml", "--checkpoint", ckpt, "--epochs", str(ft_epochs), "--lr", str(ft_lr)]
        if ft_ids:
            args += ["--dataset_ids"] + ft_ids
        code, out, err = run_cli(args)
        st.code(out + err)
        st.success("Fine-tuning finished") if code == 0 else st.error("Fine-tuning failed")

with tabs[5]:
    st.header("Prediction scenario")
    pred_ds = st.selectbox("Prediction dataset", processed_ids or [""], key="pred_ds")
    ckpts = [str(p.relative_to(PROJECT_ROOT)) for p in (PROJECT_ROOT / "outputs/checkpoints").glob("*.pt")]
    pred_ckpt = st.selectbox("Checkpoint", ckpts or ["outputs/checkpoints/best_model.pt"], key="pred_ckpt")
    steps = st.number_input("Rollout steps", value=100, min_value=1)
    init_source = st.selectbox("Initial state", ["from_dataset", "nearest_scenario", "user_defined"])
    if st.button("Run prediction", type="primary"):
        args = ["scripts/run_prediction.py", "--config", "configs/inference.yaml", "--dataset_id", pred_ds, "--checkpoint", pred_ckpt, "--rollout_steps", str(steps), "--initial_state_source", init_source]
        code, out, err = run_cli(args)
        st.code(out + err)
        st.success("Prediction finished") if code == 0 else st.error("Prediction failed")

with tabs[6]:
    st.header("Visualization / Export")
    fig_dir = PROJECT_ROOT / "outputs/figures"
    anim_dir = PROJECT_ROOT / "outputs/animations"
    figs = sorted(fig_dir.glob("*.png"))
    anims = sorted(anim_dir.glob("*.gif")) + sorted(anim_dir.glob("*.mp4"))
    st.write(f"Figures: {len(figs)} | Animations: {len(anims)}")
    for p in figs[:20]:
        st.image(str(p), caption=p.name)
    for p in anims[:10]:
        st.write(p.name)

with tabs[7]:
    st.header("COMSOL vs AI Validation")
    st.info("Use scripts/evaluate_model.py for one-step metrics. For full rollout comparison, export COMSOL ground truth trajectory with the same fields/times and extend src/visualization/comparison.py.")
    eval_ds = st.multiselect("Evaluation datasets", processed_ids, default=processed_ids[:1])
    eval_ckpt = st.selectbox("Checkpoint for evaluation", ckpts or ["outputs/checkpoints/best_model.pt"], key="eval_ckpt")
    if st.button("Evaluate checkpoint"):
        args = ["scripts/evaluate_model.py", "--config", "configs/base.yaml", "--checkpoint", eval_ckpt]
        if eval_ds:
            args += ["--dataset_ids"] + eval_ds
        code, out, err = run_cli(args)
        st.code(out + err)
        metrics = PROJECT_ROOT / "outputs/logs/evaluation_metrics.json"
        if metrics.exists():
            st.json(json.loads(metrics.read_text(encoding="utf-8")))
