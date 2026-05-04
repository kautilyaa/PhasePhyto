"""Streamlit interface for the PhasePhyto demo."""
from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from phasephyto.demo_runtime import (
    PLANT_DISEASE_25_CLASSES,
    available_plot_paths,
    choose_device,
    discover_model_artifacts,
    load_demo_model,
    load_demo_samples,
    predict_image,
    read_json,
    read_text,
    resolve_artifact_root,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhasePhyto Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* General */
    .main > div { padding-top: 0.8rem; }
    [data-testid="stSidebar"] { background: #0d2137; }
    [data-testid="stSidebar"] * { color: #e8f0fe !important; }
    [data-testid="stSidebar"] .stRadio label { font-size: 0.9rem; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #7dd3fc !important; }

    /* Hero banner */
    .pp-hero {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f3d2e 0%, #105b72 55%, #1a2340 100%);
        color: white; margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }
    .pp-hero h1 { margin: 0 0 0.3rem 0; font-size: 1.6rem; }
    .pp-subtitle { color: rgba(255,255,255,0.82); font-size: 0.93rem; line-height: 1.5; }
    .pp-badge {
        display: inline-block;
        padding: 0.18rem 0.55rem; margin: 0.1rem 0.25rem 0.1rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.15);
        font-size: 0.76rem;
    }

    /* Result card */
    .pp-result-card {
        border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem; background: #f8fafc; margin-bottom: 0.5rem;
    }
    .pp-pred-label { font-size: 1.25rem; font-weight: 700; color: #1e40af; }
    .pp-confidence { font-size: 1.0rem; color: #15803d; }

    /* Plot gallery */
    .pp-plot-title { font-size: 0.82rem; color: #64748b; text-align: center;
                     margin-top: 0.35rem; }

    /* Sample thumb button */
    div[data-testid="stButton"] > button {
        border-radius: 10px; border: 2px solid transparent;
        transition: border 0.15s;
    }
    div[data-testid="stButton"] > button:hover { border: 2px solid #3b82f6; }

    /* Tables */
    .dataframe { font-size: 0.85rem !important; }

    /* Metric delta color override */
    [data-testid="stMetricDelta"] { font-size: 0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data loading ──────────────────────────────────────────────────────────────
artifact_root = resolve_artifact_root()
models = discover_model_artifacts(artifact_root)
samples = load_demo_samples()
device = choose_device()

inferable = {k: v for k, v in models.items() if v.can_infer}

# ── Model descriptions ────────────────────────────────────────────────────────
MODEL_DESCRIPTIONS: dict[str, str] = {
    "full":          "All 3 streams + cross-attention fusion (87M params)",
    "full_ema":      "Full model with EMA weights",
    "backbone_only": "ViT-B/16 + CLAHE only, no PC stream",
    "no_fusion":     "All 3 streams, simple concat (no cross-attention)",
    "baseline":      "Plain fine-tuned ViT-B/16 with no OOD recipe",
    "pc_only":       "Phase Congruency + CLAHE only (metrics only)",
}

ABLATION_COLORS: dict[str, str] = {
    "full":          "#2563eb",
    "full_ema":      "#7c3aed",
    "backbone_only": "#0891b2",
    "no_fusion":     "#d97706",
    "baseline":      "#64748b",
    "pc_only":       "#dc2626",
}

# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model(model_key: str, _root_str: str, _device: str):
    return load_demo_model(models[model_key], device=_device)


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_classification_report(text: str) -> pd.DataFrame:
    """Parse sklearn classification report text into a tidy DataFrame."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("accuracy") or line.startswith("macro") or line.startswith("weighted"):
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 5:
            rows.append({
                "Class":     parts[0].replace("___", " / ").replace("_(", " ("),
                "Precision": float(parts[1]),
                "Recall":    float(parts[2]),
                "F1":        float(parts[3]),
                "Support":   int(parts[4]),
            })
    return pd.DataFrame(rows)


def history_to_df(history: dict) -> pd.DataFrame:
    """Convert training history dict to a plottable DataFrame."""
    n = len(history.get("train_loss", []))
    df = pd.DataFrame({
        "Epoch":      list(range(1, n + 1)),
        "Train Loss": history.get("train_loss", [None] * n),
        "Train Acc":  history.get("train_acc",  [None] * n),
        "Val Acc":    history.get("val_acc",    [None] * n),
        "Val F1":     history.get("val_f1",     [None] * n),
    })
    snap_epochs = history.get("target_snapshot_epoch", [])
    snap_acc    = history.get("target_snapshot_acc",   [])
    snap_f1     = history.get("target_snapshot_f1",    [])
    if snap_epochs:
        tgt = pd.DataFrame({
            "Epoch":      snap_epochs,
            "Target Acc": snap_acc,
            "Target F1":  snap_f1,
        })
        df = df.merge(tgt, on="Epoch", how="left")
    return df


def build_diagnostic_strip(
    display_image: np.ndarray,
    pc_maps: dict[str, np.ndarray] | None,
    attention_overlay: np.ndarray | None,
    predicted_label: str,
    confidence: float,
) -> bytes:
    """Composite figure modelled on the saved analysis_sample_*.png plots.

    Layout: Input | PC Magnitude | Phase Symmetry | Oriented Energy [| Cross-Attn]
    Returned as PNG bytes ready to feed into st.image().
    """
    panels: list[tuple[str, np.ndarray, str | None]] = [
        ("Input", display_image, None),
    ]
    if pc_maps:
        panels.append(("PC Magnitude",    pc_maps["pc_magnitude"],    "viridis"))
        panels.append(("Phase Symmetry",  pc_maps["phase_symmetry"],  "magma"))
        panels.append(("Oriented Energy", pc_maps["oriented_energy"], "inferno"))
    if attention_overlay is not None:
        panels.append(("Cross-Attn", attention_overlay, "jet"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        f"Prediction: {predicted_label.replace('___', ' / ')}    Confidence: {confidence:.1%}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    for ax, (title, img, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def domain_shift_cards(model_key: str):
    model = models[model_key]
    if model.results_dir is None:
        return
    summary = read_json(model.results_dir / "phasephyto_domain_shift.json")
    if summary is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source Accuracy",  f"{summary.get('source_acc', 0):.1%}")
    c2.metric("Target Accuracy",  f"{summary.get('target_acc', 0):.1%}",
              delta=f"{summary.get('delta_acc', 0):.1%}")
    c3.metric("Source Macro-F1",  f"{summary.get('source_f1', 0):.3f}")
    c4.metric("Target Macro-F1",  f"{summary.get('target_f1', 0):.3f}")
    st.caption(
        f"Checkpoint: **{summary.get('checkpoint', 'N/A')}** | "
        f"TTA flip: {summary.get('tta_hflip', False)} | "
        f"TENT: {summary.get('tent_enabled', False)} "
        f"({summary.get('tent_steps', 0)} steps)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## PhasePhyto")
    st.caption("Physics-informed fusion for plant disease OOD transfer.")
    st.divider()

    st.markdown("### Select Model")
    model_options = list(models.keys())
    model_display = [
        f"**{models[k].label}**\n{MODEL_DESCRIPTIONS.get(k, '')}"
        for k in model_options
    ]
    selected_idx = st.radio(
        "model_radio",
        options=range(len(model_options)),
        format_func=lambda i: models[model_options[i]].label,
        label_visibility="collapsed",
    )
    selected_key = model_options[selected_idx]
    selected_model = models[selected_key]

    st.caption(MODEL_DESCRIPTIONS.get(selected_key, ""))
    if not selected_model.can_infer:
        st.warning("Metrics only. No checkpoint is available for inference.")

    st.divider()
    st.caption(f"Device: `{device}`")
    st.caption(f"Inference-ready: **{len(inferable)}** models")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="pp-hero">
      <h1>PhasePhyto Explorer</h1>
      <div class="pp-subtitle">
        Compare PhasePhyto model variants on plant disease images.
        Inspect predictions, training curves, confusion matrices, and
        per class metrics for the PlantVillage to PlantDoc OOD benchmark.
      </div>
      <div style="margin-top:0.7rem;">
        <span class="pp-badge">ViT-B/16 backbone</span>
        <span class="pp-badge">Phase Congruency fusion</span>
        <span class="pp-badge">25 disease classes</span>
        <span class="pp-badge">OOD transfer study</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_infer, tab_results, tab_about = st.tabs(
    ["Try it: Inference", "Results and Metrics", "About"]
)


# Tab 1: Inference
with tab_infer:

    if not selected_model.can_infer:
        st.info(
            f"**{selected_model.label}** has no available checkpoint. "
            "Select a different model from the sidebar, or view its metrics in the Results tab."
        )
        st.stop()

    # ── Input mode ────────────────────────────────────────────────────────────
    input_mode = st.radio(
        "Image source",
        ["Sample images", "Upload your own"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()

    image: Image.Image | None = None
    chosen_sample = None

    if input_mode == "Sample images":
        curated = [s for s in samples if selected_key in s.known_good_models]
        if not curated:
            st.warning("No curated samples are registered for this model. Try uploading your own image.")
        else:
            st.markdown("**Choose a sample image.** Click one to select it.")
            thumb_cols = st.columns(min(len(curated), 4))
            selected_sample_key: str | None = st.session_state.get("selected_sample_key")

            for i, sample in enumerate(curated):
                with thumb_cols[i % len(thumb_cols)]:
                    thumb = Image.open(sample.image_path)
                    st.image(thumb, caption=sample.label.replace("___", "\n"), use_container_width=True)
                    if st.button(f"Select", key=f"sample_btn_{i}"):
                        st.session_state["selected_sample_key"] = sample.key

            if selected_sample_key:
                match = next((s for s in curated if s.key == selected_sample_key), None)
                if match:
                    chosen_sample = match
                    image = Image.open(match.image_path)

            if chosen_sample is None and curated:
                chosen_sample = curated[0]
                image = Image.open(chosen_sample.image_path)
                st.caption(f"Auto-selected: **{chosen_sample.label}**")

    else:
        uploaded = st.file_uploader(
            "Upload a leaf image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded:
            image = Image.open(uploaded)

    # ── Inference ─────────────────────────────────────────────────────────────
    if image is not None:
        st.divider()
        with st.spinner(f"Running {selected_model.label}…"):
            model_nn, spec = get_model(selected_key, str(artifact_root), device)
        result = predict_image(model_nn, spec, image, device=device)

        left_col, right_col = st.columns([0.85, 1.6], gap="large")

        with left_col:
            st.markdown("#### Input Image")
            st.image(image, use_container_width=True)
            if chosen_sample:
                if chosen_sample.label_index < 0:
                    st.caption(f"Source: **{chosen_sample.label}**")
                else:
                    st.caption(f"Expected: **{chosen_sample.label.replace('___', ' / ')}**")

            pred_short = result.predicted_label.replace("___", " / ")
            conf_pct = result.confidence * 100
            st.markdown(
                f"""
                <div class="pp-result-card">
                  <div class="pp-pred-label">{pred_short}</div>
                  <div class="pp-confidence">Confidence: {conf_pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if chosen_sample is not None and chosen_sample.label_index >= 0:
                if result.predicted_index == chosen_sample.label_index:
                    st.success("Correct prediction.")
                else:
                    st.warning(f"Mismatch. Expected: {chosen_sample.label.replace('___', ' / ')}")
            elif chosen_sample is not None and chosen_sample.label_index < 0:
                st.info("This sample is out of distribution for the 25 class label space (there is no Tomato class). The model maps it to the closest learned class.")

        with right_col:
            st.markdown("#### Diagnostic Strip")
            if result.pc_maps:
                st.caption(
                    "Live diagnostic for the selected sample. The panels are the input, "
                    "three Phase Congruency maps, and the cross attention overlay (Full model only)."
                )
                strip_png = build_diagnostic_strip(
                    display_image=result.display_image,
                    pc_maps=result.pc_maps,
                    attention_overlay=result.attention_overlay,
                    predicted_label=result.predicted_label,
                    confidence=result.confidence,
                )
                st.image(strip_png, use_container_width=True)
            else:
                st.info(
                    "Phase Congruency maps are only produced by the PhasePhyto variants "
                    "(Full / Full EMA / Backbone Only / No Fusion). The Baseline ViT does not "
                    "expose intermediate structural maps."
                )
                st.markdown(
                    f"**Ablation:** `{spec.ablation}`  \n"
                    f"**Model type:** `{spec.model_type}`  \n"
                    f"**Image size:** {spec.image_size}px"
                )

    else:
        st.info("Select a sample or upload an image above to run inference.")


# Tab 2: Results and Metrics
with tab_results:

    # Model selector for this tab (independent of sidebar)
    results_model_labels = [v.label for v in models.values() if v.results_dir is not None]
    if not results_model_labels:
        st.warning("No result bundles found in the artifact root.")
        st.stop()

    chosen_results_label = st.selectbox(
        "View results for:",
        results_model_labels,
        index=min(selected_idx, len(results_model_labels) - 1),
        key="results_model_select",
    )
    results_key = next(k for k, v in models.items() if v.label == chosen_results_label)
    results_model = models[results_key]

    # ── Metric cards ──────────────────────────────────────────────────────────
    st.markdown("### Domain Shift Summary")
    domain_shift_cards(results_key)
    st.divider()

    sub_training, sub_confusion, sub_gallery, sub_report = st.tabs(
        ["Training Curves", "Confusion Matrix", "All Plots", "Classification Report"]
    )

    # ── Training curves ───────────────────────────────────────────────────────
    with sub_training:
        history = read_json(
            results_model.results_dir / "history.json"
            if results_model.results_dir else None
        )
        training_png = (
            results_model.plots_dir / "training_curves.png"
            if results_model.plots_dir else None
        )

        if history:
            df_hist = history_to_df(history)
            st.markdown("#### Source domain (validation) performance")
            val_cols = [c for c in ["Val Acc", "Val F1", "Train Acc"] if c in df_hist.columns]
            st.line_chart(df_hist.set_index("Epoch")[val_cols], height=280)

            if "Target Acc" in df_hist.columns:
                st.markdown("#### Target domain snapshots (taken every 3 epochs, no TTA/TENT)")
                tgt_cols = [c for c in ["Target Acc", "Target F1"] if c in df_hist.columns]
                tgt_df = df_hist[["Epoch"] + tgt_cols].dropna()
                st.line_chart(tgt_df.set_index("Epoch")[tgt_cols], height=220)
                st.caption(
                    "Target snapshots are raw evaluations during training. "
                    "The final reported target accuracy (with TTA + TENT) is shown in the metric cards above."
                )
            st.dataframe(df_hist.round(4), use_container_width=True, hide_index=True)

        elif training_png and training_png.exists():
            st.image(str(training_png), caption="Training curves (saved plot)", use_container_width=True)
        else:
            st.info("No training history available for this model.")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    with sub_confusion:
        confusion_path = (
            results_model.plots_dir / "confusion_matrices.png"
            if results_model.plots_dir else None
        )
        if confusion_path and confusion_path.exists():
            st.image(str(confusion_path), caption="Source (left) and Target (right) confusion matrices",
                     use_container_width=True)
        else:
            st.info("No confusion matrix plot available for this model.")

    # ── All plots gallery ─────────────────────────────────────────────────────
    with sub_gallery:
        plot_paths = available_plot_paths(results_model)
        if not plot_paths:
            st.info("No plots found for this model.")
        else:
            PLOT_LABELS = {
                "training_curves.png":      "Training Curves",
                "confusion_matrices.png":   "Confusion Matrices",
                "illumination_invariance.png": "Illumination Invariance",
                "leaf_mask_sanity.png":     "Leaf Mask Sanity Check",
                "analysis_sample_0.png":    "Analysis Sample A",
                "analysis_sample_1.png":    "Analysis Sample B",
                "analysis_sample_2.png":    "Analysis Sample C",
            }
            items = list(plot_paths.items())
            for row_start in range(0, len(items), 2):
                row_items = items[row_start: row_start + 2]
                cols = st.columns(len(row_items), gap="medium")
                for col, (name, path) in zip(cols, row_items):
                    with col:
                        label = PLOT_LABELS.get(name, name.replace("_", " ").replace(".png", "").title())
                        st.image(str(path), use_container_width=True)
                        st.markdown(f'<p class="pp-plot-title">{label}</p>', unsafe_allow_html=True)
                st.write("")  # spacing

    # ── Classification report ─────────────────────────────────────────────────
    with sub_report:
        report_text = read_text(
            results_model.results_dir / "target_classification_report.txt"
            if results_model.results_dir else None
        )
        if report_text is None:
            st.info("No target classification report available for this model.")
        else:
            st.markdown("#### Per-class target (PlantDoc) metrics")
            df_report = parse_classification_report(report_text)
            if not df_report.empty:
                # Colour F1 column
                styled = (
                    df_report.style
                    .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1)
                    .background_gradient(subset=["Recall"], cmap="Blues", vmin=0, vmax=1)
                    .format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"})
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)
                avg_f1 = df_report[df_report["Support"] > 0]["F1"].mean()
                st.caption(f"Average F1 across classes with support > 0: **{avg_f1:.3f}**")
            st.markdown("##### Raw report")
            st.text(report_text)


# Tab 3: About
with tab_about:
    col_a, col_b = st.columns([1.2, 1], gap="large")

    with col_a:
        st.markdown("### What is PhasePhyto?")
        st.markdown(
            """
            **PhasePhyto** is a three stream image classification model and an
            OOD training recipe for plant disease detection under domain shift.

            We study the PlantVillage to PlantDoc benchmark, where models trained on
            **clean lab images** are evaluated on **cluttered field images**.

            **Three processing streams:**
            | Stream | Description |
            |--------|-------------|
            | Phase Congruency | Fixed log-Gabor filter bank that gives amplitude-invariant structural maps |
            | ViT-B/16 | Pretrained Vision Transformer semantic backbone (about 86M params) |
            | CLAHE | Contrast normalised illumination stream |

            The PC and ViT streams are fused through cross attention, where PC tokens query ViT tokens.

            **Key finding.** The OOD recipe (a 23.5 point gain over the plain baseline) is the
            primary contribution. The PC fusion stream matched but did not beat
            the backbone only variant, which isolates the recipe as the causal driver.
            """
        )

    with col_b:
        st.markdown("### Ablation Table")
        ablation_df = pd.DataFrame([
            {"Model": "Baseline ViT (no recipe)", "Src Acc": 0.9931, "Tgt Acc": 0.2876, "Tgt F1": 0.2119},
            {"Model": "Full PhasePhyto",          "Src Acc": 0.9975, "Tgt Acc": 0.5229, "Tgt F1": 0.3944},
            {"Model": "Backbone Only",            "Src Acc": 0.9975, "Tgt Acc": 0.5229, "Tgt F1": 0.3859},
            {"Model": "No Fusion",                "Src Acc": 0.9975, "Tgt Acc": 0.4902, "Tgt F1": 0.3464},
            {"Model": "PC Only",                  "Src Acc": 0.7340, "Tgt Acc": 0.0915, "Tgt F1": 0.0791},
        ])
        st.dataframe(
            ablation_df.style
            .background_gradient(subset=["Tgt Acc", "Tgt F1"], cmap="RdYlGn", vmin=0.1, vmax=0.6)
            .format({"Src Acc": "{:.1%}", "Tgt Acc": "{:.1%}", "Tgt F1": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### OOD Recipe")
        st.markdown(
            """
            - Label smoothing (ε = 0.1)
            - Differential LR (ViT 10× lower)
            - EMA weight averaging (decay 0.999)
            - SAM optimiser (ρ = 0.05)
            - RandAugment (N=2, M=9)
            - Random erasing (p = 0.25)
            - HSV foreground gating
            - Horizontal-flip TTA
            - TENT test-time entropy minimisation
            """
        )

    st.divider()
    st.markdown("### Model Availability")
    availability = pd.DataFrame([
        {
            "Model":      m.label,
            "Ablation":   m.ablation,
            "Inference":  "Yes" if m.can_infer else "No",
            "Has Plots":  "Yes" if m.plots_dir else "No",
            "Has Metrics":"Yes" if m.results_dir else "No",
        }
        for m in models.values()
    ])
    st.dataframe(availability, use_container_width=True, hide_index=True)

    st.markdown(
        """
        ---
        **Resources:** [HF Space](https://huggingface.co/spaces/Mathesh0803/phasephyto-explorer) ·
        [Model weights](https://huggingface.co/Mathesh0803/phasephyto-weights) ·
        DATA 640, University of Maryland College Park
        """
    )
