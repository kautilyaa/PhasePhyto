from __future__ import annotations

from pathlib import Path

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

st.set_page_config(
    page_title="PhasePhyto Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

artifact_root = resolve_artifact_root()
models = discover_model_artifacts(artifact_root)
samples = load_demo_samples()
device = choose_device()

st.markdown(
    """
    <style>
      .main > div {padding-top: 1.5rem;}
      .pp-hero {
        padding: 1.25rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f3d2e 0%, #105b72 55%, #1a2340 100%);
        color: white;
        margin-bottom: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
      }
      .pp-subtle {
        color: rgba(255,255,255,0.85);
        font-size: 0.98rem;
        line-height: 1.5;
      }
      .pp-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        margin: 0.15rem 0.35rem 0.15rem 0;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.14);
        font-size: 0.8rem;
      }
      .pp-card {
        border: 1px solid rgba(49, 51, 63, 0.12);
        border-radius: 16px;
        padding: 1rem 1rem 0.65rem 1rem;
        background: linear-gradient(180deg, rgba(241,247,245,0.95), rgba(255,255,255,0.98));
        margin-bottom: 1rem;
      }
      .pp-small {
        color: #5b6472;
        font-size: 0.9rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pp-hero">
      <h1 style="margin:0 0 0.4rem 0;">PhasePhyto Explorer</h1>
      <div class="pp-subtle">
        Interactively compare saved PhasePhyto variants and the baseline ViT on curated
        leaf samples, inspect predictions, and browse plots + result artifacts from the final project.
      </div>
      <div style="margin-top:0.85rem;">
        <span class="pp-badge">Streamlit demo</span>
        <span class="pp-badge">Docker-ready</span>
        <span class="pp-badge">Hugging Face Space-ready</span>
        <span class="pp-badge">Curated correct samples</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_loaded_model(model_key: str, root_str: str, device_name: str):
    del root_str
    return load_demo_model(models[model_key], device=device_name)


def metric_cards(model_key: str):
    model = models[model_key]
    if model.results_dir is None:
        st.info("No metrics bundle found for this model.")
        return
    summary = read_json(model.results_dir / "phasephyto_domain_shift.json")
    if summary is None:
        st.info("No summary JSON available for this model.")
        return
    cols = st.columns(4)
    cols[0].metric("Source accuracy", f"{summary.get('source_acc', 0):.3f}")
    cols[1].metric("Target accuracy", f"{summary.get('target_acc', 0):.3f}")
    cols[2].metric("Source macro-F1", f"{summary.get('source_f1', 0):.3f}")
    cols[3].metric("Target macro-F1", f"{summary.get('target_f1', 0):.3f}")
    st.caption(
        f"Checkpoint: {summary.get('checkpoint', 'saved checkpoint')} | "
        f"TTA: {summary.get('tta_hflip', False)} | "
        f"TENT: {summary.get('tent_enabled', False)}"
    )


def sample_options_for_model(model_key: str):
    return [sample for sample in samples if model_key in sample.known_good_models]


def show_probability_table(probabilities: np.ndarray):
    top_idx = np.argsort(probabilities)[::-1][:5]
    frame = pd.DataFrame(
        {
            "class": [PLANT_DISEASE_25_CLASSES[i] for i in top_idx],
            "probability": [float(probabilities[i]) for i in top_idx],
        }
    )
    st.dataframe(frame, width="stretch", hide_index=True)


def show_phasephyto_visuals(result):
    if not result.pc_maps:
        return
    cols = st.columns(4)
    cols[0].image(result.display_image, caption="Input", width="stretch")
    cols[1].image(result.pc_maps["pc_magnitude"], caption="PC Magnitude", width="stretch")
    cols[2].image(result.pc_maps["phase_symmetry"], caption="Phase Symmetry", width="stretch")
    cols[3].image(result.pc_maps["oriented_energy"], caption="Oriented Energy", width="stretch")
    if result.attention_overlay is not None:
        st.image(result.attention_overlay, caption="Cross-attention overlay", width="stretch", clamp=True)


inferable_models = [m for m in models.values() if m.can_infer]
st.sidebar.header("Demo status")
st.sidebar.caption(f"Artifacts root: `{artifact_root}`")
st.sidebar.caption(f"Device: `{device}`")
st.sidebar.metric("Inference-ready models", len(inferable_models))
st.sidebar.metric("Curated samples", len(samples))
metrics_only = [m.label for m in models.values() if not m.can_infer and m.results_dir is not None]
if metrics_only:
    st.sidebar.write("Metrics-only models:")
    for label in metrics_only:
        st.sidebar.write(f"- {label}")


tab_demo, tab_metrics, tab_about = st.tabs(["Demo", "Metrics & Plots", "About / Deploy"])

with tab_demo:
    if not inferable_models:
        st.error(
            "No checkpoints found. Set PHASEPHYTO_ASSETS_ROOT to a directory that contains the final project artifacts."
        )
    else:
        hero_cols = st.columns([1, 1, 1])
        hero_cols[0].metric("Models available", len(inferable_models))
        hero_cols[1].metric("Curated good samples", len(samples))
        hero_cols[2].metric("Metrics bundles", sum(1 for m in models.values() if m.results_dir is not None))

        model_labels = {m.label: m.key for m in inferable_models}

        controls_col, preview_col = st.columns([0.95, 1.55], gap="large")
        with controls_col:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            selected_model_label = st.selectbox("Model", list(model_labels.keys()))
            selected_model_key = model_labels[selected_model_label]
            model_artifact = models[selected_model_key]

            curated = sample_options_for_model(selected_model_key)
            curated_names = [f"{s.label} — {s.key}" for s in curated]
            sample_choice = st.selectbox(
                "Known-good sample for this model",
                ["None"] + curated_names,
                help="Only samples with known-correct predictions for the selected model are shown here.",
            )
            uploaded = st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg"])

            st.markdown(
                f"""
                <div class="pp-small">
                  <strong>Selected model:</strong> {model_artifact.label}<br/>
                  <strong>Ablation:</strong> {model_artifact.ablation}<br/>
                  <strong>Checkpoint source:</strong> {"available" if model_artifact.can_infer else "missing"}
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with preview_col:
            st.markdown('<div class="pp-card">', unsafe_allow_html=True)
            st.subheader("Inference preview")
            image: Image.Image | None = None
            source_caption = None
            chosen_sample = None
            if uploaded is not None:
                image = Image.open(uploaded)
                source_caption = "Uploaded image"
            elif sample_choice != "None":
                chosen_sample = curated[curated_names.index(sample_choice)]
                image = Image.open(chosen_sample.image_path)
                source_caption = f"Curated sample — expected: {chosen_sample.label}"

            if image is not None:
                with st.spinner(f"Loading {model_artifact.label}..."):
                    model, spec = get_loaded_model(selected_model_key, str(artifact_root), device)
                result = predict_image(model, spec, image, device=device)

                left, right = st.columns([1.05, 1], gap="large")
                with left:
                    st.image(image, caption=source_caption, width="stretch")
                with right:
                    st.subheader(result.predicted_label)
                    st.metric("Confidence", f"{result.confidence:.1%}")
                    if chosen_sample is not None:
                        verdict = "Correct" if result.predicted_index == chosen_sample.label_index else "Mismatch"
                        st.success(f"{verdict} vs expected label") if verdict == "Correct" else st.warning(verdict)
                        st.caption(f"Expected label: {chosen_sample.label}")
                    show_probability_table(result.probabilities)
                    st.caption(
                        f"Checkpoint type: {result.checkpoint_spec.model_type} | "
                        f"Ablation: {result.checkpoint_spec.ablation}"
                    )

                if result.checkpoint_spec.model_type == "phasephyto":
                    st.subheader("Model diagnostics")
                    show_phasephyto_visuals(result)
            else:
                st.info("Choose a curated sample or upload an image to run inference.")
            st.markdown("</div>", unsafe_allow_html=True)

with tab_metrics:
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    model_choice = st.selectbox("Results bundle", [m.label for m in models.values()], key="metrics_model")
    model_key = next(k for k, v in models.items() if v.label == model_choice)
    model_artifact = models[model_key]
    metric_cards(model_key)

    report_text = read_text(model_artifact.results_dir / "target_classification_report.txt" if model_artifact.results_dir else None)
    history = read_json(model_artifact.results_dir / "history.json" if model_artifact.results_dir else None)
    plot_paths = available_plot_paths(model_artifact)

    view_mode = st.selectbox(
        "What to view",
        ["Plots", "Classification report", "Training history JSON"],
        key="metrics_view",
    )
    if view_mode == "Plots":
        if not plot_paths:
            st.info("No plots available for this model.")
        else:
            plot_name = st.selectbox("Plot", list(plot_paths.keys()))
            st.image(str(plot_paths[plot_name]), caption=plot_name, width="stretch")
    elif view_mode == "Classification report":
        if report_text is None:
            st.info("No classification report available.")
        else:
            st.text(report_text)
    else:
        if history is None:
            st.info("No history.json available.")
        else:
            st.json(history)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_about:
    st.markdown(
        """
        ### What this app expects
        - A final-project artifact root containing folders like `Full`, `Backbone only`, `No_fusion`, and `baseline_vit.pt`.
        - For local Docker, mount your existing `Final Project/Finalresults` folder into the container and point `PHASEPHYTO_ASSETS_ROOT` at it.
        - For Hugging Face Spaces, this app can auto-download public checkpoints from the default public weights repo `Mathesh0803/phasephyto-weights`. You can still override that with `PHASEPHYTO_HF_MODEL_REPO`.

        ### Current model availability
        - **Inference-capable if checkpoint exists**: Full, Full EMA, Backbone Only, No Fusion, Baseline ViT
        - **Metrics-only by default**: PC Only (no checkpoint was found in the provided `Final Project` folder)
        """
    )
    availability = pd.DataFrame(
        [
            {
                "model": model.label,
                "can_infer": model.can_infer,
                "checkpoint": str(model.checkpoint_path) if model.checkpoint_path else "missing",
                "plots_dir": str(model.plots_dir) if model.plots_dir else "-",
            }
            for model in models.values()
        ]
    )
    st.dataframe(availability, width="stretch", hide_index=True)
