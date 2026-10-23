import pandas as pd
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from main import (DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, CUSTOM_MODEL_DIR,
                  TrainingConfig, list_models, model_info, predict_controller, train_controller,
                  MICROBIOME_DATA_PATH, DEFAULT_MICROBIOME_MODEL_DIR, CUSTOM_MICROBIOME_MODEL_DIR,
                  MicrobiomeConfig, train_microbiome_controller, predict_microbiome_values_controller)
from runtime import TRAINING_DEVICE, DEVICE_NAME


st.set_page_config(
    page_title="Metabolite & Microbiome Prediction",
    page_icon="🧬",
    layout="wide",
)


# Keep widget help beside the label instead of stretching to the right edge.
st.html("""
<style>
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] > div,
[data-testid="stNumberInput"] [data-testid="stWidgetLabel"] > div {
    flex: 0 0 auto;
    justify-content: flex-start;
}
[data-testid="stNumberInput"] button {
    touch-action: none;
    user-select: none;
}
.prediction-scroll {
    max-height: 400px;
    overflow: auto;
    scrollbar-gutter: stable;
    padding-right: 16px;
    border: 1px solid var(--st-border-color, #8886);
    border-radius: 8px;
}
.prediction-table {
    width: 100%;
    border-collapse: collapse;
}
.prediction-table th, .prediction-table td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid var(--st-border-color, #8886);
}
.prediction-table th {
    position: sticky;
    top: 0;
    background: var(--st-background-color, white);
}
.prediction-table th:last-child, .prediction-table td:last-child {
    width: 120px;
    border-left: 1px solid var(--st-border-color, #8886);
}
</style>
""")

st.html(Path(__file__).parent / "static/number_input_hold.html", unsafe_allow_javascript=True)

st.title("Metabolite & Microbiome Prediction")
st.caption("One participant form, two saved models: binary metabolites, genus percentages, and bounded numeric abundances.")
if TRAINING_DEVICE.startswith("/GPU"):
    st.success(f"Metabolite training device: {DEVICE_NAME}")
else:
    st.caption("Metabolite training: CPU. Microbiome regression also runs on CPU.")

st.subheader("1. Choose models")
model_choice = st.radio("Model source", ["Default models", "Uploaded-data models"], horizontal=True)
custom = model_choice == "Uploaded-data models"
tasks = {
    "metabolite": {"label": "Metabolite", "path": DEFAULT_DATA_PATH,
                   "directory": CUSTOM_MODEL_DIR if custom else DEFAULT_MODEL_DIR,
                   "train": train_controller},
    "microbiome": {"label": "Microbiome", "path": MICROBIOME_DATA_PATH,
                   "directory": CUSTOM_MICROBIOME_MODEL_DIR if custom else DEFAULT_MICROBIOME_MODEL_DIR,
                   "train": train_microbiome_controller},
}
infos = {task: model_info(details["directory"]) for task, details in tasks.items()}
validation_labels = {"5-fold cross-validation": "kfold", "Fast holdout (20%)": "holdout",
                     "Leave-one-out (slow)": "loo", "Train only (no evaluation)": "none"}
with st.expander("Build or update saved models", expanded=not all(infos.values())):
    st.caption("Excel layout: column A = unique participant ID; B–K = the ten participant features; L onward = targets. Each row must be one independent participant. The two files may have different participants, but must use the same feature names and units.")
    uploads = {}
    for task, details in tasks.items():
        if custom:
            uploads[task] = st.file_uploader(f"Upload {details['label']} training data", type=["xlsx", "xls"], key=f"upload_{task}")
        else:
            st.caption(f"{details['label']} source: {details['path'].name}")
    metabolite_validation = st.selectbox("Metabolite evaluation method", list(validation_labels), index=1,
        help="Holdout evaluates 20% of rows. Five-fold evaluates every row once. Leave-one-out fits once per held-out participant. Train only provides no quality estimate. Final models always use all rows.")
    epochs = st.number_input("Metabolite maximum epochs per model", min_value=1, max_value=5000, value=500,
        help="Maximum training passes for each neural network; early stopping may finish sooner.")
    batch_size = st.number_input("Metabolite batch size", min_value=1, max_value=4096, value=32,
        help="Training rows processed in each neural-network update.")
    microbiome_validation = st.selectbox("Microbiome evaluation method", list(validation_labels),
        help="Recommended: five-fold outer validation. A separate inner cross-validation selects the regression and regularization using only training rows. Train only still selects the model internally but provides no independent evaluation.")
    st.caption("Microbiome outputs include percentages totaling 100% and a separately trained numeric estimate in the file’s original units. Numeric predictions are bounded by each genus’s observed training minimum and maximum. Both predictors compare regularized regression against a population baseline. No epochs or batch size are needed.")
    force = st.checkbox("Force retraining even if the data and settings are unchanged")
    build_both = st.button("Build / reuse both models", type="primary")
    build_metabolite = st.button("Build / reuse Metabolite model")
    build_microbiome = st.button("Build / reuse Microbiome model")
    requested = list(tasks) if build_both else (["metabolite"] if build_metabolite else (["microbiome"] if build_microbiome else []))
    configs = {
        "metabolite": TrainingConfig(epochs=int(epochs), batch_size=int(batch_size), validation=validation_labels[metabolite_validation]),
        "microbiome": MicrobiomeConfig(validation=validation_labels[microbiome_validation], output_mode="relative"),
    }
    if requested and custom and any(uploads[task] is None for task in requested):
        st.error("Upload a training file for each model being built. Use an individual build button to update only one model.")
    else:
        for task in requested:
            details = tasks[task]
            temp_path = None
            status = st.empty()
            try:
                data_path = details["path"]
                if custom:
                    upload = uploads[task]
                    with NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as temp_file:
                        temp_file.write(upload.getvalue())
                        temp_path = Path(temp_file.name)
                    data_path = temp_path
                with st.spinner(f"Preparing {details['label']} model..."):
                    infos[task] = details["train"](data_path, model_dir=details["directory"],
                        config=configs[task], force=force, progress=status.write)
                st.success(f"{details['label']} model is ready.")
            except Exception as exc:
                st.error(f"{details['label']} training failed: {exc}")
            finally:
                status.empty()
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

selected_versions = {}
model_descriptions = {}
for task, details in tasks.items():
    st.subheader(f"{details['label']} model")
    builds = list_models(details["directory"])
    selected_versions[task] = None
    model_descriptions[task] = f"{model_choice} — {details['label']}"
    if builds:
        by_version = {build["version"]: build["info"] for build in builds}
        def build_description(version, saved_builds=by_version):
            info = saved_builds[version]
            return f"{info['trained_at']} | {info.get('config', {}).get('validation', 'unknown')} | {version}"
        version = st.selectbox(f"{details['label']} saved model build", list(by_version), format_func=build_description,
            key=f"saved_{task}_{model_choice}_{builds[0]['version']}",
            help="All successful builds are retained. Selecting an older build does not change the latest saved model.")
        selected_versions[task] = version
        infos[task] = by_version[version]
        model_descriptions[task] += f" — {build_description(version)}"
    info = infos[task]
    if info is None:
        st.info(f"Build a {details['label']} model before predicting both outputs.")
        continue
    st.caption(f"Saved model ready: {info['samples']} participants, {info['targets']} outputs.")
    metrics = info["metrics"]
    if metrics.get("rmse") is not None:
        r2 = f"R² {metrics['r2']:.4f}, " if metrics.get("r2") is not None else ""
        units = " percentage points" if task == "microbiome" else ""
        st.caption(f"{metrics['validation']} evaluation: {r2}RMSE {metrics['rmse']:.4f}{units}; {metrics['evaluation_samples']} held-out participants.")
    else:
        st.caption("No independent evaluation is available for this build.")
    if task == "microbiome":
        method = info.get("history", {}).get("selected_method", "unknown")
        st.caption(f"Selected percentage model: {method}. Outputs: relative abundance (%).")
        numeric_metrics = metrics.get("numeric_values")
        if numeric_metrics is not None:
            numeric_method = info.get("history", {}).get("numeric_values", {}).get("selected_method", "unknown")
            st.caption(f"Selected numeric-value model: {numeric_method}. Predictions are bounded by each genus’s training range.")
            if numeric_method == "mean":
                st.warning("Numeric model selection chose a constant population estimate; numeric values do not vary by participant for this build.")
            if numeric_metrics.get("rmse") is not None:
                st.caption(f"Numeric-value evaluation: RMSE {numeric_metrics['rmse']:.4f}, MAE {numeric_metrics['mae']:.4f} in original file units.")
            numeric_skill = numeric_metrics.get("skill_vs_baseline")
            if numeric_skill is not None:
                st.caption(f"Numeric improvement over baseline: {numeric_skill:.1%} on {numeric_metrics.get('loss_name', 'log1p MSE')}.")
            if numeric_metrics.get("per_genus"):
                with st.expander("Microbiome accuracy by genus"):
                    st.caption("Held-out evaluation in original file units. Lower RMSE/MAE is better. Negative R² or a negative MSE improvement means poor performance against the corresponding baseline. These are validation results, not training-row matches.")
                    report = pd.DataFrame(numeric_metrics["per_genus"]).sort_values("RMSE", ascending=False)
                    st.dataframe(report, hide_index=True, width="stretch")
                    st.download_button("Download microbiome validation report", report.to_csv(index=False),
                                       "microbiome_validation.csv", "text/csv", on_click="ignore", key="microbiome_validation_download")
        else:
            st.info("This older build provides percentages only. Rebuild the Microbiome model to add numeric predictions and training ranges.")
        skill = metrics.get("skill_vs_baseline")
        if skill is not None:
            st.caption(f"Improvement over the population-average baseline: {skill:.1%} on {metrics['loss_name']} (higher is better).")
        if method == "mean":
            st.warning("Cross-validation selected the population-average profile. This build returns the same profile for every participant because the tested regressions did not improve the selection score.")
        if metrics.get("r2") is not None and metrics["r2"] < 0:
            st.warning("Average per-genus R² is negative. Individual-genus predictions remain weak even when the whole-profile score improves.")
        if skill is not None and skill <= 0:
            st.warning("Held-out evaluation did not outperform the population-average baseline. Individual microbiome estimates are not established by these data.")
        st.caption("With few participants and many genera, these are exploratory estimates. A zero estimate does not establish absence. More independent training participants are needed to assess reliability.")

ready = all(infos.values())
with st.sidebar:
    st.header("Input guide")
    st.write("Build or select a saved Metabolite model and a saved Microbiome model from the same source category. Enter the participant once, then click Predict for both outputs.")
    st.write("Use the same measurement units and dietary intake period as the training files. MUAC means mid-upper arm circumference. Name identifies the row and is not a model input.")
    st.write("Metabolite outputs are binary indicators. Microbiome percentages total 100%. Numeric predictions use the original file’s units and are restricted to each genus’s observed range. These separately trained outputs need not imply the same profile; neither measures absolute bacterial load.")
    st.write("Each output has its own CSV download. The last successful pair remains available until a new pair is predicted.")

with st.form("user_profile_form"):
    st.subheader("Participant details")

    name = st.text_input("Name", placeholder="Enter name")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    with col2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    with col3:
        height_cm = st.number_input("Height (cm)", min_value=0.0, value=170.0, step=0.1)

    col4, col5, col6 = st.columns(3)
    with col4:
        weight_kg = st.number_input("Weight (Kgs)", min_value=0.0, value=70.0, step=0.1)
    with col5:
        bmi = st.number_input("BMI", min_value=0.0, value=24.2, step=0.1)
    with col6:
        body_fat_percentage = st.number_input("Body fat percentage", min_value=0.0, value=20.0, step=0.1)

    col7, col8, col9 = st.columns(3)
    with col7:
        muac_cm = st.number_input("MUAC (cm)", min_value=0.0, value=27.0, step=0.1)
    with col8:
        protein_grams = st.number_input("Protein (grams)", min_value=0.0, value=75.0, step=0.1)
    with col9:
        carbohydrates_grams = st.number_input("Carbohydrates  (grams)", min_value=0.0, value=200.0, step=0.1)

    fats_grams = st.number_input("Fats (grams)", min_value=0.0, value=60.0, step=0.1)

    submitted = st.form_submit_button("Predict", disabled=not ready)


if submitted:
    user_row = pd.DataFrame(
        [
            {
                "Name": name,
                "Age": age,
                "Sex": sex,
                "Height (cm)": height_cm,
                "Weight (Kgs)": weight_kg,
                "BMI": bmi,
                "Body fat percentage": body_fat_percentage,
                "MUAC (cm)": muac_cm,
                "Protein (grams)": protein_grams,
                "Carbohydrates  (grams)": carbohydrates_grams,
                "Fats (grams)": fats_grams,
            }
        ]
    )

    try:
        outputs = {}
        for task, details in tasks.items():
            predicted = predict_controller(user_row, model_dir=details["directory"], version=selected_versions[task])
            result = predicted.T.reset_index()
            result.columns = ["Target" if task == "metabolite" else "Genus",
                              "Prediction" if task == "metabolite" else "Relative abundance (%)"]
            if task == "microbiome" and infos[task].get("target_ranges"):
                numeric = predict_microbiome_values_controller(user_row, model_dir=details["directory"], version=selected_versions[task])
                result["Predicted value"] = numeric.loc[user_row.index[0], result["Genus"]].to_numpy()
            outputs[task] = {"results": result, "model": model_descriptions[task]}
        # Publish together so a failure cannot mix participants or previous results.
        ranges = {("Weight (Kgs)" if name == "Weight (cm)" else name): bounds
                  for name, bounds in infos["microbiome"].get("feature_ranges", {}).items()}
        outside = [name for name, (low, high) in ranges.items()
                   if name != "Sex" and name in user_row and not low <= float(user_row.iloc[0][name]) <= high]
        st.session_state["last_dual_prediction"] = {"input": user_row.copy(), "outputs": outputs, "outside": outside}
    except Exception as exc:
        st.error(f"Could not produce both predictions: {exc}. Any results below are the previous successful pair.")

last_prediction = st.session_state.get("last_dual_prediction")
if last_prediction is not None:
    st.subheader("Last successful predictions")
    st.subheader("Submitted data")
    st.dataframe(last_prediction["input"].rename(columns={"Weight (cm)": "Weight (Kgs)"}), width="stretch")
    if last_prediction.get("outside"):
        st.warning("Inputs outside the microbiome training range: " + ", ".join(last_prediction["outside"]) + ". These estimates extrapolate beyond the observed participants.")
    for task, output in last_prediction["outputs"].items():
        label = tasks[task]["label"]
        results = output["results"]
        if task == "microbiome":
            # Also hide range columns in results retained from an earlier session run.
            results = results.drop(columns=["Training minimum", "Training maximum"], errors="ignore")
        st.subheader(f"{label} prediction")
        st.caption(f"Model used: {output['model']}")
        if task == "metabolite":
            st.caption("Binary predictions: 1 = above the dataset-wide mean threshold; 0 = at or below it.")
        else:
            st.caption("Relative abundance (%) totals 100% before rounding. CSV preserves full precision.")
            if "Predicted value" in output["results"]:
                st.caption("Numeric values are separately predicted in original file units and constrained to the ranges observed during training.")
        st.html(
            f'<div class="prediction-scroll" tabindex="0" role="region" aria-label="{label} predictions">'
            + results.to_html(index=False, border=0, classes="prediction-table", escape=True,
                                        float_format=lambda value: f"{value:.6g}")
            + "</div>"
        )
        st.download_button(f"Download {label} predictions", results.to_csv(index=False),
                           f"{task}_predictions.csv", "text/csv", on_click="ignore", key=f"download_{task}")
