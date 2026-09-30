import pandas as pd
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from main import (DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, CUSTOM_MODEL_DIR,
                  TrainingConfig, list_models, model_info, predict_controller, train_controller)
from runtime import TRAINING_DEVICE, DEVICE_NAME


st.set_page_config(
	page_title="Microbiome Prediction",
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
</style>
""")

st.title("Microbiome Prediction")
st.caption("Build the default model once, then reuse it for predictions, or train a model from your own Excel file.")


if TRAINING_DEVICE.startswith("/GPU"):
	st.success(f"Training device: {DEVICE_NAME}")
else:
	st.warning("Training device: CPU. For NVIDIA GPU support, install requirements-gpu.txt and restart the app.")

st.subheader("1. Choose a model")
model_choice = st.radio("Prediction model", ["Default GCMS model", "Uploaded-data model"], horizontal=True)
selected_model_dir = DEFAULT_MODEL_DIR if model_choice == "Default GCMS model" else CUSTOM_MODEL_DIR
info = model_info(selected_model_dir)
if info:
    st.info(f"Saved model ready: {info['samples']} training samples, {info['targets']} outputs. No retraining needed for prediction.")
else:
    if model_choice == "Default GCMS model":
        st.info("First-time setup: click Build default model below to train using the bundled GCMS dataset. No upload is needed.")
    else:
        st.info("Upload your training dataset and build a model below before predicting.")

with st.expander("Build or update a saved model", expanded=info is None):
    st.caption("Default dataset: GCMS_Metabolite_Data_Combined.xlsx")
    uploaded_excel = None
    if model_choice == "Uploaded-data model":
        uploaded_excel = st.file_uploader("Upload Excel file for training", type=["xlsx", "xls"])
    validation_labels = {
        "Fast holdout (20%)": "holdout",
        "5-fold cross-validation": "kfold",
        "Leave-one-out (slow)": "loo",
        "Train only (no evaluation)": "none",
    }
    validation_label = st.selectbox("Evaluation method", list(validation_labels), help=(
        "**Fast holdout (20%):** Trains on roughly 80% of the rows and evaluates on the remaining "
        "20% (at least two rows). A quick check using a fixed, reproducible split.\n\n"
        "**5-fold cross-validation:** Repeats training on different groups and evaluates each row "
        "once. Gives a broader assessment but takes longer; small datasets use fewer folds "
        "and at least six rows are required.\n\n"
        "**Leave-one-out:** Holds out one participant at a time and trains on all others. "
        "Evaluates every participant, but is the slowest option.\n\n"
        "**Train only:** Skips evaluation and builds the saved model directly. "
        "No R² or RMSE scores are available.\n\n"
        "After evaluation, every method fits the saved model on all training rows."
    ))
    st.caption("Holdout metrics use a fixed test split. Cross-validation metrics use multiple test splits and are not directly comparable.")
    epochs = st.number_input("Maximum epochs per model", min_value=1, max_value=5000, value=500,
                             help="An epoch is one pass through the training rows. This caps training for each network in every fit. Training may stop early after 25 epochs without sufficient loss improvement; more epochs can take longer.")
    batch_size = st.number_input("Batch size", min_value=1, max_value=4096, value=32,
                                 help="Number of training rows processed per weight update. Smaller batches make more updates per epoch; larger batches use more memory. If the batch exceeds the training set, all rows form one batch.")
    force = st.checkbox("Force retraining even if the data and settings are unchanged")
    build_label = "Build default model" if info is None and model_choice == "Default GCMS model" else "Build / reuse selected model"
    train_clicked = st.button(build_label, type="primary")
    if train_clicked:
        if model_choice == "Uploaded-data model" and uploaded_excel is None:
            st.error("Please upload an Excel file before training.")
        else:
            temp_path = None
            try:
                data_path = DEFAULT_DATA_PATH
                if uploaded_excel is not None:
                    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_excel.name).suffix) as temp_file:
                        temp_file.write(uploaded_excel.getvalue())
                        temp_path = Path(temp_file.name)
                    data_path = temp_path
                status = st.empty()
                with st.spinner("Preparing saved model..."):
                    info = train_controller(
                        data_path, model_dir=selected_model_dir,
                        config=TrainingConfig(epochs=int(epochs), batch_size=int(batch_size),
                                              validation=validation_labels[validation_label]),
                        force=force, progress=status.write,
                    )
                status.empty()
                st.success("Saved model is ready for prediction.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

selected_version = None
model_description = model_choice
saved_models = list_models(selected_model_dir)
if saved_models:
    builds = {build["version"]: build["info"] for build in saved_models}

    def build_description(version):
        details = builds[version]
        settings = details.get("config", {})
        return (f"{details['trained_at']} | {settings.get('validation', 'unknown')} | "
                f"epochs {settings.get('epochs', '?')} | batch {settings.get('batch_size', '?')} | {version}")

    selected_version = st.selectbox(
        "Saved model build", list(builds), format_func=build_description,
        key=f"saved_build_{model_choice}_{saved_models[0]['version']}",
        help="Choose any saved build for prediction. All successful builds are kept, including the last five and older builds. Newest builds appear first. Reusing an unchanged model does not create another build.",
    )
    info = builds[selected_version]
    model_description = f"{model_choice} — {build_description(selected_version)}"
    st.caption(f"{len(saved_models)} saved builds available. Selected build: {info['samples']} training samples, {info['targets']} outputs.")

if info:
    metrics = info["metrics"]
    if metrics["r2"] is not None:
        st.caption(f"{metrics['validation']} evaluation: R² {metrics['r2']:.4f}, RMSE {metrics['rmse']:.4f} "
                   f"on {metrics['evaluation_samples']} samples. Original training: {info['training_seconds']:.1f}s.")
    else:
        st.caption("This model was trained without an evaluation split.")


with st.sidebar:
	st.header("Input guide")
	st.write("Choose the default GCMS model or a model trained from your uploaded dataset. For first-time use, build a model using the training controls. You can then select a saved build for prediction.")
	st.write("Enter the participant’s age, sex, body measurements, and protein, carbohydrate, and fat intake. MUAC means mid-upper arm circumference. Use the units and intake period used in your training dataset; replace the example values with the participant’s measurements.")
	st.write("Name identifies the submitted row and is not used to calculate predictions. Check your entries, then click Predict to view the submitted data and target outputs.")
	st.write("Outputs are binary: 1 means above the training dataset’s global target mean; 0 means at or below it. Download predictions as CSV. The last successful result remains visible in this session, even if you cancel the download or change the selected model; click Predict to generate a new result.")


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
		weight_cm = st.number_input("Weight (cm)", min_value=0.0, value=70.0, step=0.1)
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

	submitted = st.form_submit_button("Predict", disabled=info is None)


if submitted:
	user_row = pd.DataFrame(
		[
			{
				"Name": name,
				"Age": age,
				"Sex": sex,
				"Height (cm)": height_cm,
				"Weight (cm)": weight_cm,
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
		predicted_value = predict_controller(user_row, model_dir=selected_model_dir, version=selected_version)
		results = predicted_value.T.reset_index()
		results.columns = ["Target", "Prediction"]
		st.session_state["last_prediction"] = {
			"input": user_row.copy(), "results": results.copy(), "model": model_description,
		}
	except Exception as e:
		st.warning(f"Prediction is not available yet: {e}")

# Render independently of form submission so widget reruns and cancelled
# downloads retain the last successful prediction and its original inputs.
last_prediction = st.session_state.get("last_prediction")
if last_prediction is not None:
	st.subheader("Last successful prediction")
	st.caption(f"Model used: {last_prediction['model']}")
	st.subheader("Submitted data")
	st.dataframe(last_prediction["input"], use_container_width=True)
	st.subheader("Predicted outputs")
	st.caption("Binary predictions: 1 = above the dataset-wide mean threshold; 0 = at or below it.")
	st.dataframe(last_prediction["results"], use_container_width=True, hide_index=True)
	st.download_button("Download predictions", last_prediction["results"].to_csv(index=False),
	                   "predictions.csv", "text/csv", on_click="ignore")
