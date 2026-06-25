import pandas as pd
import streamlit as st
from pathlib import Path
from tempfile import NamedTemporaryFile

from main import predict_controller, train_controller


st.set_page_config(
	page_title="Microbiome Prediction UI",
	page_icon="🧬",
	layout="wide",
)


st.title("Microbiome Prediction App")
st.caption("Upload an Excel file to train the model, then enter a participant profile for prediction.")


st.subheader("1. Training data")
uploaded_excel = st.file_uploader(
	"Upload Excel file for training",
	type=["xlsx", "xls"],
	help="Upload the Excel file that contains the training data."
)

train_clicked = st.button("Train model", type="primary")

if train_clicked:
	if uploaded_excel is None:
		st.error("Please upload an Excel file before training.")
	else:
		file_suffix = Path(uploaded_excel.name).suffix or ".xlsx"
		with NamedTemporaryFile(delete=False, suffix=file_suffix) as temp_file:
			temp_file.write(uploaded_excel.getvalue())
			temp_path = temp_file.name

		with st.spinner("Training model..."):
			try:
				overall_r2, rmse = train_controller(temp_path)
				st.success("Training completed successfully.")
				st.write(f"Overall R2: {overall_r2:.4f}")
				st.write(f"RMSE: {rmse:.4f}")
			except Exception as e:
				st.error(f"Training failed: {e}")


with st.sidebar:
	st.header("Input guide")
	st.write("Fill in the fields below and submit the form.")
	st.write("The app will build a dataframe that matches the model input structure.")


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

	submitted = st.form_submit_button("Create input row")


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

	st.success("Input row created successfully.")
	st.subheader("Submitted data")
	st.dataframe(user_row, use_container_width=True)	
	st.subheader("Preview")
	st.json(user_row.iloc[0].to_dict())

	try:
		predicted_value = predict_controller(user_row)
		st.subheader("Predicted value")
		st.write(predicted_value)
	except Exception as e:
		st.warning(f"Prediction is not available yet: {e}")

