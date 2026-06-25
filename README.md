# Microbiome Prediction App (Streamlit UI)

This project provides a Streamlit UI to train a microbiome prediction model using an uploaded Excel file.

## Project Structure

- `app/ui.py` - Streamlit UI app
- `app/loadandpreprocess.py` - data loading and preprocessing
- `app/encoder.py` - autoencoder and decoder model builder
- `app/predictor.py` - predictor model builder
- `app/results/` - exported training results (Excel)
- `app/temp_uploads/` - temporary uploaded files from UI

## Prerequisites

- Python 3.9+ (recommended)
- pip

## 1) Setup Environment

From project root:

### Option A: Using uv (recommended)

If you have `uv` installed, create and activate a virtual environment with:

```bash
uv venv
```

### On Windows (cmd)

```bash
.venv\Scripts\activate
```

### On PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies with uv

```bash
uv pip install streamlit tensorflow numpy pandas scikit-learn openpyxl
```

### Option B: Using python venv

```bash
python -m venv .venv
```

### On Windows (cmd)

```bash
.venv\Scripts\activate
```

### On PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

## 2) Install Dependencies

Install required packages:

```bash
pip install streamlit tensorflow numpy pandas scikit-learn openpyxl
```

> `openpyxl` is required for reading/writing `.xlsx` files.

## 3) Run the App

From project root:

```bash
streamlit run app/ui.py --server.port 10880
```

Then open the local URL shown in terminal (usually `http://localhost:10880`).

## 4) How to Use the UI

1. In **Mode**, keep **Train Model** selected.
2. Click **Upload Excel File** and choose your `.xlsx`/`.xls` file.
3. Set training parameters:
   - **Latent Dimension** (default: `10`)
   - **Epochs** (default: `500`)
   - **Batch Size** (default: `8`)
4. Click **Start Training**.
5. Watch progress/logs in the UI.
6. On completion:
   - You will see **Overall R2** and **RMSE**
   - Top 20 genera by R2 are displayed
   - Results are saved to `app/results/` with timestamped filename.

## Output Files

Each run creates a timestamped file:

- `app/results/Microbiome_Prediction_Results_YYYYMMDD_HHMMSS.xlsx`

Workbook sheets:

- `Metrics` - per-genus R2 and RMSE
- `Predictions` - predicted values for all output genera

## Notes

- If you get `Import ... could not be resolved` for TensorFlow/Streamlit in your editor, verify the active Python environment and install packages in that same environment.
- Training can take time due to Leave-One-Out training loops.

## Troubleshooting

### Streamlit not found

```bash
pip install streamlit
```

### TensorFlow not found

```bash
pip install tensorflow
```

### Excel read errors

- Ensure file format is `.xlsx` or `.xls`
- Re-save the Excel file and retry if it is corrupted
- Ensure required columns are present in expected positions for preprocessing logic
