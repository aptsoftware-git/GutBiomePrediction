"""Shared, validated preprocessing for training and prediction."""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "Input_Data/GCMS_Metabolite_Data_Combined.xlsx"


MICROBIOME_DATA_PATH = DEFAULT_DATA_PATH.with_name("Genus_Abundance_Fecal Data_Combined.xlsx")
FEATURE_COLUMNS = ["Age", "Sex", "Height (cm)", "Weight (Kgs)", "BMI",
                   "Body fat percentage", "MUAC (cm)", "Protein (grams)",
                   "Carbohydrates  (grams)", "Fats (grams)"]


def prepare_features(frame, feature_columns=None):
    """Encode Sex independently of pandas dtype and align the saved schema."""
    data = frame.copy()
    # New fits use kilograms; legacy bundles still receive their exact saved key.
    weight_key = next((name for name in (feature_columns or [])
                       if name in {"Weight (cm)", "Weight (kg)", "Weight (Kgs)"}), "Weight (Kgs)")
    aliases = {name: weight_key for name in ("Weight (cm)", "Weight (kg)", "Weight (Kgs)")}
    aliases["Carbohydrates (grams)"] = "Carbohydrates  (grams)"
    data = data.rename(columns=aliases)
    if data.columns.duplicated().any():
        raise ValueError("Feature names must be unique, including aliases.")
    if feature_columns is not None:
        missing = [name for name in feature_columns if name not in data.columns]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")
        data = data.loc[:, list(feature_columns)].copy()
    if data.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")
    if "Sex" in data:
        values = data["Sex"].astype("string").str.strip().str.lower()
        encoded = values.map({"male": 1, "female": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0})
        if encoded.isna().any():
            raise ValueError("Sex must be Male, Female, 1, or 0.")
        data["Sex"] = encoded
    for column in data:
        try:
            data[column] = pd.to_numeric(data[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must contain numeric values.") from exc
    data = data.astype("float32")
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Features must not contain missing or infinite values.")
    return data


def loadandpreprocess(file_path=DEFAULT_DATA_PATH, *, task="metabolite"):
    with pd.ExcelFile(file_path) as workbook:
        header = pd.read_excel(workbook, header=None, nrows=1)
        if header.empty or header.iloc[0].isna().any() or header.iloc[0].duplicated().any():
            raise ValueError("Training columns must have unique, nonempty names.")
        df = pd.read_excel(workbook)
    if df.shape[1] < 12 or len(df) < 3:
        raise ValueError("Training data needs at least 3 rows, an identifier, 10 features, and target columns.")
    if task not in {"metabolite", "microbiome"}:
        raise ValueError("Unknown prediction task.")
    identifiers = df.iloc[:, 0].astype("string").str.strip()
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.duplicated().any():
        raise ValueError("Each training row needs a unique, nonempty participant identifier.")
    X = prepare_features(df.iloc[:, 1:11], FEATURE_COLUMNS)
    try:
        Y = df.iloc[:, 11:].apply(pd.to_numeric, errors="raise").astype("float64")
    except (ValueError, TypeError) as exc:
        raise ValueError("Target columns must contain numeric values.") from exc
    if not np.isfinite(Y.to_numpy()).all():
        raise ValueError("Targets must not contain missing or infinite values.")
    if task == "microbiome":
        if (Y.to_numpy() < 0).any():
            raise ValueError("Microbiome abundances must be nonnegative.")
        if (Y.sum(axis=1) <= 0).any():
            raise ValueError("Each microbiome training row must have a positive total abundance.")
        return X, Y
    # Preserve the application's existing binary target definition.
    return X, (Y > Y.to_numpy().mean()).astype("float32")
