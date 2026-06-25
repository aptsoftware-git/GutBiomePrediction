import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import pickle

from loadandpreprocess import loadandpreprocess
from encoder import encoder
from predictor import predictor

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from scipy.stats import pearsonr
from tensorflow.keras.models import load_model

latent_dimension = 10
predictor_model = None
decoder_model = None
x_scaler = StandardScaler()
y_scaler = StandardScaler()
feature_columns = None
target_columns = None

MODEL_DIR = Path(__file__).resolve().parent / "models"
PREDICTOR_MODEL_PATH = MODEL_DIR / "predictor.keras"
DECODER_MODEL_PATH = MODEL_DIR / "decoder.keras"
META_PATH = MODEL_DIR / "meta.pkl"


def _save_trained_artifacts():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    predictor_model.save(PREDICTOR_MODEL_PATH)
    decoder_model.save(DECODER_MODEL_PATH)

    with open(META_PATH, "wb") as meta_file:
        pickle.dump(
            {
                "x_scaler": x_scaler,
                "y_scaler": y_scaler,
                "feature_columns": list(feature_columns) if feature_columns is not None else None,
                "target_columns": list(target_columns) if target_columns is not None else None,
            },
            meta_file,
        )


def _load_trained_artifacts():
    global predictor_model, decoder_model, x_scaler, y_scaler, feature_columns, target_columns

    if not (PREDICTOR_MODEL_PATH.exists() and DECODER_MODEL_PATH.exists() and META_PATH.exists()):
        return False

    predictor_model = load_model(PREDICTOR_MODEL_PATH, compile=False)
    decoder_model = load_model(DECODER_MODEL_PATH, compile=False)

    with open(META_PATH, "rb") as meta_file:
        meta = pickle.load(meta_file)

    x_scaler = meta["x_scaler"]
    y_scaler = meta["y_scaler"]
    feature_columns = meta["feature_columns"]
    target_columns = meta.get("target_columns")
    return True

def load_data(file_path):
    """
    Load and preprocess the data from the given file path.
    Returns the feature matrix X and target matrix Y.
    """
    X, Y = loadandpreprocess(file_path=file_path)
    return X, Y


def _fit_models(X_train, Y_train, latent_dimension):
    global predictor_model, decoder_model, x_scaler, y_scaler

    X_train_scaled = x_scaler.fit_transform(X_train)
    Y_train_scaled = y_scaler.fit_transform(Y_train)

    early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

    autoencoder_model, encoder_model, decoder_model = encoder(Y_train.shape[1], latent_dimension)
    autoencoder_model.fit(Y_train_scaled, Y_train_scaled, epochs=500, batch_size=8, verbose=0, callbacks=[early_stop])
    latent_train = encoder_model.predict(Y_train_scaled, verbose=0)

    predictor_model = predictor(X_train_scaled.shape[1], latent_dimension)
    predictor_model.fit(X_train_scaled, latent_train, epochs=500, batch_size=8, verbose=0, callbacks=[early_stop])

    return predictor_model, decoder_model

def model_training(X, Y, latent_dimension):
    """
    Train the autoencoder and predictor models using Leave-One-Out cross-validation.
    Returns the predictions for all samples.
    """
    global predictor_model, decoder_model, x_scaler, y_scaler, feature_columns, target_columns
    feature_columns = X.columns
    target_columns = Y.columns
    loo = LeaveOneOut()
    predictions = np.zeros(Y.shape)
    sample_counter = 0

    for train_idx, test_idx in loo.split(X):
        print(f"Training run for sample {sample_counter+1}/{len(X)}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]

        predictor_model, decoder_model = _fit_models(X_train, Y_train, latent_dimension)

        X_train_scaled = x_scaler.transform(X_train)
        X_test_scaled = x_scaler.transform(X_test)

        latent_predicted = predictor_model.predict(X_test_scaled, verbose=0)
        Y_predicted_scaled = decoder_model.predict(latent_predicted, verbose=0)
        Y_predicted = y_scaler.inverse_transform(Y_predicted_scaled)
        Y_predicted = (Y_predicted > 0.5).astype(int)

        predictions[test_idx, :] = Y_predicted

        # Compare Y_predicted with Y_test for this sample
        y_test_arr = Y_test.values.astype(int)
        fold_match_ratio = (y_test_arr == Y_predicted).mean()

        print(f"Fold exact-match ratio: {fold_match_ratio:.4f}")

        compare_df = pd.DataFrame({
            "Genus": Y.columns[:10],
            "Y_test": y_test_arr[0, :10],
            "Y_predicted": Y_predicted[0, :10]
        })
        print("Y_test vs Y_predicted (first 10 genera):")
        print(compare_df)

        sample_counter += 1

    true_values = Y.values
    overall_r2 = r2_score(true_values, predictions, multioutput='uniform_average')
    rmse = np.sqrt(mean_squared_error(true_values, predictions, multioutput='uniform_average'))

    print("Overall R2 =", overall_r2)
    print("RMSE =", rmse)

    _fit_models(X, Y, latent_dimension)

    return overall_r2, rmse

def train_controller(file_path):
    """
    Controller function to load data and train models.
    Returns the trained predictor and decoder models, along with overall R2 and RMSE.
    """
    global predictor_model, decoder_model
    X, Y = load_data(file_path)
    overall_r2, rmse = model_training(X, Y, latent_dimension)
    _save_trained_artifacts()
    return overall_r2, rmse

def predict_controller(user_input_row):
    """
    Predict output values for a single user input row.
    """
    global predictor_model, decoder_model, x_scaler, y_scaler, feature_columns

    if predictor_model is None or decoder_model is None or feature_columns is None:
        print("Loading trained artifacts...")
        loaded = _load_trained_artifacts()
        if not loaded:
            raise ValueError("Model is not trained yet. Call train_controller first.")
    print(user_input_row)
    user_input = user_input_row.copy()
    if "Name" in user_input.columns:
        user_input = user_input.drop(columns=["Name"])

    if "Sex" in user_input.columns and user_input["Sex"].dtype == object:
        user_input["Sex"] = user_input["Sex"].map({"Male": 1, "Female": 0})

    # user_input = user_input.reindex(columns=feature_columns)
    print("=== DEBUG: predict_controller ===")
    print("Reordered input columns:", list(user_input.columns))
    print("Reordered input values:\n", user_input)

    X_test_scaled = x_scaler.transform(user_input)
    print("Scaled input values:\n", X_test_scaled)

    latent_predicted = predictor_model.predict(X_test_scaled, verbose=0)
    print("Latent predicted shape:", latent_predicted.shape)
    print("Latent predicted values:", latent_predicted)

    Y_predicted_scaled = decoder_model.predict(latent_predicted, verbose=0)
    print("Decoder output (scaled) shape:", Y_predicted_scaled.shape)
    print("Decoder output (scaled, first 10):", Y_predicted_scaled[0, :10])

    Y_predicted = y_scaler.inverse_transform(Y_predicted_scaled)
    print("Inverse-transformed output (first 10):", Y_predicted[0, :10])

    Y_predicted = (Y_predicted > 0.5).astype(int)
    print("Final binarized output (first 10):", Y_predicted[0, :10])
    print("=== END DEBUG ===\n")

    if target_columns is not None:
        return pd.DataFrame(Y_predicted, columns=target_columns)
    return Y_predicted


