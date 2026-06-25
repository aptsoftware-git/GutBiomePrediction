import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def loadandpreprocess(file_path="..\\Input_Data\\Genus_Abundance_Fecal Data_Combined.xlsx"):
    df = pd.read_excel(file_path)

    # Convert Sex column from string to float (Male: 1, Female: 0)
    df['Sex'] = df['Sex'].map({'Male': 1, 'Female': 0}).astype(float)

    # Non-invasive variables (B:K)
    X = df.iloc[:,1:11].copy()

    # Invasive variables (L:BKK)
    Y = df.iloc[:,11:].copy()

    # Fill missing values
    # X = X.fillna(X.mean())
    # Y = Y.fillna(Y.mean())

    # Binarize Y using overall mean as threshold
    y_mean = Y.values.mean()
    Y = (Y > y_mean).astype(int)

    # Keep a copy for downstream compatibility
    # Y_log = Y.copy()

    return X, Y