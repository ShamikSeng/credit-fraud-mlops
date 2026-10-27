from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import RobustScaler
from prediction_model.config import config
import numpy as np


class MeanImputer(BaseEstimator,TransformerMixin):
    """Defensive numeric imputation. The ULB fraud CSV has no missing values, but a
    NaN can still arrive via the batch CSV path — fitting the fill value on train
    only keeps this leakage-safe."""
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
            self.mean_dict[col] = X[col].mean()
        return self

    def transform(self,X):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna(self.mean_dict[col])
        return X


class RobustScalerTransformer(BaseEstimator,TransformerMixin):
    """Column-wise RobustScaler (median / IQR) applied only to the given columns,
    preserving the DataFrame's columns and their order. Used for the two non-PCA,
    heavily-skewed columns (Amount, Time); V1..V28 are already PCA-scaled.

    Leakage-safe: the scaler is fit inside the sklearn Pipeline, so it learns the
    median/IQR from the training fold only (never the held-out rows)."""
    def __init__(self,variables=None):
        self.variables = variables

    def fit(self,X,y=None):
        self.scaler_ = RobustScaler()
        self.scaler_.fit(X[self.variables])
        return self

    def transform(self,X):
        X = X.copy()
        X[self.variables] = self.scaler_.transform(X[self.variables])
        return X


class OneHotTransformer(BaseEstimator,TransformerMixin):
    """One-hot encode categorical columns against a FIXED category list, keeping a DataFrame.

    Each category becomes a 0/1 column named "<feature>=<value>" (e.g. "upi_app=Paytm"),
    and the original column is dropped. The fixed list (not one learned from the data) means
    every model trained on this data has the same columns, and an unseen value simply encodes as all
    zeros. The "=" naming lets explain.py add the dummies' SHAP values back up to one
    contribution per original feature."""
    def __init__(self,categories=None):
        self.categories = categories

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X = X.copy()
        for col, values in self.categories.items():
            column = X.pop(col).astype(str)
            for value in values:
                X[f"{col}={value}"] = (column == value).astype(float)
        return X
