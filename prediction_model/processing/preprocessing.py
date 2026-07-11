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
