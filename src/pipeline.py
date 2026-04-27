"""Build the scikit-learn pipeline for the Abalone regression problem."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline(X, random_state=42, n_estimators=100):
    """
    Build a preprocessing + regression pipeline.

    Parameters
    ----------
    X : DataFrame
        The training features. Used to identify numeric vs categorical columns.
    random_state : int
        Seed for reproducibility.
    n_estimators : int
        Number of trees in the Random Forest.

    Returns
    -------
    Pipeline
        Untrained sklearn Pipeline ready to be fit on data.
    """
    # Identify column types from X (NOT from train — avoids the id/target trap)
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    return model