"""Data loading and splitting utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Project root is two folders up from this file (src/data.py -> src/ -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_train():
    """Load the training CSV from data/raw/."""
    return pd.read_csv(DATA_DIR / "train.csv")


def load_test():
    """Load the test CSV from data/raw/."""
    return pd.read_csv(DATA_DIR / "test.csv")


def split_features_and_target(train, target_col="Rings", drop_cols=("id",)):
    """Separate features (X) from target (y), dropping ID-like columns."""
    cols_to_drop = [c for c in [target_col, *drop_cols] if c in train.columns]
    X = train.drop(columns=cols_to_drop)
    y = train[target_col]
    return X, y


def get_train_val_split(X, y, test_size=0.2, random_state=42):
    """Hold out a fraction of training data for local validation."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)