"""Train the Abalone regression pipeline and save the trained model."""
import warnings
warnings.filterwarnings("ignore")


from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

from src.data import (
    get_train_val_split,
    load_train,
    split_features_and_target,
)
from src.pipeline import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
MODEL_PATH = WEIGHTS_DIR / "model.joblib"


def train_and_save(tune=True):
    """
    Train the regression pipeline and save the trained model to weights/.

    Parameters
    ----------
    tune : bool
        If True, run RandomizedSearchCV. If False, fit the pipeline with defaults.
    """
    # Load and split
    train = load_train()
    X, y = split_features_and_target(train)
    X_train, X_val, y_train, y_val = get_train_val_split(X, y)

    # Build pipeline
    model = build_pipeline(X_train)

    if tune:
        param_distributions = {
            "regressor__n_estimators":      [100, 200],
            "regressor__max_depth":         [None, 20],
            "regressor__max_features":      ["sqrt"],
            "regressor__min_samples_split": [2, 5],
            "regressor__min_samples_leaf":  [1, 2],
        }

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=8,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best CV MAE: {-search.best_score_:.3f}")
        print(f"Best params: {search.best_params_}")
    else:
        model.fit(X_train, y_train)
        best_model = model

    # Evaluate on validation
    y_pred = best_model.predict(X_val)
    val_mae = mean_absolute_error(y_val, y_pred)
    val_r2 = r2_score(y_val, y_pred)
    print(f"Validation MAE: {val_mae:.3f} rings")
    print(f"Validation R²:  {val_r2:.4f}")

    # Save the trained model
    WEIGHTS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    return best_model


if __name__ == "__main__":
    train_and_save(tune=True)