"""Load the trained model and generate a Kaggle submission file."""

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_test

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "weights" / "model.joblib"
SUBMISSION_PATH = PROJECT_ROOT / "submission.csv"


def generate_submission():
    """Load the trained model and create a Kaggle submission CSV."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. Run `python -m src.train` first."
        )

    # Load saved model + test data
    model = joblib.load(MODEL_PATH)
    test = load_test()

    # Separate ID and features
    test_ids = test["id"]
    X_test = test.drop("id", axis=1)

    # Predict
    predictions = model.predict(X_test)

    # Build submission DataFrame
    submission = pd.DataFrame({
        "id": test_ids,
        "Rings": predictions,
    })

    # Save and report
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print(f"Predictions: min={predictions.min():.2f}, "
          f"max={predictions.max():.2f}, mean={predictions.mean():.2f}")

    return submission


if __name__ == "__main__":
    generate_submission()