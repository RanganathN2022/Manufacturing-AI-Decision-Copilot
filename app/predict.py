import json
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "failure_prediction_model.joblib"
)

CONFIG_PATH = (
    BASE_DIR
    / "models"
    / "model_config.json"
)


model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

THRESHOLD = config["decision_threshold"]


def predict_failure(machine_data):
    """
    Predict failure risk for a single machine.
    """

    df = pd.DataFrame([machine_data])

    probability = model.predict_proba(df)[0][1]

    prediction = int(
        probability >= THRESHOLD
    )

    if probability >= 0.70:
        risk_level = "Critical"
    elif probability >= 0.50:
        risk_level = "High"
    elif probability >= 0.30:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "failure_probability": round(
            float(probability),
            4
        ),
        "failure_prediction": prediction,
        "risk_level": risk_level,
        "decision_threshold": THRESHOLD
    }
    
def get_feature_importance():
    """
    Return model feature importance values.
    """

    model_step = model.named_steps["model"]

    preprocessor_step = model.named_steps["preprocessing"]

    feature_names = (
        preprocessor_step
        .get_feature_names_out()
    )

    importances = model_step.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return importance_df