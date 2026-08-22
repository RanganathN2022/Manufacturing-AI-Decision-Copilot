import json
from pathlib import Path

import joblib
import pandas as pd
import shap
from app.recommendation import generate_recommendation


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# LOAD MODEL AND CONFIGURATION
# ============================================================

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

THRESHOLD = config["decision_threshold"]


# ============================================================
# MODEL COMPONENTS
# ============================================================

rf_model = model.named_steps["model"]

preprocessor = model.named_steps["preprocessing"]

shap_explainer = shap.TreeExplainer(
    rf_model
)


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_prediction(machine_data, top_n=5):
    """
    Generate SHAP-based explanation for one machine.
    """

    df = pd.DataFrame([machine_data])

    transformed_data = preprocessor.transform(df)

    shap_all = shap_explainer.shap_values(
        transformed_data
    )

    # SHAP output shape:
    # (samples, features, classes)
    #
    # Class 0 = No Failure
    # Class 1 = Failure

    failure_shap = shap_all[0, :, 1]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": failure_shap
    })

    explanation_df["abs_shap"] = (
        explanation_df["shap_value"].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return explanation_df.head(top_n)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():
    """
    Return global model feature importance values.
    """

    model_step = model.named_steps["model"]

    preprocessor_step = (
        model.named_steps["preprocessing"]
    )

    feature_names = (
        preprocessor_step
        .get_feature_names_out()
    )

    importances = (
        model_step.feature_importances_
    )

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


# ============================================================
# FAILURE PREDICTION
# ============================================================

def predict_failure(machine_data):
    """
    Predict failure risk and provide explanations
    for a single machine.
    """

    df = pd.DataFrame([machine_data])

    probability = (
        model.predict_proba(df)[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if probability >= 0.70:
        risk_level = "Critical"

    elif probability >= 0.50:
        risk_level = "High"

    elif probability >= 0.30:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    # --------------------------------------------------------
    # Engineering-based key factors
    # --------------------------------------------------------

    key_factors = []

    if machine_data["Operating_Hours"] >= 30000:
        key_factors.append(
            "High operating hours"
        )

    if machine_data["Last_Maintenance_Days"] >= 200:
        key_factors.append(
            "Long time since last maintenance"
        )

    if machine_data["Vibration_mm_s"] >= 5:
        key_factors.append(
            "Elevated vibration"
        )

    if machine_data["Temperature_C"] >= 75:
        key_factors.append(
            "Elevated temperature"
        )

    if machine_data["Machine_Age_Years"] >= 10:
        key_factors.append(
            "High machine age"
        )

    if machine_data["Failure_Count"] >= 3:
        key_factors.append(
            "Previous failure history"
        )

    # --------------------------------------------------------
    # SHAP explanation
    # --------------------------------------------------------

    explanation_df = explain_prediction(
        machine_data,
        top_n=5
    )
    recommendation = generate_recommendation(
    machine_data,
    float(probability),
    risk_level
)

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "failure_probability": round(
            float(probability),
            4
        ),
        "failure_prediction": prediction,
        "risk_level": risk_level,
        "decision_threshold": THRESHOLD,
        "key_factors": key_factors,
        "shap_explanation": (
            explanation_df
            .to_dict(orient="records")
        ),
        "maintenance_recommendation": recommendation
    }