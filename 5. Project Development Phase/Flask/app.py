"""
Credit Card Approval Prediction System - Flask Application
SmartBridge AI/ML Internship | Team ID: SWTID-2026-9508

This Flask app loads the pre-trained ML model (best of Logistic Regression,
Decision Tree, Random Forest, XGBoost - selected in the training notebook)
and serves a web form where a bank officer or applicant can enter applicant
details and instantly receive an Approved / Rejected prediction.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# ------------------------------------------------------------------
# App & logging setup
# ------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Model")

# ------------------------------------------------------------------
# Load model artifacts once at startup (not per-request, for performance)
# ------------------------------------------------------------------
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    logger.info("Model artifacts loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    model = encoders = scaler = feature_columns = None

# Dropdown options for anonymized categorical fields (values match training data)
CATEGORY_OPTIONS = {
    "Gender": ["a", "b"],
    "Married": ["l", "u", "y"],
    "BankCustomer": ["g", "gg", "p"],
    "EducationLevel": ["aa", "c", "cc", "d", "e", "ff", "i", "j", "k", "m", "q", "r", "w", "x"],
    "Ethnicity": ["bb", "dd", "ff", "h", "j", "n", "o", "v", "z"],
    "PriorDefault": ["f", "t"],
    "Employed": ["f", "t"],
    "Citizen": ["g", "p", "s"],
}

NUMERIC_FIELDS = ["Age", "Debt", "YearsEmployed", "CreditScore", "Income"]


def build_feature_vector(form_data):
    """
    Convert raw form input into a model-ready feature vector, applying the
    same encoding and scaling used during training.
    """
    row = []
    for col in feature_columns:
        value = form_data.get(col)
        if col in NUMERIC_FIELDS:
            row.append(float(value))
        else:
            le = encoders[col]
            if value not in le.classes_:
                raise ValueError(f"Invalid value '{value}' for field '{col}'")
            row.append(int(le.transform([value])[0]))
    X = pd.DataFrame([row], columns=feature_columns)
    X_scaled = scaler.transform(X)
    return X_scaled


@app.route("/")
def index():
    """Render the credit card application form."""
    return render_template(
        "index.html",
        category_options=CATEGORY_OPTIONS,
        numeric_fields=NUMERIC_FIELDS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle form submission, run the ML model, and show the result."""
    if model is None:
        return render_template("result.html", error="Model is not available. Please contact the administrator.")

    try:
        form_data = request.form.to_dict()

        # Basic validation: all fields must be present
        missing = [c for c in feature_columns if c not in form_data or form_data[c] == ""]
        if missing:
            return render_template(
                "index.html",
                category_options=CATEGORY_OPTIONS,
                numeric_fields=NUMERIC_FIELDS,
                error=f"Missing required field(s): {', '.join(missing)}",
                form_data=form_data,
            )

        X = build_feature_vector(form_data)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

        result = "Approved" if prediction == 1 else "Rejected"
        confidence = round(float(probability) * 100, 2) if probability is not None else None

        logger.info(f"Prediction made: {result} (confidence={confidence})")

        return render_template("result.html", result=result, confidence=confidence)

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        return render_template(
            "index.html",
            category_options=CATEGORY_OPTIONS,
            numeric_fields=NUMERIC_FIELDS,
            error=str(ve),
            form_data=request.form.to_dict(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template("result.html", error="Something went wrong while generating the prediction. Please try again.")


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
