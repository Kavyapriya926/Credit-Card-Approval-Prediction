"""
Credit Card Approval Prediction System - Flask Application
SmartBridge AI/ML Internship | Team ID: SWTID-2026-9508

This Flask app loads the pre-trained ML model (best of Logistic Regression,
Decision Tree, Random Forest, XGBoost - selected in the training notebook)
and serves a web form where a bank officer or applicant can enter applicant
details and instantly receive an Approved / Rejected prediction.

FIX (from the previous version): dropdown options are no longer hardcoded.
Hardcoding CATEGORY_OPTIONS is what caused the mismatch you saw (dropdowns
showing meaningless codes like "a"/"b" instead of real category names) -
whatever the model's encoders were actually fit on, the hardcoded dict could
silently drift out of sync with it. This version reads the valid category
values directly off each column's fitted LabelEncoder (`encoders[col].classes_`)
at startup, so the form always matches the model exactly - no matter which
dataset it was trained on. It also auto-detects which columns are numeric
vs. categorical instead of relying on a hand-maintained NUMERIC_FIELDS list.
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

# Optional: friendlier display text for common raw values, without changing
# what actually gets submitted to the model. Add to this as needed - if a
# raw value isn't listed here, the raw value itself is shown as-is.
DISPLAY_LABELS = {
    "Gender": {
        "a": "Male",
        "b": "Female",
    },

    "Married": {
        "l": "Legally Separated",
        "u": "Unmarried",
        "y": "Married",
    },

    "BankCustomer": {
        "g": "Government",
        "gg": "Private",
        "p": "Public",
    },

    "EducationLevel": {
        "aa": "Primary School",
        "c": "High School",
        "cc": "Intermediate",
        "d": "Diploma",
        "e": "Graduate",
        "ff": "Post Graduate",
        "i": "Professional",
        "j": "Doctorate",
        "k": "Technical",
        "m": "Vocational",
        "q": "Other",
        "r": "Unknown",
        "w": "Special",
        "x": "Not Specified",
    },

    "Ethnicity": {
        "bb": "Asian",
        "dd": "African",
        "ff": "European",
        "h": "Hispanic",
        "j": "Middle Eastern",
        "n": "Native",
        "o": "Other",
        "v": "Mixed",
        "z": "Unknown",
    },

    "PriorDefault": {
        "t": "Yes",
        "f": "No",
    },

    "Employed": {
        "t": "Yes",
        "f": "No",
    },

    "Citizen": {
        "g": "Citizen",
        "p": "Permanent Resident",
        "s": "Foreign Citizen",
    },
}

# Friendlier labels for the form field names themselves (falls back to the
# raw column name if not listed here).
FIELD_LABELS = {
    "Gender": "Gender",
    "Age": "Age",
    "Married": "Marital Status",
    "BankCustomer": "Bank Customer",
    "EducationLevel": "Education Level",
    "Ethnicity": "Ethnicity",
    "YearsEmployed": "Years Employed",
    "PriorDefault": "Prior Default",
    "Employed": "Currently Employed",
    "CreditScore": "Credit Score",
    "Citizen": "Citizenship",
    "Income": "Annual Income",
    "Debt": "Debt",
}

# ------------------------------------------------------------------
# Load model artifacts once at startup (not per-request, for performance)
# ------------------------------------------------------------------
try:
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    print("Feature Columns:")
    for col in feature_columns:
        print(col)

    # Any column that has a fitted LabelEncoder is categorical; everything
    # else in feature_columns is treated as numeric. This is derived from
    # the actual trained artifacts, not hardcoded, so it can never drift
    # out of sync with the model like the old CATEGORY_OPTIONS dict did.
    CATEGORICAL_FIELDS = [c for c in feature_columns if c in encoders]
    NUMERIC_FIELDS = [c for c in feature_columns if c not in encoders]

    # Build dropdown options straight from each encoder's known classes,
    # with a friendly display label alongside the raw value that actually
    # gets submitted.
    CATEGORY_OPTIONS = {
    col: [
        {
            "value": value,
            "label": DISPLAY_LABELS.get(col, {}).get(value, value)
        }
        for value in encoders[col].classes_
    ]
    for col in CATEGORICAL_FIELDS
}
except Exception as e:
    logger.error(f"Failed to load model artifacts: {e}")
    model = encoders = scaler = feature_columns = None
    CATEGORICAL_FIELDS = []
    NUMERIC_FIELDS = []
    CATEGORY_OPTIONS = {}


def build_feature_vector(form_data):
    """
    Convert raw form input into a model-ready feature vector, applying the
    same encoding and scaling used during training.
    """
    row = []
    for col in feature_columns:
        value = form_data.get(col)
        if col in NUMERIC_FIELDS:
            try:
                row.append(float(value))
            except (TypeError, ValueError):
                raise ValueError(f"'{FIELD_LABELS.get(col, col)}' must be a number.")
        else:
            le = encoders[col]
            if value not in le.classes_:
                raise ValueError(
                    f"Invalid value '{value}' for field '{FIELD_LABELS.get(col, col)}'. "
                    f"Expected one of: {', '.join(le.classes_)}"
                )
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
        field_labels=FIELD_LABELS,
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
            missing_labels = [FIELD_LABELS.get(c, c) for c in missing]
            return render_template(
                "index.html",
                category_options=CATEGORY_OPTIONS,
                numeric_fields=NUMERIC_FIELDS,
                field_labels=FIELD_LABELS,
                error=f"Missing required field(s): {', '.join(missing_labels)}",
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
            field_labels=FIELD_LABELS,
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
