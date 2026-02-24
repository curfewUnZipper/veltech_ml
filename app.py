from flask import Flask, request, jsonify
import numpy as np
import joblib
import logging

# -----------------------------
# INIT
# -----------------------------
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("battery_rf_model.pkl")
    app.logger.info("✅ Model loaded successfully")
except Exception as e:
    app.logger.error(f"❌ Model load failed: {e}")
    model = None

# Expected feature order (VERY IMPORTANT)
FEATURE_ORDER = [
    "voltage_rest",
    "voltage_load",
    "voltage_sag",
    "current",
    "temperature",
    "dv_dt"
]

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Battery ML API running"
    })

# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()

        # ---- Validate input ----
        missing = [f for f in FEATURE_ORDER if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        # ---- Build feature vector ----
        features = np.array([[data[f] for f in FEATURE_ORDER]])

        # ---- Predict ----
        prob = model.predict_proba(features)[0][1]
        risk_percent = round(prob * 100, 2)

        # ---- Risk banding (EV-style) ----
        # ---- Risk banding (improved) ----
        if risk_percent >= 85:
            risk_level = "CRITICAL"
        elif risk_percent >= 60:
            risk_level = "HIGH RISK"
        elif risk_percent >= 30:
            risk_level = "DEGRADING"
        else:
            risk_level = "HEALTHY"

        return jsonify({
            "failure_risk_percent": risk_percent,
            "risk_level": risk_level
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)