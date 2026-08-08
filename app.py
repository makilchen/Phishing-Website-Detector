"""
app.py
-------
Flask web app: user pastes a URL, backend extracts features and
returns a Legitimate / Phishing prediction with a confidence score.

Run: python3 app.py
Then open http://localhost:5000 in a browser.
"""

import os
import joblib
from flask import Flask, request, jsonify, render_template

from features import extract_features, features_to_vector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

app = Flask(__name__)

_bundle = None


def get_model_bundle():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model/model.pkl not found. Run `python3 train_model.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.route("/")
def index():
    bundle = get_model_bundle()
    return render_template("index.html", model_name=bundle["name"])


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Please enter a URL."}), 400

    bundle = get_model_bundle()
    model = bundle["model"]

    feats = extract_features(url)
    vector = [features_to_vector(feats)]

    pred = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = round(max(proba) * 100, 2)

    label = "Phishing" if pred == 1 else "Legitimate"

    return jsonify({
        "url": url,
        "label": label,
        "confidence": confidence,
        "model_used": bundle["name"],
        "features": feats,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
