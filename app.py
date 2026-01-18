import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("crop_model.pkl")

@app.route("/")
def home():
    return "KisanBandhu ML API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[ 
        float(data["N"]),
        float(data["P"]),
        float(data["K"]),
        float(data["temperature"]),
        float(data["humidity"]),
        float(data["ph"]),
        float(data["rainfall"])
    ]])

    pred = model.predict(features)[0]
    return jsonify({"success": True, "recommended_crop": pred})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
