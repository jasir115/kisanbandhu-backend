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

    try:
        features = np.array([[ 
            float(data["N"]),
            float(data["P"]),
            float(data["K"]),
            float(data["temperature"]),
            float(data["humidity"]),
            float(data["ph"]),
            float(data["rainfall"])
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "success": True,
            "recommended_crop": prediction
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
