
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("random_forest_adc_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    amp = float(data["amplitude"])
    dmm1 = float(data["dmm1"])
    dmm2 = float(data["dmm2"])

    features = [
        dmm1, dmm2, amp,
        dmm1 * dmm2,
        amp * dmm1,
        dmm2 * amp,
        dmm1 ** 2,
        dmm2 ** 2,
        amp ** 2
    ]

    prediction = model.predict([features])[0]
    return jsonify({"predicted_adc_avg": round(prediction, 3)})
