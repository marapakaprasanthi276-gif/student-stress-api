from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# ✅ Home route
@app.route("/")
def home():
    return "Student Stress API is running"

# ✅ Load model
model = joblib.load("stress_model.pkl")

# ✅ Predict route (ONLY ONCE)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return jsonify({
        "stress_level": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)