from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("iris_model.pkl")
le = joblib.load("iris_label_encoder.pkl")

col_order = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[col]) for col in col_order]
        pred = model.predict([features])[0]
        species = le.inverse_transform([pred])[0]
        return jsonify({"prediction": species})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
