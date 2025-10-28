from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('car_price_model.pkl')

# The order of features as expected by the model:
FEATURES = [
    "Car_Name",        # Encoded integer (see below)
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",       # Encoded integer
    "Selling_type",    # Encoded integer
    "Transmission",    # Encoded integer
    "Owner"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        # Collect inputs in the expected order as float/int
        input_features = [float(data[feature]) for feature in FEATURES]
        
        price = model.predict([input_features])[0]
        return jsonify({'prediction': f"{price:,.2f}"})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
