from flask import Flask, request, jsonify
import joblib
import pandas as pd

airbnb_price_prediction_app = Flask(__name__)
model = joblib.load('final_airbnb_price_model.pkl')

@airbnb_price_prediction_app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])  # Convert JSON to DataFrame
    prediction = model.predict(df)
    return jsonify({'predicted_price': prediction[0]})

if __name__ == '__main__':
    airbnb_price_prediction_app.run(debug=False)
