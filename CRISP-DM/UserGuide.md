# Airbnb Price Prediction Model - User Guide

### Overview
This model predicts the price of an Airbnb listing based on features such as location, room type, and listing availability. It was trained on NYC Airbnb data and fine-tuned for accuracy.

### Requirements
- Python 3.x
- Required packages: `pandas`, `joblib`, `sklearn`

### Loading the Model
To use the model, first load the saved model file.

```python
import joblib
model = joblib.load('final_airbnb_price_model.pkl')
