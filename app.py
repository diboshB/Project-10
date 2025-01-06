# app.py

import joblib
import pandas as pd
import sqlite3
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('EMT_model.joblib')  # Load model from EMT_model.joblib
scaler = joblib.load('EMT_scaler.joblib')  # Load scaler from EMT_scaler.joblib

# Load the training data to get the correct column names for one-hot encoding
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
train_data = pd.read_sql_query('SELECT * FROM Electric_cars', conn)
conn.close()

# Pre-process the data (similar to the training data)
numeric_cols = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'pm']
train_data[numeric_cols] = train_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
train_data = train_data.dropna(subset=numeric_cols)

# One-hot encode the categorical column 'profile_id' the same way it was done during training
train_data_encoded = pd.get_dummies(train_data, columns=['profile_id'], drop_first=True)

# The model expects features based on this encoded format, so we need to keep the columns consistent
expected_columns = train_data_encoded.drop('pm', axis=1).columns

# Define the API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.get_json()

        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # One-hot encode the input data for 'profile_id' the same way we did during training
        input_df_encoded = pd.get_dummies(input_df, columns=['profile_id'], drop_first=True)

        # Ensure that the input data has all required one-hot encoded columns
        for col in expected_columns:
            if col not in input_df_encoded.columns:
                input_df_encoded[col] = 0

        # Reorder the columns to match the training data's order
        input_df_encoded = input_df_encoded[expected_columns]

        # Scale the input data using the same scaler
        input_scaled = scaler.transform(input_df_encoded)

        # Make the prediction
        prediction = model.predict(input_scaled)

        # Return the prediction result as a JSON response
        return jsonify({'predicted_pm': prediction[0]}), 200

    except Exception as e:
        # If there's an error, return a 400 error with the message
        return jsonify({'error': str(e)}), 400

# Run the app directly on port 5005
if __name__ == '__main__':
    app.run(debug=True, port=5005)
