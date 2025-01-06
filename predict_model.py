# predict_model.py

import joblib
import numpy as np
import pandas as pd
import sqlite3

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

# Input data (replace this with your own input)
input_data = {
    'u_q': 0.5,
    'coolant': 0.2,
    'u_d': 0.4,
    'motor_speed': 3000,
    'i_d': 0.8,
    'i_q': 0.6,
    'ambient': 25,
    'profile_id_type_1': 1,  # Example of one-hot encoded profile_id
    # You should include all the other profile_id columns here if necessary (e.g., profile_id_type_2, etc.)
}

# Convert the input data into a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure that the input DataFrame has the same columns as the training data
# Add missing columns with value 0 if they are not in the input
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder the columns to match the training data's order
input_df = input_df[expected_columns]

# Scale the input data using the same scaler
input_scaled = scaler.transform(input_df)

# Make predictions
prediction = model.predict(input_scaled)

# Output the prediction
print(f"Predicted 'pm': {prediction[0]}")
