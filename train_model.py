# train_model.py

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib  

print("Connected to database...\n")
# Connecting to the database
conn = sqlite3.connect('/Users/diboshbaruah/Desktop/Database.db')
data = pd.read_sql_query('SELECT * FROM Electric_cars', conn)
conn.close()

# Pre-processing the data
numeric_cols = ['u_q', 'coolant', 'u_d', 'motor_speed', 'i_d', 'i_q', 'ambient', 'pm']
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Dropping rows with missing values in numeric columns
data = data.dropna(subset=numeric_cols)

# One-hot encoding for the categorical variable
data_encoded = pd.get_dummies(data, columns=['profile_id'], drop_first=True)

print("Data pre-processing completed \n")

# Features and target variable
features = data_encoded.drop('pm', axis=1)
target = data_encoded['pm']

# Train-test splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Model Training stated using RandomForestRegressor!!!\n")
# Model training (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Saving the trained model and scaler using joblib
joblib.dump(model, 'EMT_model.joblib')  # Save model as EMT_model.joblib
joblib.dump(scaler, 'EMT_scaler.joblib')  # Save scaler as EMT_scaler.joblib

print("Model and Scaler have been saved as EMT_model.joblib and EMT_scaler.joblib.")
