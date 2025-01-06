# Project-10
Capstone_Project - Regression - Electric motor temperature prediction

1. Introduction
Electric motor temperature prediction is a critical task in ensuring the
optimal performance and longevity of electric motors, particularly permanent
magnet synchronous motors (PMSMs), which are widely used in applications
such as electric vehicles, wind turbines, and industrial machinery.
Overheating of motors can lead to inefficiencies, failures, and significant
maintenance costs. Therefore, accurate temperature prediction can help
optimize motor operation, prevent overheating, and extend the motor's
service life. In this project, the goal is to predict the temperature of a
PMSM's permanent magnets based on various motor parameters like
voltage, current, motor speed, and ambient temperature. The dataset
provided for this task contains measurements of these motor parameters
across different driving cycles, with the target variable being the permanent
magnet temperature ('pm'). A RandomForest Regressor model was chosen
for its robust performance in regression tasks.
2. Data Collection
The dataset for this project consists of 1.33 million records from electric
cars, specifically monitoring various performance parameters of the PMSM.
These parameters include:
• Motor Voltage (u_q, u_d): The motor voltage in two components,
quadrature and direct.
• Motor Currents (i_q, i_d): The motor current in two components,
quadrature and direct.
• Motor Speed (motor_speed): The rotational speed of the motor.
• Ambient Temperature (ambient): The surrounding temperature of
the environment.
• Coolant Temperature (coolant): The temperature of the coolant
used in the motor.
• Permanent Magnet Temperature (pm): The target variable
representing the temperature of the motor’s permanent magnets.
The dataset was loaded into a Pandas DataFrame, which was then
preprocessed for further analysis and modeling.
3. Data Preprocessing
Before applying the machine learning model, several preprocessing steps
were performed to clean and prepare the data:
3.1 Handling Missing Values
Upon checking for missing values, it was found that certain rows had null
values. These were removed by dropping rows containing any missing values
from the numeric columns (u_q, u_d, i_q, i_d, motor_speed, ambient, coolant, pm).
3.2 Feature Engineering
The dataset contained a categorical feature, profile_id, which was encoded
using one-hot encoding. This transformation was necessary to convert the
categorical values into a format suitable for machine learning algorithms.
The encoded data excluded the first category (drop_first=True) to avoid
multicollinearity.
3.3 Feature Scaling
To ensure all features had similar scales, especially for algorithms like
Random Forest, which are sensitive to feature scaling, the continuous
variables were standardized using the StandardScaler from scikit-learn. This
step was important to prevent any single feature from dominating the model
due to differing scales.
4. Model Building
The objective of the model was to predict the permanent magnet
temperature (pm) of the motor based on the input features. The steps
involved in building the predictive model are outlined below:
4.1 Splitting the Dataset
The dataset was divided into independent variables (features) and the target
variable (pm). Then, it was further split into training and testing sets using
an 80-20 split.
4.2 Training the RandomForest Regressor Model
A RandomForest Regressor model was initialized with 10 estimators and
trained using the training data. Random forests are powerful ensemble
models that perform well with complex, non-linear relationships, making
them suitable for this regression task.
4.3 Making Predictions
Once the model was trained, it was used to make predictions on the test set.
These predictions were then compared to the actual pm values to evaluate
the model's performance.
5. Model Evaluation
The model’s performance was evaluated using several key metrics to
understand its predictive accuracy:
• Mean Squared Error (MSE): A measure of the average squared
difference between the predicted and actual values.
• Mean Absolute Error (MAE): The average of the absolute
differences between the predicted and actual values.
• Root Mean Squared Error (RMSE): The square root of the MSE,
providing a more interpretable measure of prediction error.
• R-Squared (R²): A metric that indicates the proportion of variance in
the target variable that is predictable from the features.
5.1 Performance Metrics
The model's evaluation metrics were as follows:
• Mean Squared Error (MSE): 1.63
• Mean Absolute Error (MAE): 0.45
• Root Mean Squared Error (RMSE): 1.28
• R-Squared (R²): 0.996
These metrics indicate that the model performed exceptionally well, with a
very high R² value, suggesting that the model was able to explain nearly
99.6% of the variance in the permanent magnet temperature.
6. Results & Discussion
6.1 Model Performance
The RandomForest Regressor model performed extremely well, with an R²
value close to 1, indicating that it was able to predict the motor’s permanent
magnet temperature with high accuracy. The RMSE of 1.28 suggests that
the model’s predictions were within a reasonable range of the actual values.
Despite this high performance, some potential improvements could include:
• Hyperparameter Tuning: The model's performance might further
improve with optimized hyperparameters, such as adjusting the
number of estimators, maximum depth of the trees, or other
parameters.
• Feature Engineering: Exploring additional features or
transformations of existing features could enhance the model’s
predictive power.
6.2 Feature Importance
The RandomForest model also provides insight into feature importance,
indicating which variables contributed most to predicting the permanent
magnet temperature. The top features, based on the model’s feature
importance scores, included:
• Coolant Temperature
• Motor Speed
• Ambient Temperature
• Motor Currents (i_d, i_q)
These features were found to have the highest influence on the model's
predictions, and understanding their relationships with temperature could
lead to more targeted optimizations for motor performance.
7. Conclusion
This project successfully demonstrated the use of machine learning,
specifically RandomForest Regressor, to predict the permanent magnet
temperature in permanent magnet synchronous motors. The model achieved
impressive performance, with an R² value of 0.996, meaning that it
explained nearly all of the variance in the target variable. The key features
influencing temperature prediction were coolant temperature, motor speed,
and ambient temperature.
