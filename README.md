# Car Price Prediction Model

## Overview
This project implements a machine learning model to predict car prices based on various features such as year, mileage, fuel type, and other specifications. The model uses Random Forest Regression with preprocessing pipelines for both numerical and categorical data.

## Features
- Data preprocessing with StandardScaler and OneHotEncoder
- Random Forest Regression model
- Comprehensive visualization of results including:
  - Correlation heatmap
  - Actual vs Predicted prices scatter plot
  - Error distribution plot
- Performance metrics calculation (R², RMSE, MAE)

## Dataset
The project uses 'car_data.csv' which should contain the following columns:
- Car_Name
- Year
- Present_Price
- Driven_kms
- Fuel_Type
- Selling_type
- Transmission
- Owner
- Selling_Price (target variable)

## Project Structure
car_price_prediction/
│
├── car_data.csv
├── main.py
├── requirements.txt
└── README.md
Copy
## Installation
1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
Usage
Run the main script:
bashCopypython main.py
Dependencies

Python 3.7+
pandas
numpy
scikit-learn
matplotlib
seaborn

Model Details

Algorithm: Random Forest Regressor
Features preprocessing:

Numerical features: StandardScaler
Categorical features: OneHotEncoder


Train-test split: 80-20

Performance Metrics
The model's performance is evaluated using:

R² Score
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)

Visualizations
The code generates three main visualizations:

Correlation heatmap of numeric features
Actual vs Predicted prices scatter plot
Distribution of prediction errors

Resources and References

Scikit-learn documentation: https://scikit-learn.org/
Pandas documentation: https://pandas.pydata.org/
Seaborn documentation: https://seaborn.pydata.org/

