import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    df = pd.read_csv('e:/internship/car_data.csv')
    numeric_features = ['Year', 'Present_Price', 'Driven_kms']
    categorical_features = ['Car_Name', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
    return df, numeric_features, categorical_features

def create_pipeline(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, 
                                          max_depth=15,
                                          min_samples_split=5,
                                          random_state=42))
    ])
    return model

def create_visualizations(df, y_test, y_pred, model, numeric_features):
    sns.set_style("whitegrid")
    
    numeric_df = df[numeric_features + ['Selling_Price']]
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap of Numeric Features', pad=20, size=12)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    g = sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Selling Price', size=10)
    plt.ylabel('Predicted Selling Price', size=10)
    plt.title('Actual vs Predicted Car Prices', pad=20, size=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='blue', alpha=0.6)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    plt.title('Distribution of Prediction Errors', pad=20, size=12)
    plt.xlabel('Prediction Error', size=10)
    plt.ylabel('Frequency', size=10)
    plt.tight_layout()
    plt.show()

def main():
    df, numeric_features, categorical_features = load_and_prepare_data()
    
    X = df.drop('Selling_Price', axis=1)
    y = df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = create_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nModel Performance Metrics:")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f}")
    
    print("\nPrice Comparisons (in lakhs):")
    comparison_df = pd.DataFrame({
        'Actual Price': y_test,
        'Predicted Price': y_pred,
        'Difference': y_test - y_pred
    })
    print("\nFirst 10 predictions:")
    print(comparison_df.round(2).head(10))
    
    create_visualizations(df, y_test, y_pred, model, numeric_features)
    
    return model, model.named_steps['preprocessor']

if __name__ == "__main__":
    model, preprocessor = main()