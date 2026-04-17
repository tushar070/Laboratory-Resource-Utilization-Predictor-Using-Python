import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def main():
    print("--- Laboratory Resource Utilization Predictor: Training Pipeline ---")
    
    # 1. Load the Dataset
    print("\n[1] Loading dataset...")
    data_path = 'lab_inventory_usage.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run generate_data.py first.")
        return
        
    df = pd.read_csv(data_path)
    print(f"Data Loaded Successfully. Shape: {df.shape}")
    print(df.head())

    # Create an output directory for saving plots
    os.makedirs('outputs', exist_ok=True)

    # 2. Exploratory Data Analysis (EDA)
    print("\n[2] Performing Exploratory Data Analysis (EDA)...")
    
    # Plot 1: Total Quantity Used by Resource Type
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Resource_Type', y='Quantity_Used', estimator=sum, ci=None, palette="viridis")
    plt.title('Total Quantity Used by Resource Type')
    plt.ylabel('Total Quantity')
    plt.savefig('outputs/quantity_by_type.png')
    plt.close()
    print(" - Saved plot: outputs/quantity_by_type.png")

    # Plot 2: Daily Usage Trend (Aggregated)
    df['Date'] = pd.to_datetime(df['Date'])
    daily_usage = df.groupby('Date')['Quantity_Used'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_usage, x='Date', y='Quantity_Used', color='coral')
    plt.title('Daily Cumulative Resource Usage')
    plt.savefig('outputs/daily_usage_trend.png')
    plt.close()
    print(" - Saved plot: outputs/daily_usage_trend.png")

    # 3. Feature Engineering
    print("\n[3] Feature Engineering...")
    # Extract useful information from Date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    
    # Drop original Date column as ML models only understand numbers
    df = df.drop(columns=['Date'])
    
    # In a real project, we might add lag features (e.g. usage yesterday), 
    # but for simplicity, we stick to date components and categorical info.

    # 4. Data Preprocessing
    print("\n[4] Data Preprocessing...")
    
    # Separate features (X) and target variable (y)
    # Target variable: 'Quantity_Used' (what we want to predict)
    # We will exclude 'Resource_Name' since 'Resource_ID' represents the same thing.
    y = df['Quantity_Used']
    X = df.drop(columns=['Quantity_Used', 'Resource_Name'])

    # Handle Categorical Data using Label Encoding
    # Algorithms need numbers, not words (like "Chemical")
    categorical_cols = ['Resource_ID', 'Resource_Type', 'Lab_Department', 'Usage_Frequency']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le # Save encoder for later use
        
    # Scale Numerical Data using StandardScaler
    # Ensures all numerical values have the same scale (e.g. Stock_Level vs Day)
    scaler = StandardScaler()
    numerical_cols = ['Reorder_Level', 'Max_Stock', 'Stock_Level', 'Year', 'Month', 'Day', 'DayOfWeek']
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save preprocessing objects
    joblib.dump(label_encoders, 'outputs/label_encoders.pkl')
    joblib.dump(scaler, 'outputs/scaler.pkl')
    
    # Split data into Training Set (80%) and Testing Set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f" - Training Set: {X_train.shape[0]} samples")
    print(f" - Testing Set: {X_test.shape[0]} samples")

    # 5. Model Building & Training
    print("\n[5] Training Machine Learning Models...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # 6. Evaluation
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2 Score": round(r2, 4)
        })
        
        if name == "Random Forest":
            best_model = model
            # Save actual vs predicted for plotting
            rf_predictions = y_pred

    # Display Results
    print("\n--- Model Evaluation Results ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Plot 3: Model Comparison (R2 Score)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x='Model', y='R2 Score', palette='Blues_d')
    plt.title('Model R² Score Comparison Higher is Better')
    plt.ylim(0, 1)
    plt.savefig('outputs/model_comparison.png')
    plt.close()
    
    # Plot 4: Actual vs Predicted (Random Forest)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, rf_predictions, alpha=0.5, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2) # Diagonal line
    plt.xlabel('Actual Quantity Used')
    plt.ylabel('Predicted Quantity Used')
    plt.title('Random Forest: Actual vs Predicted')
    plt.savefig('outputs/actual_vs_predicted.png')
    plt.close()

    # 7. Save Best Model
    print("\n[7] Saving Best Model (Random Forest)...")
    joblib.dump(best_model, 'outputs/rf_model.pkl')
    # Save the base features template for streamlit app
    joblib.dump(list(X.columns), 'outputs/model_features.pkl')
    print(" - Model saved to outputs/rf_model.pkl")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
