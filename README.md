# Laboratory Resource Utilization Predictor 🧪

This is a complete Machine Learning project built using Python. The objective is to analyze laboratory inventory usage and predict future resource needs. It helps in optimizing stock management and reducing waste in laboratory environments.

**Features included:**
1. Synthetic Data Generation script
2. Beautifully commented Data Preprocessing, feature engineering, and model training workflow
3. Multiple algorithms compared: Linear Regression, Decision Tree, Random Forest
4. A rich, interactive Streamlit Web Dashboard

---

## 🛠 Required Libraries

Ensure you have Python installed. Then, install the required packages using the terminal:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

Follow these steps exactly to run the project from start to finish.

### Step 1: Generate the Dataset
We use a synthetic data generator to create highly realistic data tailored exactly to our features (Resource_ID, Stock Levels, Reorder Levels, etc). Run:

```bash
python generate_data.py
```
*This will create the dataset `lab_inventory_usage.csv` with over 6000 records.*

### Step 2: Train the Machine Learning Models
This script handles everything from data cleaning to model comparisons. It trains three different models and exports the best one along with visualization graphs!

```bash
python train_model.py
```
*This will create a new folder called `/outputs` containing graphs (like `actual_vs_predicted.png`) and the trained ML model files (`rf_model.pkl`).*

### Step 3: Run the Web Dashboard
After data is generated and the model is trained, start the interactive dashboard:

```bash
streamlit run app.py
```
*Your browser will automatically open and display the Dashboard, Predictions, and Reorder Alerts!*

---

## 🧠 What's Happening Under The Hood?

1. **Preprocessing (`train_model.py`)**: 
   - We load the dataset and identify categorical data (like "Chemical") mapping them into numbers using `LabelEncoder`.
   - We scale numerical data using `StandardScaler` so that high values (like Max_Stock = 1000) don't overpower low values (like Day = 2).

2. **Feature Engineering**:
   - Dates cannot be fed directly into standard models. We extract the `Year`, `Month`, `Day`, and `DayOfWeek`. Since ML algorithms try to find patterns, knowing it's a Monday or Friday significantly helps the prediction.

3. **Machine Learning Model**: 
   - **Random Forest** is chosen as the winner. It works by building hundreds of independent "decision trees" and averages their answers to make incredibly robust predictions on future inventory requirements.

---

*This project was created for educational purposes to demonstrate an end-to-end Machine Learning pipeline.*
