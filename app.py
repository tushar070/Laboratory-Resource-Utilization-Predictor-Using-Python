import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set page config for beautiful layout
st.set_page_config(page_title="Lab Resource Predictor", page_icon="🧪", layout="wide")

# Custom CSS for glassmorphism and modern UI
st.markdown("""
    <style>
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card h3 {
        color: #64748b !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    .metric-card h2 {
        color: #0f172a !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
    /* specifically target Low Stock Items text color if inline style doesn't work perfectly */
    .metric-card.low-stock h2 {
        color: #ef4444 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Laboratory Resource Utilization Predictor")
st.markdown("Monitor inventory, view usage trends, and predict future resource needs.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("lab_inventory_usage.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure 'generate_data.py' was run.")
        return None

@st.cache_resource
def load_models():
    try:
        model = joblib.load("outputs/rf_model.pkl")
        label_encoders = joblib.load("outputs/label_encoders.pkl")
        scaler = joblib.load("outputs/scaler.pkl")
        feature_names = joblib.load("outputs/model_features.pkl")
        return model, label_encoders, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure 'train_model.py' was run.")
        return None, None, None, None

df = load_data()
model_objects = load_models()

if df is not None and model_objects[0] is not None:
    model, label_encoders, scaler, feature_names = model_objects
    
    # ---------------------------
    # Sidebar
    # ---------------------------
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Predictions", "Reorder Alerts"])
    
    # ---------------------------
    # Page 1: Dashboard
    # ---------------------------
    if page == "Dashboard":
        st.header("📊 Current Inventory Dashboard")
        
        # Display high-level metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Unique Resources</h3><h2>{df['Resource_ID'].nunique()}</h2></div>", unsafe_allow_html=True)
        with col2:
            current_low = len(df.groupby('Resource_ID').last().query('Stock_Level <= Reorder_Level'))
            st.markdown(f"<div class='metric-card low-stock'><h3>Low Stock Items</h3><h2>{current_low}</h2></div>", unsafe_allow_html=True)
        with col3:
            total_used_7d = df[df['Date'] >= (df['Date'].max() - pd.Timedelta(days=7))]['Quantity_Used'].sum()
            st.markdown(f"<div class='metric-card'><h3>Total Used (Last 7 Days)</h3><h2>{total_used_7d}</h2></div>", unsafe_allow_html=True)

        st.subheader("Usage Trends")
        resource_list = df['Resource_Name'].unique()
        selected_resource_chart = st.selectbox("Select Resource to View History", resource_list)
        
        # Filter for the selected resource
        res_data = df[df['Resource_Name'] == selected_resource_chart].sort_values(by='Date')
        
        # Plot Usage over time
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=res_data, x='Date', y='Quantity_Used', color='#3498db', ax=ax, label='Daily Usage')
        # Add moving average
        res_data['7-Day MA'] = res_data['Quantity_Used'].rolling(window=7).mean()
        sns.lineplot(data=res_data, x='Date', y='7-Day MA', color='#e74c3c', ax=ax, label='7-Day Moving Avg')
        
        ax.set_title(f"Usage Trend: {selected_resource_chart}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Quantity Used")
        st.pyplot(fig)
        
        # Plot model performance images
        st.subheader("Model Performance")
        st.markdown("We trained several machine learning models to predict resource usage. Below is their evaluation.")
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            if os.path.exists("outputs/model_comparison.png"):
                st.image("outputs/model_comparison.png", use_container_width=True)
        with col_img2:
            if os.path.exists("outputs/actual_vs_predicted.png"):
                st.image("outputs/actual_vs_predicted.png", use_container_width=True)


    # ---------------------------
    # Page 2: Predictions
    # ---------------------------
    elif page == "Predictions":
        st.header("🔮 Predict Future Resource Requirements")
        st.markdown("Use our trained Random Forest model to estimate how much of a resource you will need on a given day.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dropdowns for user input based on encoders
            selected_res_name = st.selectbox("Resource", df['Resource_Name'].unique())
            
            # Fetch default values for the selected item to populate form
            item_info = df[df['Resource_Name'] == selected_res_name].iloc[-1]
            
            selected_dept = st.selectbox("Department Requesting", df['Lab_Department'].unique(), index=list(df['Lab_Department'].unique()).index(item_info['Lab_Department']))
            pred_date = st.date_input("Date for Prediction", value=datetime.now())
            
        with col2:
            st.markdown("#### Item Characteristics")
            st.info(f"**Reorder Level:** {item_info['Reorder_Level']}")
            st.info(f"**Max Stock:** {item_info['Max_Stock']}")
            
        if st.button("Predict Usage", type="primary"):
            # Prepare data row for prediction
            row = {
                'Resource_ID': item_info['Resource_ID'],
                'Resource_Type': item_info['Resource_Type'],
                'Lab_Department': selected_dept,
                'Usage_Frequency': item_info['Usage_Frequency'],
                'Reorder_Level': item_info['Reorder_Level'],
                'Max_Stock': item_info['Max_Stock'],
                'Stock_Level': item_info['Stock_Level'],  # assuming current stock level
                'Year': pred_date.year,
                'Month': pred_date.month,
                'Day': pred_date.day,
                'DayOfWeek': pred_date.weekday()
            }
            
            input_df = pd.DataFrame([row])
            
            # Encode Categorical variables
            for col in ['Resource_ID', 'Resource_Type', 'Lab_Department', 'Usage_Frequency']:
                # handle unseen labels gracefully by mapping to something known or raising error
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col])
                
            # Scale numerical variables
            num_cols = ['Reorder_Level', 'Max_Stock', 'Stock_Level', 'Year', 'Month', 'Day', 'DayOfWeek']
            input_df[num_cols] = scaler.transform(input_df[num_cols])
            
            # Ensure columns are in the exact same order as training
            input_df = input_df[feature_names]
            
            # Predict
            prediction = model.predict(input_df)
            
            st.success(f"Predicted Quantity Used on **{pred_date.strftime('%Y-%m-%d')}**:")
            st.metric(label=selected_res_name, value=f"{int(round(prediction[0]))} units")
            
            if int(round(prediction[0])) > item_info['Stock_Level']:
                st.warning(f"⚠️ **Warning**: Predicted usage exceeds currently recorded stock level ({item_info['Stock_Level']} units)!")

    # ---------------------------
    # Page 3: Reorder Alerts
    # ---------------------------
    elif page == "Reorder Alerts":
        st.header("🚨 Reorder Alerts & Inventory Check")
        st.markdown("List of items that are running critically low based on their reorder thresholds.")
        
        # Get the latest stock record for each resource
        latest_stock = df.sort_values('Date').groupby('Resource_ID').last().reset_index()
        
        # Filter for items that need reordering
        alerts = latest_stock[latest_stock['Stock_Level'] <= latest_stock['Reorder_Level']]
        
        if alerts.empty:
            st.success("All items are well-stocked! 🎉 No reorders needed currently.")
        else:
            for _, item in alerts.iterrows():
                st.error(f"**{item['Resource_Name']}** ({item['Resource_ID']})")
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Stock", f"{item['Stock_Level']}")
                col2.metric("Reorder Threshold", f"{item['Reorder_Level']}")
                col3.metric("Deficit", f"{item['Stock_Level'] - item['Reorder_Level']}", delta_color="inverse")
                st.markdown("---")
