import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_lab_data(output_file='lab_inventory_usage.csv', num_days=365):
    """
    Generates a synthetic dataset for Laboratory Resource Utilization.
    Saves the output to a CSV file.
    """
    print("Generating synthetic laboratory inventory data...")
    
    # Random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # 1. Define base resources with their characteristics
    base_resources = [
        {"id": "R001", "name": "Ethanol 95%", "type": "Chemical", "reorder_level": 50, "max_stock": 500, "usage_freq": "High"},
        {"id": "R002", "name": "Petri Dishes (Pack)", "type": "Consumable", "reorder_level": 20, "max_stock": 200, "usage_freq": "High"},
        {"id": "R003", "name": "Pipette Tips (Box)", "type": "Consumable", "reorder_level": 15, "max_stock": 150, "usage_freq": "Medium"},
        {"id": "R004", "name": "Hydrochloric Acid 1M", "type": "Chemical", "reorder_level": 30, "max_stock": 100, "usage_freq": "Medium"},
        {"id": "R005", "name": "Acetone", "type": "Chemical", "reorder_level": 40, "max_stock": 300, "usage_freq": "High"},
        {"id": "R006", "name": "Microscope Slides", "type": "Consumable", "reorder_level": 10, "max_stock": 100, "usage_freq": "Medium"},
        {"id": "R007", "name": "Centrifuge Tubes", "type": "Consumable", "reorder_level": 25, "max_stock": 250, "usage_freq": "High"},
        {"id": "R008", "name": "Sodium Hydroxide 1M", "type": "Chemical", "reorder_level": 20, "max_stock": 80, "usage_freq": "Low"},
        {"id": "R009", "name": "Gloves (Box)", "type": "Consumable", "reorder_level": 50, "max_stock": 400, "usage_freq": "High"},
        {"id": "R010", "name": "Distilled Water (L)", "type": "Chemical", "reorder_level": 100, "max_stock": 1000, "usage_freq": "Very High"}
    ]

    departments = ["Microbiology", "Biochemistry", "Analytical Chemistry", "Pathology", "General AI Lab"]

    # Generate dates (past 'num_days' up to today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    date_list = [start_date + timedelta(days=x) for x in range(num_days)]

    data = []
    
    # Generate daily records
    for current_date in date_list:
        # Not all resources are used every day, so we take a sample
        # Weekends have less usage
        is_weekend = current_date.weekday() >= 5
        num_transactions = random.randint(3, 8) if is_weekend else random.randint(15, 30)

        for _ in range(num_transactions):
            resource = random.choice(base_resources)
            
            # Base usage based on frequency
            if resource['usage_freq'] == 'Very High':
                qty_used = random.randint(5, 50)
            elif resource['usage_freq'] == 'High':
                qty_used = random.randint(2, 20)
            elif resource['usage_freq'] == 'Medium':
                qty_used = random.randint(1, 10)
            else:
                qty_used = random.randint(1, 5)

            # Introduce some seasonality/random spikes
            if current_date.month in [11, 12] and resource['type'] == 'Consumable': 
                # Year end project burst
                qty_used = int(qty_used * random.uniform(1.2, 1.8))
                
            department = random.choice(departments)
            
            data.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Resource_ID": resource["id"],
                "Resource_Name": resource["name"],
                "Resource_Type": resource["type"],
                "Quantity_Used": qty_used,
                "Lab_Department": department,
                "Usage_Frequency": resource["usage_freq"],
                "Reorder_Level": resource["reorder_level"],
                "Max_Stock": resource["max_stock"]
            })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by date
    df = df.sort_values(by="Date").reset_index(drop=True)

    # Now let's calculate Stock_Level dynamically by assuming restocks happen
    # We will simulate a running stock for each item over time
    stocks = {r['id']: r['max_stock'] for r in base_resources}
    stock_levels = []

    for index, row in df.iterrows():
        res_id = row['Resource_ID']
        used = row['Quantity_Used']
        
        # Deduct used
        stocks[res_id] -= used
        
        # If stock is below reorder level (or goes negative), we "reordered" and it arrived
        if stocks[res_id] < row['Reorder_Level']:
            # Restock to Max_Stock
            stocks[res_id] = row['Max_Stock']
            
        stock_levels.append(stocks[res_id])

    df['Stock_Level'] = stock_levels

    # Add a bit of noise directly related to our target variable (Quantity_Used) 
    # to make the prediction model have something to learn (e.g. usage depends on day of week)
    df['Date'] = pd.to_datetime(df['Date'])
    df.loc[df['Date'].dt.dayofweek == 0, 'Quantity_Used'] += np.random.randint(1, 5) # Mondays are busy
    df.loc[df['Date'].dt.dayofweek == 4, 'Quantity_Used'] += np.random.randint(3, 8) # Fridays are busy wrapping up
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Data successfully generated and saved to {output_file}")
    print(f"Total records generated: {len(df)}")

if __name__ == "__main__":
    generate_lab_data()
