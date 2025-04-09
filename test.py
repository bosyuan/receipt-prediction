import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import SimpleNN

# Define model parameters (same as training)
INPUT_SIZE = 4
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 1

# Function to generate features for a given month and year
def generate_features(month, year):
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq='D')
    valid_dates = dates[dates.month == month]

    features = []
    for date in valid_dates:
        weekday = date.weekday()
        day = date.day
        features.append([weekday, day, month, year])
    
    return np.array(features, dtype=np.float32)

# Function to predict total receipts for a given month and year
def predict_for_month(month, year):
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    model.eval()

    # Prepare future features
    X_future = generate_features(month, year)
    X_future_tensor = torch.tensor(X_future)

    # Predict
    with torch.no_grad():
        predictions = model(X_future_tensor)
        total_receipts = predictions.sum().item()
        
    return total_receipts

if __name__ == "__main__":
    print(predict_for_month(month=3, year=2022))