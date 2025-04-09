from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import torch
from model import SimpleNN

app = Flask(__name__)

# Model setup
INPUT_SIZE = 4
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 1

# Function to generate features for a given month and year
def generate_features(month, year):
    dates = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq='D')
    valid_dates = dates[dates.month == month]
    features = [[d.weekday(), d.day, month, year] for d in valid_dates]
    return np.array(features, dtype=np.float32), valid_dates

# Function to predict total receipts for a given month and year
def predict_for_month(month, year):
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    X_future, dates = generate_features(month, year)
    X_future_tensor = torch.tensor(X_future)

    with torch.no_grad():
        predictions = model(X_future_tensor).squeeze().numpy()
        total_receipts = predictions.sum()

    return int(total_receipts)

# Flask routes
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        month = int(request.form["month"])
        year = int(request.form["year"])
        result = predict_for_month(month, year)
    return render_template("mainPage.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)