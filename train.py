import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleNN

# Define hyper parameters
INPUT_SIZE = 4
HIDDEN_SIZE1 = 64
HIDDEN_SIZE2 = 32
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS = 1000

# Load data from CSV file
def load_data():

    data = pd.read_csv('data/data_daily.csv')
    
    data['date'] = pd.to_datetime(data['# Date'])
    data['weekday'] = data['date'].dt.dayofweek
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['receiptCount'] = (data['Receipt_Count'])

    return data

# Extract features and target variable from loaded data
def extract_features(data):
    features = []
    target = []
    for i in range(len(data)):
        features.append([data['weekday'][i], data['day'][i], data['month'][i], data['year'][i]])
        target.append(data['receiptCount'][i])

    return np.array(features), np.array(target)

if __name__ == "__main__":
    data = load_data()
    X, y = extract_features(data)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Initialize model, loss function and optimizer
    model = SimpleNN(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')

