# Receipt Prediction App

A Flask web application that predicts total receipts based on year and month selection.

## Features
- Simple year/month selection form
- Flask backend with TailwindCSS frontend
- Dockerized for easy deployment
- Pre-trained machine learning model, and neural network model training pipeline

## Requirements
- Python 3.9+
- Dependencies in requirements.txt
- Docker (optional)

## File Structure
- data
    -- data_daily.csv (download data)
- templates
    -- mainPage.html (Web page file that is rendered by app.py)
- app.py (application starter)
- model.py (model structure)
- test.py (testing inference code)
- train.py (training code that generates and stores model weight)

## Quick Start

### Local Setup
```bash
# Clone repo
git clone https://github.com/bosyuan/receipt-prediction.git
cd receipt-prediction

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
```
Access at http://127.0.0.1:5000

### Docker Setup
```bash
# Build and run
docker build -t receipt-prediction-app .
docker run -p 5000:5000 receipt-prediction-app
```
Access at http://localhost:5000

## Usage
1. Select year and month from dropdown
2. Click "Predict"
3. View receipt prediction results
