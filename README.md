# AI Air Quality Prediction System

A machine learning system for predicting air quality using both classical ML models and LSTM-based deep learning, with an interactive dashboard for visualization and predictions.

## Overview

This project processes historical air quality data, engineers relevant features, and trains predictive models to forecast air quality levels. Results are served through an interactive dashboard for easy exploration and decision-making.

## Features

- Data preprocessing pipeline for cleaning and preparing raw air quality data
- Feature engineering to extract meaningful signals from time-series environmental data
- Two modeling approaches:
  - Classical ML models (e.g., regression/tree-based models)
  - LSTM-based deep learning model for sequential/time-series prediction
- Prediction pipeline for generating forecasts on new data
- Interactive dashboard for visualizing air quality trends and predictions

## Project Structure

```
Air-Quality-Prediction-ML/
├── dashboard/
│   └── app.py                  # Interactive dashboard application
├── src/
│   ├── data_preprocessing.py   # Data cleaning and preparation
│   ├── feature_engineering.py  # Feature extraction and transformation
│   ├── train_ml_model.py       # Training script for classical ML models
│   ├── train_lstm_model.py     # Training script for LSTM model
│   ├── predict.py              # Prediction/inference pipeline
│   └── utils.py                 # Shared utility functions
└── .gitignore
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vivekreddy-25/Air-Quality-Prediction-ML.git
   cd Air-Quality-Prediction-ML
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # On Windows
   source venv/bin/activate   # On macOS/Linux
   pip install -r requirements.txt
   ```

   > Note: Add a `requirements.txt` listing your dependencies (e.g., pandas, numpy, scikit-learn, tensorflow/keras, streamlit or dash, matplotlib) if not already present.

## Usage

**Preprocess the data:**
```bash
python src/data_preprocessing.py
```

**Engineer features:**
```bash
python src/feature_engineering.py
```

**Train a model:**
```bash
python src/train_ml_model.py        # Classical ML model
python src/train_lstm_model.py      # LSTM model
```

**Generate predictions:**
```bash
python src/predict.py
```

**Run the dashboard:**
```bash
python dashboard/app.py
```

## Tech Stack

- Python
- Machine Learning: scikit-learn (or similar)
- Deep Learning: TensorFlow/Keras (LSTM)
- Dashboard: Streamlit/Dash (update based on actual framework used)
- Data handling: pandas, NumPy

## Future Improvements

- Add automated hyperparameter tuning
- Expand model evaluation metrics and comparison reports
- Add real-time data ingestion for live predictions
- Deploy dashboard to a cloud platform

## License

This project is available for educational and research purposes.
