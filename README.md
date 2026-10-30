# Air Quality Prediction

This project explores air quality prediction with historical pollutant data. It includes scripts for preparing data, training a Random Forest model and an LSTM model, and viewing results in a Streamlit dashboard.

## What's included

- **Data preparation:** Sorts records by date, fills missing values, and caps numeric outliers.
- **Feature engineering:** Calculates an AQI estimate from PM2.5, PM10, NO2, and O3, then adds lag and rolling features.
- **Model training:** Trains a Random Forest regressor or an LSTM model and reports evaluation metrics.
- **Dashboard:** Displays AQI estimates, pollutant levels, historical trends, and an LSTM forecast.

The AQI calculation in this project is an estimate based on a weighted combination of the four pollutants; it is not an official AQI calculation.

## Repository layout

```text
dashboard/
  app.py                  Streamlit dashboard
src/
  data_preprocessing.py   Data loading, cleaning, and outlier handling
  feature_engineering.py  AQI estimate and time-series features
  train_ml_model.py       Random Forest training and evaluation
  train_lstm_model.py     LSTM training and evaluation
  utils.py                AQI category labels
```

## Getting started

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages used by the training scripts and dashboard:

```bash
python -m pip install pandas numpy scikit-learn joblib tensorflow streamlit streamlit-autorefresh plotly
```

Place the input data at `data/ancona_data.csv`. The scripts expect a `Date` column and pollutant columns named `PM2.5`, `PM10`, `NO2`, and `O3`. The data file is not included in this repository.

## Run the scripts

Run these commands from the repository root:

```bash
python src/data_preprocessing.py
python src/feature_engineering.py
```

The preparation script prints checks on the cleaned data; it does not save a new CSV. The training scripts each read the original CSV and perform their own feature preparation:

```bash
python -c "from pathlib import Path; Path('models').mkdir(exist_ok=True)"
python src/train_ml_model.py
python src/train_lstm_model.py
```

The models are saved under `models/` as `rf_model.pkl` and `lstm_model.keras`. Train both models before starting the dashboard:

```bash
python -m streamlit run dashboard/app.py
```

The dashboard reads the same local CSV and the saved model files. Its Random Forest card currently uses fixed demo feature values, so treat that reading as a demonstration rather than a live measurement. The LSTM forecast is generated from the historical data in the CSV.

## Notes

- The repository does not include the dataset, trained model files, a `requirements.txt`, or a separate prediction script.
- No license file is currently included.
