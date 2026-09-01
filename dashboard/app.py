import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
import streamlit as st
import pandas as pd
import joblib
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import compute_aqi, create_features
from src.utils import aqi_category
from streamlit_autorefresh import st_autorefresh

# PAGE CONFIG 
st.set_page_config(page_title="AI Air Quality Monitoring", layout="wide")
st.title("🌍 AI Powered Air Quality Monitoring System")

# Auto refresh every 10 seconds
st_autorefresh(interval=10000, key="refresh")

#  LOAD DATA 
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ancona_data.csv')
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').ffill()

# Feature Engineering
df = compute_aqi(df)
df = create_features(df)

# LOAD MODELS 
# Random Forest (Current AQI)
rf_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'rf_model.pkl')
rf_model = joblib.load(rf_path)

# LSTM (Future AQI)
lstm_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm_model.keras')
lstm_model = load_model(lstm_path)

#  RF PREDICTION 
latest = df.iloc[-1]

# Use exact feature names from trained model to avoid errors
features = rf_model.feature_names_in_
X_rf = latest[features].to_frame().T

# Simulate AQI ~50 for demo by overriding features
# You can adjust these numbers to change predicted AQI
X_rf_sim = X_rf.copy()
for col in X_rf_sim.columns:
    X_rf_sim[col] = 50  # simple simulation for all features

pred = rf_model.predict(X_rf_sim)[0]

#  LSTM FUTURE PREDICTION 
def predict_next_24hrs_lstm(df, steps=24):
    aqi = df['AQI'].values.reshape(-1,1)
    aqi = np.log1p(aqi)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(aqi)

    last_seq = scaled[-24:]
    preds = []

    for _ in range(steps):
        X = last_seq.reshape(1,24,1)
        p = lstm_model.predict(X, verbose=0)[0][0]
        preds.append(p)
        last_seq = np.append(last_seq[1:], [[p]], axis=0)

    preds = np.expm1(scaler.inverse_transform(np.array(preds).reshape(-1,1)))
    return preds.flatten()

future_preds = predict_next_24hrs_lstm(df)

#  DASHBOARD LAYOUT 
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Current AQI", round(pred))  # rounded to integer

with c2:
    st.metric(" Health", aqi_category(pred))  # shorter label

with c3:
    st.metric(" Last Updated", datetime.now().strftime("%d %b %Y, %H:%M"))

with c4:
    st.metric(" Next Hour AQI", round(future_preds[0]))

#  AQI GAUGE 
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pred,
    title={'text': "Live AQI Meter"},
    gauge={
        'axis': {'range': [0, 300]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 100], 'color': "yellow"},
            {'range': [100, 150], 'color': "orange"},
            {'range': [150, 200], 'color': "red"},
            {'range': [200, 300], 'color': "darkred"}
        ]
    }
))
st.plotly_chart(fig_gauge, use_container_width=True)

#  AQI TREND 
st.subheader(" Historical AQI Trend")
fig_trend = px.line(df.tail(500), x='Date', y='AQI', title="AQI Trend")  # reduced points
st.plotly_chart(fig_trend, use_container_width=True)

#  LSTM FORECAST 
st.subheader(" Next 24 Hours AQI Forecast (LSTM)")
future_time = pd.date_range(start=df['Date'].iloc[-1], periods=24, freq='H')
fig_forecast = px.line(x=future_time, y=future_preds, labels={'x':'Time','y':'AQI'},title="LSTM Based Future AQI Prediction")
st.plotly_chart(fig_forecast, use_container_width=True)

#  HEATMAP 
st.subheader(" Daily & Hourly AQI Heatmap")
df['hour'] = df['Date'].dt.hour
df['day'] = df['Date'].dt.day
pivot = df.pivot_table(values='AQI', index='day', columns='hour', aggfunc='mean').tail(15)
fig_heat = px.imshow(pivot, aspect='auto', color_continuous_scale='Turbo')
st.plotly_chart(fig_heat, use_container_width=True)

# POLLUTANT CONTRIBUTION 
st.subheader("Pollutant Contribution")
poll = latest[['PM2.5','PM10','NO2','O3']]
fig_bar = px.bar(x=poll.index, y=poll.values, title="Current Pollutant Levels")
st.plotly_chart(fig_bar, use_container_width=True)
