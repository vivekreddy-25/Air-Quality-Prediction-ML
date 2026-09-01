import pandas as pd #for handling CSV and dataframes.
import numpy as np #for numerical operations and array reshaping.
from tensorflow.keras.models import Sequential  # Keras model where layers are stacked sequentially.
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input  #LSTM, Dense, Dropout, 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#Input → layers used for the neural network:
#LSTM → for time-series sequences.
# Dense → fully connected layer for output.
#  Dropout → regularization to prevent overfitting. 
# Input → defines input shape explicitly.


from tensorflow.keras.optimizers import Adam #type:ignore optimizer used to train the network.                                                                      
from sklearn.preprocessing import MinMaxScaler #scales data to a [0,1] range.

print("Loading dataset...")
df = pd.read_csv("data/ancona_data.csv")

#Converts the Date column to datetime objects
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df.ffill()

# Compute AQI (scaled to realistic range)
df['AQI_raw'] = (
    0.35 * df['PM2.5'] +
    0.25 * df['PM10'] +
    0.20 * df['NO2'] +
    0.20 * df['O3']
)

# Scale AQI to approximate 0-300 range
df['AQI'] = df['AQI_raw']*3

df = df.dropna()

# Clip extreme outliers
df['AQI'] = df['AQI'].clip(lower=df['AQI'].quantile(0.01), upper=df['AQI'].quantile(0.99))

# Log transform for stability
aqi = np.log1p(df['AQI'].values.reshape(-1,1))

scaler = MinMaxScaler()
scaled = scaler.fit_transform(aqi)

def make_sequences(data, step=24):
    X, y = [], []
    for i in range(step, len(data)):
        X.append(data[i-step:i])
        y.append(data[i])
    return np.array(X), np.array(y)

#Creates sliding window sequences for time-series prediction:
#step=24 → uses previous 24 hours (or rows) to predict the next AQI value.
#X → input sequences of length 24.
#y → target AQI value at the next timestep.

X, y = make_sequences(scaled)
#Generates input/output arrays ready for LSTM:

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
#Splits data into 80% training and 20% testing.
#Time-series split → preserves chronological order.

print("Training Stable LSTM model...")

model = Sequential([Input(shape=(X.shape[1],1)),LSTM(32, return_sequences=True),Dropout(0.2),LSTM(16),Dense(1)])
#LSTM model architecture:
#Input(shape=(24,1)) → 24 timesteps, 1 feature (AQI).
#LSTM(32, return_sequences=True) → first LSTM layer outputs sequence to feed next LSTM.
#Dropout(0.2) → randomly zeroes 20% of neurons for regularization.
#LSTM(16) → second LSTM layer returns only last output (default return_sequences=False).
#Dense(1) → output layer predicts single AQI value.

model.compile(optimizer=Adam(learning_rate=0.0005),loss='mse')
#Compiles model with:
#Adam optimizer → adaptive learning rate.
#MSE loss → suitable for regression tasks (predicting AQI values).

model.fit(X_train, y_train,epochs=12,batch_size=256,validation_split=0.1,verbose=1)
#Trains the model:
#epochs=12 → 12 passes over the training data.
#batch_size=256 → 256 sequences per batch.
#validation_split=0.1 → 10% of training data used for validation.
#verbose=1 → shows training progress.

model.save("models/lstm_model.keras")
#Saves the trained LSTM model to disk for later use (e.g., Streamlit or deployment).
print("\nStable LSTM Model saved to models/lstm_model.keras")
# Predict on test set
y_pred = model.predict(X_test)

# Inverse scaling and log-transform to get AQI in original scale
y_test_orig = np.expm1(scaler.inverse_transform(y_test))
y_pred_orig = np.expm1(scaler.inverse_transform(y_pred))

# Calculate metrics
mae = mean_absolute_error(y_test_orig, y_pred_orig)
from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test_orig, y_pred_orig) ** 0.5
print("RMSE:", rmse)  # squared=False gives RMSE directly
r2 = r2_score(y_test_orig, y_pred_orig)

print("\nLSTM Model Evaluation Metrics:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")

# Using this model in Streamlit
# To get the predicted AQI in the original scale, do:
# pred_scaled = model.predict(X_input)[0]
# pred_aqi = np.expm1(scaler.inverse_transform([[pred_scaled]]))[0][0]
# Then you can display pred_aqi in Streamlit
