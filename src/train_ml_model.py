import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_engineering import compute_aqi, create_features


# Load dataset

print("Loading dataset...")
df = pd.read_csv("data/ancona_data.csv")
df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime

# Feature Engineering

print("Engineering features...")
df = compute_aqi(df)  # Compute AQI (scaled realistically, x3)
df = create_features(df)  # Generate lag and rolling features for time series


# Prepare Features & Target

# Use fewer lag features for more realistic prediction (avoid overfitting)
#lag1-AQI at previous hour
##rolling_mean_3 → average AQI of the current and previous 2 hours
#rolling_std_3 → standard deviation of AQI over 3 hours
X = df[['lag1', 'rolling_mean_3', 'rolling_std_3']]

# Add small Gaussian noise to AQI to simulate real-world fluctuations
np.random.seed(42)
y = df['AQI'] + np.random.normal(0, 2, size=len(df))

# Split dataset into training and testing sets (larger test set for realism)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# Train Random Forest Regressor

print("Training Random Forest model...")
model = RandomForestRegressor(
    n_estimators=300,   # 300 decision trees in the forest
    max_depth=12,       # maximum depth of each tree to prevent overfitting
    random_state=42,    # ensures reproducibility
    n_jobs=-1           # use all CPU cores for faster training
)

model.fit(X_train, y_train)  # Train model on training set

# Evaluate Model

print("Evaluating model...")
pred = model.predict(X_test)  # Predict on test set

mae = mean_absolute_error(y_test, pred)  # Mean Absolute Error
rmse = mean_squared_error(y_test, pred) ** 0.5  # Root Mean Squared Error
r2 = r2_score(y_test, pred)  # R-squared value

print("\nModel Performance:")
print("MAE  :", round(mae, 3))
print("RMSE :", round(rmse, 3))
print("R2   :", round(r2, 4)) 

# Save Model

joblib.dump(model, "models/rf_model.pkl")  # Save trained Random Forest model
print("\nModel saved successfully in models/rf_model.pkl")
