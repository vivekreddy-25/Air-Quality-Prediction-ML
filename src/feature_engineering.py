import pandas as pd#for working with dataframes, reading CSV files, manipulating columns.
import numpy as np #for numerical computations (used here for calculations like rolling mean/std).

def compute_aqi(df):
    """
    Compute AQI using weighted pollutants and scale to realistic range.
    This gives AQI values roughly comparable to official AQI scale (0-500)
    """
    # Weighted sum of pollutants
    df['AQI_raw'] = (0.35 * df['PM2.5'] +0.25 * df['PM10'] +0.20 * df['NO2'] + 0.20 * df['O3'])
    
    # Multiplying by 3 increases the magnitude, otherwise AQI_raw would be 
    # too small to reflect meaningful AQI categories.
    df['AQI'] = df['AQI_raw'] * 3

    #Removes extreme abnormal values Keeps only 99% of useful data to prevent unrealistic AQI values.
    df['AQI'] = df['AQI'].clip(lower=df['AQI'].quantile(0.01),upper=df['AQI'].quantile(0.99))
    
    return df

def create_features(df):
    """
    Create time-series lag and rolling features for LSTM input.
    """
    #Lag means previous time values Feature	Meaning lag1-AQI at previous hour ,lag2-AQI two hours before
    df['lag1'] = df['AQI'].shift(1)
    df['lag2'] = df['AQI'].shift(2)
    #Creates rolling statistics over 3 periods:
    # rolling_mean_3 → average AQI of the current and previous 2 hours
    # rolling_std_3 → standard deviation of AQI over 3 hours
    df['rolling_mean_3'] = df['AQI'].rolling(3).mean()
    df['rolling_std_3'] = df['AQI'].rolling(3).std()
    
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data/ancona_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Compute AQI and features
    df = compute_aqi(df)
    df = create_features(df)
    
    # Display sample
    print(df[['Date','AQI','lag1','lag2','rolling_mean_3','rolling_std_3']].head())
