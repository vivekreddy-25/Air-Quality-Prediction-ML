import pandas as pd #mports pandas library for:Reading CSV files,Data cleaning,Data preprocessing

def load_and_clean_data(path):
    df = pd.read_csv(path) #Loads CSV file into DataFrame
    
    # Convert 'Date' column to datetime required for time series analysis 
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date in chronological order 
    df = df.sort_values('Date')
    
    # Fill missing values (forward fill, then backward fill if any remain)
    df = df.ffill().bfill()
    
    return df

def handle_outliers(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers at the bounds as it Reduces skewness and Improve stability
        #Replacing extreme outlier values with lower or upper boundary values instead of deleting them.
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df

def prepare_dataset(path):
    df = load_and_clean_data(path)
    df = handle_outliers(df)
    return df

if __name__ == "__main__":
    # Load, clean, and handle outliers
    df = prepare_dataset("data/ancona_data.csv")
    
    # Check if any missing values remain
    print("Missing values after cleaning:\n", df.isnull().sum())
    
    # Check outliers (optional check)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        print(f"{col}: {outliers.shape[0]} outliers remaining")
    
    # Dataset is now ready 
    print("Cleaned dataset is ready to use.")
    print(df.shape)

