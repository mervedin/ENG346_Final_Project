# model.py

import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_and_save_model(csv_path: str, model_path: str = "./code/bike_model.pkl"):
    df = pd.read_csv(csv_path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')

    # Features and target
    features = ['hr', 'weekday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed', 'season', 'mnth']
    target = 'cnt'

    X = df[features]
    y = df[target]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Validation RMSE
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse:.2f}")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    with open("./code/bike_model.pkl", "rb") as f_in:
        with gzip.open("./code/bike_model_compressed.pkl.gz", "wb") as f_out:
            f_out.write(f_in.read())

if __name__ == "__main__":
    train_and_save_model("./data/hour.csv")