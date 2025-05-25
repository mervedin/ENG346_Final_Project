import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide")
st.title("🚲 UCI Bike Sharing Dataset EDA")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/hour.csv")
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['datetime'] = df['dteday'] + pd.to_timedelta(df['hr'], unit='h')
    return df

df = load_data()

# --- Adjustable time window ---
st.subheader("📅 Bike Count Over Time")
start_date, end_date = st.date_input("Select time range", [df['dteday'].min(), df['dteday'].max()])
filtered_df = df[(df['dteday'] >= pd.to_datetime(start_date)) & (df['dteday'] <= pd.to_datetime(end_date))]

fig, ax = plt.subplots(figsize=(9,4))
sns.lineplot(data=filtered_df, x='datetime', y='cnt', ax=ax)
ax.set(title="Bike Rentals Over Time")
st.pyplot(fig)

st.markdown("""---""")
st.subheader("🔍 Hourly Bike Usage Patterns")

# WEEKDAY vs WEEKEND
fig, ax = plt.subplots(figsize=(10,5))
sns.pointplot(data=df, x='hr', y='cnt', hue='weekday', ax=ax)
ax.set(title='Count of Bikes During Weekdays vs Weekends (All Users)')
st.pyplot(fig)
st.markdown("- Peak usage for registered users is around **8 AM and 5-6 PM** on weekdays (commute hours).")
st.markdown("- Weekends show higher usage during **afternoon hours** (more casual trips).")

# CASUAL USERS
fig, ax = plt.subplots(figsize=(10,5))
sns.pointplot(data=df, x='hr', y='casual', hue='weekday', ax=ax)
ax.set(title='Unregistered Users (Casual) — Hourly by Weekday')
st.pyplot(fig)
st.markdown("- Casual users mostly rent bikes in the **afternoons and evenings**, especially on weekends.")

# REGISTERED USERS
fig, ax = plt.subplots(figsize=(10,5))
sns.pointplot(data=df, x='hr', y='registered', hue='weekday', ax=ax)
ax.set(title='Registered Users — Hourly by Weekday')
st.pyplot(fig)
st.markdown("- Registered users dominate during **rush hours** on weekdays. Typical commuting behavior.")

# WEATHER EFFECT
fig, ax = plt.subplots(figsize=(10,5))
sns.pointplot(data=df, x='hr', y='cnt', hue='weathersit', ax=ax)
ax.set(title='Bike Count by Hour and Weather Condition')
st.pyplot(fig)
st.markdown("- Clear weather (1) has the highest counts. Usage drops significantly in bad weather (3 and 4).")

# SEASONAL EFFECT
fig, ax = plt.subplots(figsize=(10,5))
sns.pointplot(data=df, x='hr', y='cnt', hue='season', ax=ax)
ax.set(title='Bike Count by Hour and Season')
st.pyplot(fig)
st.markdown("- Summer and fall have higher usage throughout the day. Winter shows reduced activity.")

# MONTHLY USAGE
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=df, x='mnth', y='cnt', ax=ax)
ax.set(title='Monthly Bike Rental Count')
st.pyplot(fig)
st.markdown("- Usage rises from spring, peaks in **summer/fall**, and drops in **winter**.")

# WEEKDAY USAGE
fig, ax = plt.subplots(figsize=(10,5))
sns.barplot(data=df, x='weekday', y='cnt', ax=ax)
ax.set(title='Average Bike Rentals per Weekday')
st.pyplot(fig)
st.markdown("- Slightly higher usage on weekdays, but weekend activity is still strong (especially from casuals).")

# TEMP & HUMIDITY RELATION
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20,6))
sns.regplot(x='temp', y='cnt', data=df, ax=ax1)
ax1.set(title='Relation Between Temperature and Total Count')
sns.regplot(x='hum', y='cnt', data=df, ax=ax2)
ax2.set(title='Relation Between Humidity and Total Count')
st.pyplot(fig)
st.markdown("- As temperature increases, bike usage also increases — up to a point.")
st.markdown("- High humidity tends to **slightly lower** the count — possibly due to discomfort.")

st.markdown("---")
st.header("🔮 Predict Next Week's Bike Rentals")

# Load trained model
with gzip.open("./code/bike_model_compressed.pkl.gz", "rb") as f:
    model = pickle.load(f)

# Use model for prediction
features = ['hr', 'weekday', 'workingday', 'temp', 'atemp', 'hum', 'windspeed', 'season', 'mnth']
target = 'cnt'

X = df[features]
y = df[target]

# Model performance (optional)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_preds = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
st.write(f"Model RMSE on validation set: **{rmse:.2f}**")

# Future prediction (same as before)
import datetime as dt
last_date = df['datetime'].max()
future_hours = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=168, freq='h')

future_df = pd.DataFrame({
    'datetime': future_hours,
    'hr': future_hours.hour,
    'weekday': future_hours.weekday,
    'workingday': [1 if d.weekday() < 5 and d not in [5,6] else 0 for d in future_hours],
    'mnth': future_hours.month,
    'season': ((future_hours.month % 12 + 3) // 3),
})
future_df['temp'] = df['temp'].mean()
future_df['atemp'] = df['atemp'].mean()
future_df['hum'] = df['hum'].mean()
future_df['windspeed'] = df['windspeed'].mean()

future_preds = model.predict(future_df[features])
future_df['cnt_predicted'] = future_preds

st.subheader("📆 Last 2 Weeks and Next 7 Days: Bike Rental Forecast")

# Get last 2 weeks of actual data (336 hours)
history_df = df.sort_values('datetime').copy()
last_two_weeks = history_df.iloc[-336:][['datetime', 'cnt']].copy()
last_two_weeks.rename(columns={'cnt': 'cnt_actual'}, inplace=True)

# Prepare prediction DataFrame
future_df_plot = future_df[['datetime', 'cnt_predicted']].copy()

# Combine for plotting
combined_df = pd.concat([
    last_two_weeks.set_index('datetime'),
    future_df_plot.set_index('datetime')
], axis=1).reset_index()

# Plot both
fig, ax = plt.subplots(figsize=(9,4))
sns.lineplot(data=combined_df, x='datetime', y='cnt_actual', label='Actual (Last 2 Weeks)', color='blue')
sns.lineplot(data=combined_df, x='datetime', y='cnt_predicted', label='Predicted (Next 7 Days)', color='orange')
ax.set(title="Bike Rentals: Last 2 Weeks (Actual) and Next 7 Days (Predicted)")
plt.xticks(rotation=45)
st.pyplot(fig)