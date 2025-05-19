import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset (adjust path if needed)
@st.cache_data
def load_data():
    df = pd.read_csv("weatherHistory.csv")
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df = df[df['Formatted Date'].dt.year == 2016].copy()
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = load_data()

st.title("Weather Forecast Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
month_filter = st.sidebar.multiselect(
    "Select Month(s):",
    options=df['Formatted Date'].dt.month_name().unique(),
    default=df['Formatted Date'].dt.month_name().unique()
)

df_filtered = df[df['Formatted Date'].dt.month_name().isin(month_filter)]

# Plot Temperature with Rolling Average
st.header("Temperature Over Time with Rolling Average")

df_filtered.set_index('Formatted Date', inplace=True)
df_filtered['Rolling Mean'] = df_filtered['Temperature (C)'].rolling(window=7).mean()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_filtered.index, df_filtered['Temperature (C)'], label='Daily Temp', alpha=0.4)
ax.plot(df_filtered.index, df_filtered['Rolling Mean'], label='7-Day Rolling Avg', color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

df_filtered.reset_index(inplace=True)

# Show basic stats
st.header("Basic Statistics")
st.write(df_filtered.describe())

# Optional: Show raw data toggle
if st.checkbox("Show raw data"):
    st.write(df_filtered)


# Calculate anomalies (inside your load_data or after df is loaded)
df['Temp_zscore'] = (df['Temperature (C)'] - df['Temperature (C)'].mean()) / df['Temperature (C)'].std()
threshold = 3
df['Anomaly'] = np.where((df['Temp_zscore'] > threshold) | (df['Temp_zscore'] < -threshold), True, False)

# Filter by month again if needed
df_filtered = df[df['Formatted Date'].dt.month_name().isin(month_filter)]

st.header("Temperature Over Time with Anomalies")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_filtered['Formatted Date'], df_filtered['Temperature (C)'], label='Temperature', alpha=0.5)
ax.scatter(df_filtered[df_filtered['Anomaly']]['Formatted Date'], df_filtered[df_filtered['Anomaly']]['Temperature (C)'], 
           color='red', label='Anomalies', s=50)
ax.set_xlabel("Date")
ax.set_ylabel("Temperature (C)")
ax.legend()
ax.grid(True)
st.pyplot(fig)



# Add more sections for clustering, anomalies, or prediction results here!

