import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path

# File paths
base_dir = Path(__file__).parent
model_path = base_dir / 'notebooks' / 'best_model.pkl'
data_path = base_dir / 'notebooks' / 'processed_crypto_with_target.csv'

# Load model
model = None
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error: '{model_path}' not found. Please ensure the model file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load sample data to get feature names
try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    st.error(f"Error: '{data_path}' not found. Please ensure the data file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Ensure 'date' column is datetime
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])

# Define numerical columns for scaling
numerical_cols = [
    col for col in df.columns
    if df[col].dtype != 'object' and
    col not in ['Volatility_7', 'Volatility_7_target'] and
    not col.startswith('crypto_name_') and
    col not in ['year', 'month', 'day', 'day_of_week']
]
scaler = StandardScaler()
if numerical_cols:
    scaler.fit(df[numerical_cols])
else:
    st.error("No numerical columns found for scaling.")
    st.stop()



# Define unscaled ranges for user inputs (0 to 10,000,000,000)
unscaled_ranges = {
    'open': (0.0, 10000000000.0, 50000.0),
    'high': (0.0, 10000000000.0, 51000.0),
    'low': (0.0, 10000000000.0, 49000.0),
    'close': (0.0, 10000000000.0, 50000.0),
    'volume': (0.0, 1000000000000.0, 500000000.0),
    'marketCap': (0.0, 1000000000000.0, 100000000000.0),
    'ATR_14': (0.0, 10000000000.0, 5000.0),
    'Bollinger_Width': (0.0, 10000000000.0, 5000.0),
}

# Function to calculate volatility
def calculate_volatility(df, crypto, date, window):
    df_crypto = df[df[f'crypto_name_{crypto}'] == 1].copy()
    if 'date' in df_crypto.columns and len(df_crypto) > window:
        df_crypto['Log_Return'] = np.log(df_crypto['close'] / df_crypto['open'].shift(1))
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=window)
        window_data = df_crypto[(df_crypto['date'] >= start_date) & (df_crypto['date'] <= end_date)]
        if len(window_data) >= window // 2:
            return window_data['Log_Return'].std()
    return df[f'Volatility_{window}'].mean() if f'Volatility_{window}' in df.columns else 0.05

# Function to calculate ATR_14
def calculate_atr(df, crypto, date, window=14):
    df_crypto = df[df[f'crypto_name_{crypto}'] == 1].copy()
    if 'date' in df_crypto.columns and len(df_crypto) > window:
        df_crypto['TR'] = np.maximum(
            df_crypto['high'] - df_crypto['low'],
            np.maximum(
                abs(df_crypto['high'] - df_crypto['close'].shift(1)),
                abs(df_crypto['low'] - df_crypto['close'].shift(1))
            )
        )
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=window)
        window_data = df_crypto[(df_crypto['date'] >= start_date) & (df_crypto['date'] <= end_date)]
        if len(window_data) >= window // 2:
            return window_data['TR'].mean()
    return df['ATR_14'].mean() if 'ATR_14' in df.columns else 5000.0

# Function to calculate Bollinger_Width
def calculate_bollinger_width(df, crypto, date, window=14):
    df_crypto = df[df[f'crypto_name_{crypto}'] == 1].copy()
    if 'date' in df_crypto.columns and len(df_crypto) > window:
        end_date = pd.to_datetime(date)
        start_date = end_date - pd.Timedelta(days=window)
        window_data = df_crypto[(df_crypto['date'] >= start_date) & (df_crypto['date'] <= end_date)]
        if len(window_data) >= window // 2:
            rolling_mean = window_data['close'].rolling(window=window).mean()
            rolling_std = window_data['close'].rolling(window=window).std()
            upper_band = rolling_mean + 2 * rolling_std
            lower_band = rolling_mean - 2 * rolling_std
            return (upper_band.iloc[-1] - lower_band.iloc[-1]) / rolling_mean.iloc[-1]
    return df['Bollinger_Width'].mean() if 'Bollinger_Width' in df.columns else 5000.0

# Function to simulate price paths
def simulate_price_paths(start_price, volatility, days=7, num_paths=100):
    daily_vol = volatility / np.sqrt(7)
    dates = [date_input + timedelta(days=i) for i in range(days + 1)]
    paths = []
    for _ in range(num_paths):
        prices = [start_price]
        for _ in range(days):
            log_return = np.random.normal(0, daily_vol)
            next_price = prices[-1] * np.exp(log_return)
            prices.append(next_price)
        paths.append(prices)
    return dates, paths

st.title("Cryptocurrency Volatility Prediction")
st.markdown("### Enter the cryptocurrency data:")

# Define expected feature columns
if hasattr(model, 'feature_names_in_'):
    feature_columns = list(model.feature_names_in_)
else:
    non_input_columns = ['Volatility_7_target', 'Volatility_7'] if 'Volatility_7' in df.columns else ['Volatility_7_target']
    feature_columns = [col for col in df.columns if col not in non_input_columns]


# Dynamic crypto list
crypto_columns = [col for col in df.columns if col.startswith('crypto_name_')]
crypto_options = [col.replace('crypto_name_', '') for col in crypto_columns]
if not crypto_options:
    st.error("No cryptocurrencies found in the dataset.")
    st.stop()

# Collect user inputs
user_input = {}
selected_crypto = st.selectbox("Cryptocurrency", crypto_options)

# One-hot encode crypto_name
for crypto in crypto_options:
    col_name = f"crypto_name_{crypto}"
    user_input[col_name] = 1 if crypto == selected_crypto else 0

# Date input
date_input = st.date_input("Select Date", value=datetime.today())
user_input['date'] = date_input

# Extract date features
date_obj = pd.to_datetime(user_input['date'])
user_input['year'] = date_obj.year
user_input['month'] = date_obj.month
user_input['day'] = date_obj.day
user_input['day_of_week'] = date_obj.weekday()

# Two-column layout for numerical inputs
col1, col2 = st.columns(2)
input_features = ['open', 'high', 'low', 'close', 'volume', 'marketCap', 'ATR_14', 'Bollinger_Width']
for i, feature in enumerate(input_features):
    with (col1 if i % 2 == 0 else col2):
        if feature in unscaled_ranges:
            min_val, max_val, default_val = unscaled_ranges[feature]
            user_input[feature] = st.number_input(
                f"{feature}",
                min_value=min_val,
                max_value=max_val,
                value=float(default_val),
                placeholder=f"e.g. {default_val}",
                key=feature
            )
            if user_input[feature] < min_val or user_input[feature] > max_val:
                st.warning(f"{feature} value {user_input[feature]} is outside range ({min_val}, {max_val}).")

# Calculate derived features
if 'open' in user_input and 'close' in user_input and user_input['open'] > 0:
    user_input['Log_Return'] = np.log(user_input['close'] / user_input['open'])
else:
    user_input['Log_Return'] = df['Log_Return'].mean() if 'Log_Return' in df.columns else 0.0

if all(f in user_input for f in ['high', 'low', 'close']):
    user_input['TR'] = max(
        user_input['high'] - user_input['low'],
        abs(user_input['high'] - user_input['close']),
        abs(user_input['low'] - user_input['close'])
    )
else:
    user_input['TR'] = df['TR'].mean() if 'TR' in df.columns else 0.0

if 'volume' in user_input and 'marketCap' in user_input and user_input['marketCap'] > 0:
    user_input['Liquidity_Ratio'] = user_input['volume'] / user_input['marketCap']
else:
    user_input['Liquidity_Ratio'] = df['Liquidity_Ratio'].mean() if 'Liquidity_Ratio' in df.columns else 0.0

# Calculate volatility features if needed
if 'Volatility_14' in numerical_cols or 'Volatility_14' in feature_columns:
    user_input['Volatility_14'] = calculate_volatility(df, selected_crypto, date_input, 14)
   
if 'Volatility_30' in numerical_cols or 'Volatility_30' in feature_columns:
    user_input['Volatility_30'] = calculate_volatility(df, selected_crypto, date_input, 30)

# Calculate ATR_14 and Bollinger_Width if needed
if 'ATR_14' in numerical_cols or 'ATR_14' in feature_columns:
    user_input['ATR_14'] = calculate_atr(df, selected_crypto, date_input, 14)
if 'Bollinger_Width' in numerical_cols or 'Bollinger_Width' in feature_columns:
    user_input['Bollinger_Width'] = calculate_bollinger_width(df, selected_crypto, date_input, 14)

# Convert user input into DataFrame
input_df = pd.DataFrame([user_input])

# Add missing numerical features before scaling
missing_numerical = [col for col in numerical_cols if col not in input_df.columns]
for col in missing_numerical:
    input_df[col] = df[col].mean() if col in df.columns else 0.0



# Scale numerical features
if numerical_cols:
    try:
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    except Exception as e:
        st.error(f"Error scaling numerical features: {str(e)}")
        st.stop()

# Ensure all expected feature columns are present
missing_features = [col for col in feature_columns if col not in input_df.columns]
for col in missing_features:
    input_df[col] = 0

# Check feature count
if len(input_df[feature_columns].columns) != len(feature_columns):
    st.error(f"Feature mismatch: Model expects {len(feature_columns)} features, but {len(input_df[feature_columns].columns)} provided.")
    st.stop()

# Make prediction button
if st.button("Predict Volatility"):
    try:
        # Select model input features
        model_input = input_df[feature_columns]
        prediction = model.predict(model_input)[0]
        st.success(f"Predicted Volatility (7-day) for {selected_crypto}: {prediction:.4f}")

        # Display candlestick chart with volatility annotation
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in input_df.columns for col in ohlc_cols):
            # Inverse transform all numerical features
            numerical_values = input_df[numerical_cols].values
            numerical_unscaled = scaler.inverse_transform(numerical_values)
            numerical_unscaled_df = pd.DataFrame(numerical_unscaled, columns=numerical_cols)
            ohlc_df = numerical_unscaled_df[ohlc_cols]

            fig = go.Figure(data=[go.Candlestick(
                x=[user_input['date']],
                open=[ohlc_df['open'][0]],
                high=[ohlc_df['high'][0]],
                low=[ohlc_df['low'][0]],
                close=[ohlc_df['close'][0]],
                name="OHLC",
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            fig.add_annotation(
                x=user_input['date'],
                y=ohlc_df['high'][0] * 1.05,
                text=f"Predicted Volatility: {prediction:.4f}",
                showarrow=True,
                arrowhead=1,
                ax=20,
                ay=-30,
                bgcolor="rgba(255, 255, 255, 0.8)",
                font=dict(size=12, color="black")
            )
            fig.update_layout(
                title=f"{selected_crypto} OHLC Candlestick",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                yaxis=dict(gridcolor='lightgrey', tickformat=",.2f"),
                xaxis=dict(gridcolor='lightgrey', tickformat="%Y-%m-%d")
            )
            st.plotly_chart(fig)

            # Simulate and plot 7-day price paths
            start_price = ohlc_df['close'][0]
            dates, price_paths = simulate_price_paths(start_price, prediction)
            fig_paths = go.Figure()

            # Add candlestick for input date
            fig_paths.add_trace(go.Candlestick(
                x=[user_input['date']],
                open=[ohlc_df['open'][0]],
                high=[ohlc_df['high'][0]],
                low=[ohlc_df['low'][0]],
                close=[ohlc_df['close'][0]],
                name="OHLC",
                increasing_line_color='green',
                decreasing_line_color='red'
            ))

            # Add simulated price paths
            for i, path in enumerate(price_paths):
                fig_paths.add_trace(go.Scatter(
                    x=dates,
                    y=path,
                    mode='lines',
                    line=dict(width=1, color='rgba(100, 100, 255, 0.2)'),
                    name=f"Path {i+1}",
                    showlegend=False
                ))

            # Calculate mean path and 95% confidence interval
            daily_vol = prediction / np.sqrt(7)
            mean_path = [start_price] * len(dates)
            upper_bound = [start_price * np.exp(1.96 * daily_vol * np.sqrt(i)) for i in range(len(dates))]
            lower_bound = [start_price * np.exp(-1.96 * daily_vol * np.sqrt(i)) for i in range(len(dates))]
            fig_paths.add_trace(go.Scatter(
                x=dates,
                y=mean_path,
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name="Mean Price"
            ))
            fig_paths.add_trace(go.Scatter(
                x=dates + dates[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name="95% Confidence Interval"
            ))

            fig_paths.update_layout(
                title=f"{selected_crypto} 7-Day Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                yaxis=dict(gridcolor='lightgrey', tickformat=",.2f"),
                xaxis=dict(gridcolor='lightgrey', tickformat="%Y-%m-%d"),
                margin=dict(l=50, r=50, t=100, b=50),
                hovermode="x unified"
            )
            st.plotly_chart(fig_paths)
        else:
            st.info("OHLC data not available for candlestick plot.")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")