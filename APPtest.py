import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go

# Title and user input
st.title('STOCK PRICE ANALYSIS')

# User input for date range and stock symbol
start_date_input = st.date_input('Enter Start Date', pd.to_datetime('2010-01-01'))
end_date_input = st.date_input('Enter End Date', pd.to_datetime('2019-12-31'))

if start_date_input >= end_date_input:
    st.error('Start date should be before end date.')
else:
    start = start_date_input.strftime('%Y-%m-%d')
    end = end_date_input.strftime('%Y-%m-%d')

    us_ip = st.text_input('ENTER STOCK:', 'AAPL')
    DF = yf.download(us_ip, start=start, end=end)

    st.subheader(f'DATA from {start} - {end}')
    st.write(DF.describe())

    # Create a Streamlit figure for Plotly graph
    st.subheader("CHART OF CLOSING PRICES VS TIME")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF.index, y=DF['Close'], mode='lines', name='Closing Prices', line=dict(color='royalblue')))
    fig.update_layout(title='Closing Prices Over Time', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.subheader("CHART OF CLOSING PRICES VS TIME WITH MOVING AVERAGE OF 100")
    DF['100-Day Moving Average'] = DF['Close'].rolling(100).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF.index, y=DF['Close'], mode='lines', name='Closing Prices', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=DF.index, y=DF['100-Day Moving Average'], mode='lines', name='100-Day Moving Average', line=dict(color='orange')))
    fig.update_layout(title='Closing Prices and 100-Day Moving Average Over Time', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    st.subheader("CHART OF CLOSING PRICES VS TIME WITH MOVING AVERAGE OF 100&200")
    DF['200-Day Moving Average'] = DF['Close'].rolling(200).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF.index, y=DF['Close'], mode='lines', name='Closing Prices', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=DF.index, y=DF['100-Day Moving Average'], mode='lines', name='100-Day Moving Average', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=DF.index, y=DF['200-Day Moving Average'], mode='lines', name='200-Day Moving Average', line=dict(color='purple')))
    fig.update_layout(title='Closing Prices and Moving Averages Over Time', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

    # Splitting data into training and testing
    train_size = int(len(DF) * 0.7)
    train_data, test_data = DF[:train_size]['Close'], DF[train_size:]['Close']

    # Prepare the data
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
    scaled_test = scaler.transform(test_data.values.reshape(-1, 1))

    seq_length = 100
    X_train, y_train = [], []
    for i in range(len(scaled_train) - seq_length):
        X_train.append(scaled_train[i:i + seq_length])
        y_train.append(scaled_train[i + seq_length])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test, y_test = [], []
    for i in range(len(scaled_test) - seq_length):
        X_test.append(scaled_test[i:i + seq_length])
        y_test.append(scaled_test[i + seq_length])
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Build and train the model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

    # Predict and inverse transform
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_predictions = scaler.inverse_transform(train_predictions)
    y_train = scaler.inverse_transform(y_train)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test = scaler.inverse_transform(y_test)

    # Create a Streamlit figure for Original vs Predicted graph
    st.subheader('ORIGINAL VALUE VS PREDICTED VALUE')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=DF.index[train_size + seq_length:], y=y_test.flatten(), mode='lines', name='Original Price', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=DF.index[train_size + seq_length:], y=test_predictions.flatten(), mode='lines', name='Predicted Price', line=dict(color='darkgreen')))
    fig.update_layout(title='Original Price vs Predicted Price Over Time', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)
# Suggestion about whether to buy the stock
last_original_date = DF.index[train_size + seq_length + len(y_test) - 1]
last_original_price = y_test[-1][0]
last_predicted_date = last_original_date
last_predicted_price = test_predictions[-1][0]

SUGGESTION = ""
if last_predicted_price > last_original_price:
    SUGGESTION = f"Based on my analysis, as of {last_original_date.strftime('%Y-%m-%d')}, the predicted price ({last_predicted_price:.2f}) is higher than the actual price ({last_original_price:.2f}). You might want to hold off on buying this stock for now."
else:
    SUGGESTION = f"As of {last_original_date.strftime('%Y-%m-%d')}, the predicted price ({last_predicted_price:.2f}) is not higher than the actual price ({last_original_price:.2f}). Consider buying this stock."

st.subheader("FINAL SUGGESTION:")
st.write(SUGGESTION) 







