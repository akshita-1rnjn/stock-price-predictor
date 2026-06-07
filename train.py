import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression

# Download stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")

# Create prediction column
data['Prediction'] = data['Close'].shift(-1)

# Features and target
X = np.array(data['Close']).reshape(-1, 1)[:-1]
y = np.array(data['Prediction'])[:-1]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next day price
latest_price = np.array([[data['Close'].iloc[-1]]])
prediction = model.predict(latest_price)

print("Latest Close Price:", latest_price[0][0])
print("Predicted Next Day Price:", prediction[0])
