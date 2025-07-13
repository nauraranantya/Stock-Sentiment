import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from src.lstm_model import build_lstm_model, train_lstm_model
import matplotlib.pyplot as plt
import os

def create_sequences(X, y, window_size=10):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size])
    return np.array(Xs), np.array(ys)

# Load merged dataset
df = pd.read_csv("data/merged/tesla_sentiment_stock_enhanced.csv", parse_dates=['date'])
df = df.sort_values('date').dropna()

# Select features for LSTM input (you can customize)
features = [
    'sentiment_avg',
    '%pos',
    '%neg',
    'sentiment_momentum',
    'finbert_confidence',
    'volatility',
    'volume_spike',
    'daily_return'
]
target = 'daily_return' 
# Normalize features & target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[features])
y_scaled = scaler_y.fit_transform(df[[target]])

# Create sequences
window_size = 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, window_size)

# Walk-forward validation
train_size = int(len(X_seq) * 0.8)
X_train, X_val = X_seq[:train_size], X_seq[train_size:]
y_train, y_val = y_seq[:train_size], y_seq[train_size:]

# Build LSTM model
model = build_lstm_model(input_shape=(window_size, len(features)))

# Train model
history = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=16)

# Predict on validation set
y_pred_scaled = model.predict(X_val)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_val)

# Plot predictions vs true
plt.figure(figsize=(12,6))
plt.plot(y_true, label='Actual Close Price')
plt.plot(y_pred, label='Predicted Close Price')
plt.title('LSTM: Actual vs Predicted Close Price')
plt.xlabel('Time Step')
plt.ylabel('Price')
plt.legend()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/lstm_predictions.png", dpi=300)
plt.close()

# Print final MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_true, y_pred)
print(f"LSTM Model MSE: {mse:.6f}")