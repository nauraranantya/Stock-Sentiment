# TSLA Stock Prediction using LSTM and Sentiment Analysis

## Overview
This project investigates how sentiment from Reddit (e.g., /r/teslamotors) affects the TSLA stock price. It applies **FinBERT sentiment analysis** to Reddit posts and trains an **LSTM model** to predict **daily stock returns** based on sentiment and technical signals.

---

## Workflow

### Phase 1: Data Collection & Preprocessing
- Scraped Reddit posts using PRAW (Pushshift fallback).
- Applied FinBERT to extract `positive`, `neutral`, `negative` labels and confidence scores.
- Aggregated sentiment data daily with:
  - Mean sentiment score
  - Momentum (Δ sentiment)
  - % Positive / % Negative

### Phase 2: Feature Engineering
Merged Reddit sentiment with TSLA stock data from Yahoo Finance:
- `daily_return`, `volatility`, `volume_spike`
- 7 total features used to train LSTM.

### Phase 3: Model Training (LSTM)
- Used a window size of 10 to build sequences.
- Applied MinMaxScaler to inputs and target.
- Trained with early stopping.
- Output: Predicted next-day returns.

### Phase 4: Evaluation
- Target: `daily_return` (instead of price)
- Achieved low validation error:
  - **MSE**: 0.000174
  - **RMSE** ≈ 0.013
- Plotted predicted vs actual values.

---

## Example Output

![Prediction Plot](figures/lstm_predictions.png)

---

## Data Outputs
- `data/lstm_predictions.csv` – Actual vs Predicted returns
- `figures/lstm_predictions.png` – Visualization of model predictions

---

## Key Insights
- Reddit sentiment is a **weak but meaningful** signal.
- Volume spikes and momentum boost prediction accuracy.
- LSTM captures short-term TSLA return dynamics better than ARIMA.

---

## Next Steps
- Predict price directly instead of return.
- Add binary classification (Up/Down).
- Deploy Streamlit dashboard.