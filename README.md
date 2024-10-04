# Stock Price Prediction with LSTM and Moving Averages

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) model trained on historical stock data. The model predicts future prices based on the last 60 days of stock prices. It also provides buy/sell signals based on moving averages.

## Features

- **Stock Price Prediction**: Predicts the stock price for the next 10 days based on the last 60 days of historical data using an LSTM model.
- **Moving Averages**: Includes 5-day and 10-day moving averages for better trend analysis.
- **Buy/Sell Signals**: Suggests whether to buy or sell based on the last day’s price compared to the moving averages.

## How It Works

1. **LSTM Model**: The LSTM model is trained on the historical closing prices of a stock. This project uses `BTC-USD` (Bitcoin to USD) as a demo.
2. **Scaling the Data**: The stock prices are normalized using `MinMaxScaler` to fit within the range of 0 to 1, as required for LSTM models.
3. **Predictions**: After training, the model predicts future stock prices based on the last 60 days of data.
4. **Moving Averages**: Moving averages for the last 5 and 10 days are calculated from the combined actual and predicted prices to assess stock trends.
5. **Buy/Sell Signal**: Based on the comparison of the last actual price and moving averages, a decision is made to either buy or sell the stock.

## Project Structure

- **model**: Trained LSTM model saved as `stock_price_lstm_model.h5`.
- **input/BTC-USD.csv**: Historical stock data (used for demo).
- **code**: Python script to predict stock prices and plot moving averages with buy/sell signals.

## Prerequisites

To run this project, you'll need the following Python libraries:

- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

You can install the required libraries using:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
```

2. Navigate to the project directory:

```bash
cd stock-price-prediction
```

3. Make sure you have your trained LSTM model in the `input/` folder (or train your own model). When prompted, enter the path to your stock data CSV file or press Enter to use the default (`input/BTC-USD.csv`).

4. Run the stock prediction script:

```bash
python predict_stock_prices.py
```

5. The script will display:
   - The predicted stock prices for the next 10 days.
   - Buy/Sell signals based on the last day's actual price and the moving averages.
   - A graph showing:
     - Actual prices from the last 60 days.
     - Predicted prices for the next 10 days.
     - 5-day and 10-day moving averages.

## Buy/Sell Strategy

The function `BuyOrSell` suggests whether to buy or sell based on:
- **Last Day’s Price vs. 5-Day Moving Average**: If the last price is below the 5-day MA predicted price, it signals to **BUY**, otherwise **SELL**.
- **Last Day’s Price vs. 10-Day Moving Average**: If the last price is below the 10-day MA predicted price, it signals to **BUY**, otherwise **SELL**.
- **Default Comparison**: Without considering any moving averages, it compares the last price with the predicted price.

## Example Output

```
Predicted Stock Price: 35000.21
Predicted Prices for the Next 10 Days: [35050.25, 35075.65, 35100.15, ...]

Action Based on Prediction: BUY
Action Based on Prediction MA5: SELL
Action Based on Prediction MA10: BUY
```

## Visualization

The script generates a plot that includes:

- **Last 60 Days Actual Prices**: Plotted in black.
- **Predicted Future Prices**: Plotted in green dashed lines.
- **5-Day Moving Average**: Plotted in blue dotted lines.
- **10-Day Moving Average**: Plotted in red dotted lines.

## Stock Price Prediction Plot
![Stock Price Prediction Plot](https://github.com/Skynerd/Stock_Market_Prediction/blob/main/DemoPlot.png)
