from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# number of days 
no_of_days = 60
 
# Load the saved model
model = load_model('input/stock_price_lstm_model.h5')

# Prompt the user for a CSV file path or default to 'input/BTC-USD.csv'
csv_file = input("Enter the path to the stock data CSV file (or press Enter to use 'input/BTC-USD.csv'): ")
if csv_file == '': csv_file = 'input/BTC-USD.csv'

# Load new stock data
new_data = pd.read_csv(csv_file)

# Filter the 'Close' price (or other relevant column)
new_close_data = new_data.filter(['Close'])

# Convert to numpy array
new_dataset = new_close_data.values

# Note: Use the same scaler parameters used in training
scaler = MinMaxScaler(feature_range=(0,1))

# Fit the scaler on the entire dataset (This assumes you fit the scaler on the entire dataset during training)
scaled_new_data = scaler.fit_transform(new_dataset)

# Ensure you use the last no_of_days days to create the input sequence for prediction
last_no_days = scaled_new_data[-no_of_days:]

# Prepare the test data (input for LSTM)
X_test_new = []
X_test_new.append(last_no_days)
X_test_new = np.array(X_test_new)

# Reshape to 3D format (samples, time steps, features) expected by LSTM
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))

# Predict the stock price
predicted_price = model.predict(X_test_new)

# Inverse transform the predicted price to original scale
predicted_price = scaler.inverse_transform(predicted_price)
print(f"Predicted Stock Price: {predicted_price[0][0]}")


# Function to predict future prices for the next `num_days`
def predict_next_days(model, data, scaler, num_days=10):
    predictions = []
    input_seq = data[-no_of_days:]  # Start with the last no days

    for day in range(num_days):
        X = np.reshape(input_seq, (1, input_seq.shape[0], 1))  # Reshape for LSTM input
        predicted_price = model.predict(X)
        predicted_price_unscaled = scaler.inverse_transform(predicted_price)
        predictions.append(predicted_price_unscaled[0][0])
        
        # Update input_seq by appending the predicted price and removing the first value
        input_seq = np.append(input_seq, predicted_price, axis=0)[1:]

    return predictions

def BuyOrSell(consider_MA5, consider_MA10): 
    last_day_price = last_no_days_unscaled[-1]
    
    if consider_MA10: 
        if last_day_price < moving_average_10.iloc[-1]:
            return "BUY"
        else:
            return "SELL"
    
    elif consider_MA5: 
        if last_day_price < moving_average_5.iloc[-1] :
            return "BUY"
        else:
            return "SELL"
    
    else: 
        if last_day_price < combined_data[-1]:
            return "BUY"
        else:
            return "SELL"
         

# Predict future stock prices for the next 10 days
future_predictions = predict_next_days(model, last_no_days, scaler, num_days=10)
print(f"Predicted Prices for the Next 10 Days: {future_predictions}")

# Last no days actual data (unscaled)
last_no_days_unscaled = scaler.inverse_transform(last_no_days)

# Combine the last no days of actual prices with future predictions
combined_data = np.append(last_no_days_unscaled, np.array(future_predictions).reshape(-1, 1), axis=0)

# Calculate moving average on combined data
moving_average_5 = pd.Series(combined_data.flatten()).rolling(window=5).mean()
moving_average_10 = pd.Series(combined_data.flatten()).rolling(window=10).mean()

# Create a time axis for the plot
days = np.arange(1, len(combined_data) + 1)

# Plot the data
plt.figure(figsize=(12,6))
plt.plot(days[:no_of_days], last_no_days_unscaled, label='Last no Days Actual Prices', color='black')
plt.plot(days[no_of_days-1:], combined_data[no_of_days-1:], label='Predicted Future Prices', color='green', linestyle='--')

# Plot the moving average
plt.plot(days[4:], moving_average_5[4:], label='5-Day Moving Average', color='blue', linestyle=':')
plt.plot(days[9:], moving_average_10[9:], label='10-Day Moving Average', color='red', linestyle=':')
 
# Buy Or Sell Based on Prediction
print(f"Action Based on Prediction: {BuyOrSell(False,False)}")
print(f"Action Based on Prediction MA5: {BuyOrSell(True,False)}")
print(f"Action Based on Prediction MA10: {BuyOrSell(False,True)}")

# Add titles and labels
plt.title('Stock Price Prediction for the Next 10 Days')
plt.xlabel('Days')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()











