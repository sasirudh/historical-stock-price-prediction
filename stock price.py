import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Define the ticker symbol
ticker_symbol = 'AAPL'

# Download stock data using yfinance
ticker_data = yf.download(ticker_symbol, start='2020-01-01', end='2024-01-01')

# Web scrape revenue data
url = 'https://www.macrotrends.net/stocks/charts/AAPL/apple/revenue'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
tables = soup.find_all('table')

# Attempt to extract revenue data safely
try:
    revenue_data = pd.read_html(str(tables[1]))[0]
    revenue_data.columns = ['Date', 'Revenue']
    revenue_data['Revenue'] = revenue_data['Revenue'].replace('[\$,]', '', regex=True).astype(float)
except (IndexError, ValueError, KeyError) as e:
    print("Error extracting revenue data:", e)
    revenue_data = pd.DataFrame(columns=['Date', 'Revenue'])

# Plot stock data
fig = make_subplots(rows=2, cols=1, subplot_titles=('Stock Prices', 'Revenue Data'))
fig.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], name='Close Price'), row=1, col=1)
if not revenue_data.empty:
    fig.add_trace(go.Bar(x=revenue_data['Date'], y=revenue_data['Revenue'], name='Revenue'), row=2, col=1)
fig.update_layout(title=f'{ticker_symbol} Stock Data and Revenue', height=700)
fig.show()

# Prepare data for prediction
stock_data = ticker_data.copy()
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].astype('int64') / 10**9
X = stock_data[['Date']]
y = stock_data['Close']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Predict future prices
future_dates = pd.date_range(start='2024-01-02', periods=30)
future_dates_numeric = future_dates.astype('int64') / 10**9
future_predictions = model.predict(future_dates_numeric.values.reshape(-1, 1))

# Plot predictions
fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=ticker_data.index, y=ticker_data['Close'], name='Actual Close Price'))
fig_pred.add_trace(go.Scatter(x=future_dates, y=future_predictions, name='Predicted Price', mode='lines'))
fig_pred.update_layout(title=f'{ticker_symbol} Stock Price Prediction', height=500)
fig_pred.show()

import matplotlib.pyplot as plt

# Plot predictions using matplotlib
plt.figure(figsize=(10, 6))

# Plot actual close prices
plt.plot(ticker_data.index, ticker_data['Close'], label='Actual Close Price', color='blue')

# Plot predicted future prices
plt.plot(future_dates, future_predictions, label='Predicted Price', color='red', linestyle='--')

# Title and labels
plt.title(f'{ticker_symbol} Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

