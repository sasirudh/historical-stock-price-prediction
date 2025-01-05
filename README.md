# Stock Price Prediction and Visualization

This repository contains a Python project for downloading historical stock data, web scraping revenue data, visualizing stock prices alongside revenue, and predicting future stock prices using machine learning.

## Features

- **Download Stock Data**: Stock data for Apple Inc. (AAPL) is downloaded using the `yfinance` library.
- **Web Scraping**: Revenue data for Apple is scraped from MacroTrends using the `BeautifulSoup` library.
- **Data Visualization**: Stock price and revenue data are visualized using `plotly` and `matplotlib`.
- **Stock Price Prediction**: A machine learning model is trained to predict future stock prices using the `RandomForestRegressor` from `sklearn`.

## Requirements

To run this project, you need the following Python libraries:

- `yfinance`
- `requests`
- `beautifulsoup4`
- `plotly`
- `pandas`
- `scikit-learn`
- `matplotlib`
  You can install the required libraries using `pip`:
  ```bash
   pip install yfinance requests beautifulsoup4 plotly pandas scikit-learn matplotlib
  
Usage
Download Stock Data: The yfinance library is used to download historical stock data for the ticker symbol AAPL from January 1, 2020 to January 1, 2024.

Web Scraping Revenue Data: The BeautifulSoup library is used to scrape revenue data from MacroTrends.

Data Visualization: The stock price and revenue data are visualized on two separate subplots using the plotly.graph_objs library.

Stock Price Prediction: A RandomForestRegressor model is trained to predict future stock prices based on the historical closing prices. The prediction is visualized using matplotlib.

Example of the prediction plot:
The actual stock prices and the predicted future stock prices for the next 30 days are plotted on a line graph. The actual stock prices are shown as a blue line, and the predicted prices are shown as a red dashed line.

Code Overview
Stock Data Download:

Using yfinance, historical stock data for AAPL is fetched from the specified date range.
Revenue Data Extraction:

Revenue data is extracted from the second table on the MacroTrends page using BeautifulSoup and pandas.
Machine Learning Model:

A RandomForestRegressor is trained on the stock data, with the date as the feature and closing price as the target.
The trained model predicts future stock prices, which are visualized.
Visualization:

plotly is used to plot the stock price and revenue data.
matplotlib is used to plot the predicted stock prices for the next 30 days.
Example Output
Stock Price and Revenue Visualization: A plot showing Apple’s stock price and revenue over time.
Stock Price Prediction: A plot showing actual stock prices along with predicted prices for the next 30 days.
Future Enhancements
Implement additional prediction models for comparison (e.g., LinearRegression, LSTM).
Improve error handling for web scraping.
Provide a GUI for input of different stock ticker symbols.





















