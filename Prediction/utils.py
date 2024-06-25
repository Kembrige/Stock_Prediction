import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist[['Close']]



