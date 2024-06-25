import yfinance as yf
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.shortcuts import render
from .forms import TickerForm
from .ltsm_model import create_lstm_model
from datetime import datetime, timedelta



def index(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']

            end_date = datetime.now().strftime('%Y-%m-%d')
            data = yf.download(ticker, start="2018-01-01", end=end_date)


            if data.empty:

                return render(request, 'index.html', {'form': form, 'error': 'No data found for the selected ticker.'})

            close_data = data['Close'].values.reshape(-1, 1)
            dates = data.index
            model, train_predict, test_predict, actual_data, future_predictions, future_dates, elapsed_time = create_lstm_model(
                close_data, dates)

            plt.figure(figsize=(12, 6))
            plt.plot(dates, actual_data, color='black', label='Actual Data')

            train_predict_indices = dates[100:len(train_predict) + 100]
            test_predict_indices = dates[len(train_predict) + 200:len(train_predict) + 200 + len(test_predict)]

            plt.plot(train_predict_indices, train_predict, color='green', label='Train Prediction')
            plt.plot(test_predict_indices, test_predict, color='red', label='Test Prediction')

            future_dates = future_dates[:len(future_predictions)]
            plt.plot(future_dates, future_predictions, color='blue', label='Future Prediction')
            plt.title(f'Stock Price Prediction for {ticker}')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()

            plot_path = f'static/prediction_{ticker}.png'
            if os.path.exists(plot_path):
                os.remove(plot_path)
            plt.savefig(plot_path)
            plt.close()

            context = {
                'form': form,
                'ticker': ticker,
                'plot_path': plot_path,
                'elapsed_time': elapsed_time
            }
            return render(request, 'index.html', context)
    else:
        form = TickerForm()

    return render(request, 'index.html', {'form': form})





