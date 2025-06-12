import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
from django.shortcuts import render
from .forms import TickerForm
from .ltsm_model import create_lstm_model
from datetime import datetime


def index(request):
    if request.method == 'POST':
        form = TickerForm(request.POST)
        if form.is_valid():
            ticker = form.cleaned_data['ticker']
            time_interval = request.POST.get('time_interval', '1d')  # Интервал времени

            # Загрузка данных с Yahoo Finance
            end_date = datetime.now().strftime('%Y-%m-%d')
            data = yf.download(ticker, start="2018-01-01", end=end_date)

            if data.empty:
                return render(request, 'index.html', {'form': form, 'error': 'No data found for the selected ticker.'})

            # Масштабирование данных и извлечение свечей
            close_data = data['Close'].values.reshape(-1, 1)
            dates = data.index
            ohlc_data = data[['Open', 'High', 'Low', 'Close']]

            # Обучение модели и получение предсказаний
            model, train_predict, test_predict, actual_data, future_predictions, future_dates, elapsed_time = create_lstm_model(
                close_data, dates)

            # Построение свечного графика
            trace_candlestick = go.Candlestick(
                x=ohlc_data.index,
                open=ohlc_data['Open'],
                high=ohlc_data['High'],
                low=ohlc_data['Low'],
                close=ohlc_data['Close'],
                name='Candlestick'
            )

            # Линия предсказаний
            trace_prediction = go.Scatter(
                x=future_dates[:len(future_predictions)],
                y=future_predictions.flatten(),
                mode='lines',
                name='Prediction',
                line=dict(color='blue', width=2)
            )

            # Компоновка графика
            layout = go.Layout(
                title=f"Stock Price Prediction for {ticker}",
                xaxis=dict(title="Date", rangeslider=dict(visible=True)),
                yaxis=dict(title="Price"),
            )

            fig = go.Figure(data=[trace_candlestick, trace_prediction], layout=layout)
            plot_div = pio.to_html(fig, full_html=False)

            # Передача данных в шаблон
            context = {
                'form': form,
                'plot_div': plot_div,
                'elapsed_time': elapsed_time,
                'ticker': ticker,
            }
            return render(request, 'index.html', context)

    else:
        form = TickerForm()
    return render(request, 'index.html', {'form': form})
