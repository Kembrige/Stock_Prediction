from django import forms

class TickerForm(forms.Form):
    ticker = forms.ChoiceField(choices=[
        ('AAPL', 'Apple'),
        ('GOOGL', 'Alphabet'),
        ('MSFT', 'Microsoft'),
        ('AMZN', 'Amazon'),
        ('TSLA', 'Tesla'),
    ], label="Select Stock Ticker")

    time_interval = forms.ChoiceField(choices=[
        ('1d', '1 Day'),
        ('1w', '1 Week'),
        ('1m', '1 Month'),
    ], label="Select Time Interval")