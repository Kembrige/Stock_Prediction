from django import forms

class TickerForm(forms.Form):
    ticker = forms.ChoiceField(choices=[
        ('AAPL', 'Apple'),
        ('GOOGL', 'Alphabet'),
        ('MSFT', 'Microsoft'),
        ('AMZN', 'Amazon'),
        ('TSLA', 'Tesla'),

    ])



