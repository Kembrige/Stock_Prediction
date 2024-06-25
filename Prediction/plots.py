import matplotlib.pyplot as plt

def plot_predictions(data, train_predict, test_predict):
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Actual Stock Price')
    plt.plot(range(100, len(train_predict) + 100), train_predict, label='Train Prediction')
    plt.plot(range(len(data) - len(test_predict), len(data)), test_predict, label='Test Prediction')
    plt.legend()
    plt.savefig('static/predictions.png')
    plt.close()







