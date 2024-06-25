import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your actual data and predictions
# Assuming actual_data and future_predictions are numpy arrays

actual_data = np.load('/mnt/data/ActualAAPL.npy')  # Load your actual data here
future_predictions = np.load('/mnt/data/prediction_AAPL.npy')  # Load your predictions here

# Calculate MSE and MAE
mse = mean_squared_error(actual_data, future_predictions)
mae = mean_absolute_error(actual_data, future_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plotting the actual data and predictions
plt.figure(figsize=(12, 6))
plt.plot(actual_data, label='Actual Data', color='black')
plt.plot(future_predictions, label='Future Prediction', color='blue')

plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Stock Prices for AAPL')
plt.legend()
plt.show()