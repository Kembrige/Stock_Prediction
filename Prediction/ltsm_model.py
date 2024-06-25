import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import time

def create_lstm_model(data, dates, time_step=100, future_days=365):


    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)



    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]




    # Функция для создания наборов данных с временными шагами
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)


    # Создание обучающих и тестовых данных
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)







    # Изменение формы данных для LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)







    # Создание модели LSTM
    model = Sequential()
    model.add(LSTM(150, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Раннее завершение обучения
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Время начала обучения
    start_time = time.time()

    # Обучение модели
    model.fit(X_train, y_train, batch_size=15, epochs=100, validation_split=0.2, verbose=1, callbacks=[early_stopping])

    # Время завершения обучения
    end_time = time.time()

    # Предсказание на обучающих и тестовых данных
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Восстановление данных до исходного масштаба
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)





    # Предсказание будущих значений
    last_days = scaled_data[-time_step:]
    future_predictions = []
    for _ in range(future_days):
        last_days = last_days.reshape((1, time_step, 1))
        future_pred = model.predict(last_days)
        future_pred_scaled = scaler.inverse_transform(future_pred)
        future_predictions.append(future_pred_scaled[0, 0])
        future_pred = future_pred.reshape((1, 1))
        last_days = np.append(last_days[:, 1:, :], future_pred[:, :, np.newaxis], axis=1)

    future_predictions = np.array(future_predictions)



    # Добавление вариативности к будущим предсказаниям
    variability_factor = 0.05  # 5% вариативность
    noise = np.random.normal(0, variability_factor, future_predictions.shape)
    future_predictions = future_predictions * (1 + noise)

    # Формирование дат для будущих предсказаний
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, future_days + 1)]






    # Расчёт затраченного времени на обучение
    elapsed_time = end_time - start_time
    elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    return model, train_predict, test_predict, scaler.inverse_transform(scaled_data), future_predictions, future_dates, elapsed_time_str
