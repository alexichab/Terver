import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.api import ARIMA, SimpleExpSmoothing

# Чтение данных
file_path = 'all-in.txt'
data = pd.read_csv(file_path, header=None)
data_series = data[0]

# Подготовка данных
train_data = data_series[:2800]
test_data = data_series[2800:2805]

# Простое скользящее среднее (SMA)
window_size = 5
sma_2800 = train_data.rolling(window_size).mean()
sma_predictions = sma_2800.iloc[-1]

# Экспоненциальное сглаживание
exp_model_2800 = SimpleExpSmoothing(train_data).fit(smoothing_level=0.7)
exp_predictions_2800 = exp_model_2800.fittedvalues
exp_forecast = exp_model_2800.forecast(5)

# ARIMA
p = 5
d = 3
q = 5
arima_model_2800 = ARIMA(train_data, order=(p,d,q)).fit()
arima_predictions_2800 = arima_model_2800.predict(start=1, end=len(train_data))
arima_forecast = arima_model_2800.forecast(5)

# Расширение серий для включения прогнозов
sma_extended = pd.concat([sma_2800, pd.Series([sma_predictions] * 5, index=range(2800, 2805))])
exp_extended = pd.concat([exp_predictions_2800, exp_forecast])
arima_extended = pd.concat([arima_predictions_2800, arima_forecast])

# Построение графика
plt.figure(figsize=(18, 8))
plt.plot(train_data, label='Actual Data (Train)', color='green')
plt.plot(test_data, label='Actual Data (Test)', color='blue')
plt.plot(sma_extended, label='SMA', color='red')
plt.plot(exp_extended, label='Exponential Smoothing', color='purple')
plt.plot(arima_extended, label='ARIMA', color='orange')
plt.title('Comparison of SMA, Exponential Smoothing, and ARIMA')
plt.xlabel('Time Point')
plt.ylabel('Internet Traffic')
plt.legend()
plt.grid(True)
plt.show()