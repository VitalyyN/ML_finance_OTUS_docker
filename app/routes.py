from flask import Blueprint, render_template
import pandas as pd
import numpy as np
import torch
from pybit.unified_trading import HTTP
from .model import ResNet1D, make_predictions
from .data_prepare import prepare_date, normalize_dataframe, load_scaler
import datetime

main = Blueprint('main', __name__)

# Загрузка модели и скейлера
input_shape = [30, 109]
num_classes = 3
num_blocks = [3, 3, 3]
initial_channels = 128

model = ResNet1D(input_shape=input_shape, num_classes=num_classes, num_blocks=num_blocks, initial_channels=initial_channels)
model.load_state_dict(torch.load('app/resnet_1d_3_train_best_acc.pth', map_location=torch.device('cpu')))
model.eval()

scaler = load_scaler('app/scalers/scaler.pkl')

# Подключение к Bybit
session = HTTP(testnet=True)

window_size = 30
window_lag = [3, 9, 15, 30]
window_SMA = [6, 9, 30, 50]
window_RSI = [4, 9, 20]
windows_stats = [3, 10, 30, 50]
windows_trend = [3, 10, 30, 50]

@main.route('/')
def index():
    # Получение последних 15-минутных свечей
    # Получение данных по ETH
    eth_candles = session.get_kline(symbol="ETHUSD", category="inverse", interval="15", limit=200)['result']['list']
    btc_candles = session.get_kline(symbol="BTCUSD", category="inverse", interval="15", limit=200)['result']['list']

    # Преобразование данных ETH в DataFrame
    eth_df = pd.DataFrame(eth_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'].astype(int), unit='ms')  # Преобразуем timestamp в читаемый формат
    eth_df.set_index('timestamp', inplace=True)

    # Преобразование данных BTC в DataFrame
    btc_df = pd.DataFrame(btc_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'].astype(int), unit='ms')  # Преобразуем timestamp в читаемый формат
    btc_df.set_index('timestamp', inplace=True)

    # Переименование столбцов для BTC
    btc_df = btc_df[['close']].rename(columns={'close': 'btc_close'})

    # Объединение данных по ETH и BTC по времени
    df = eth_df.join(btc_df, how='inner')

    # Переименование столбцов для ETH
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }).drop(['turnover'], axis=1)

    # Убедимся, что данные отсортированы по времени
    df = df.sort_index()

    # Преобразуем все числовые колонки в float
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'btc_close']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Умножаем колонку Volume на цену
    df['Volume'] = df['Volume'] * df['Close']

    expected_columns = ['Close', 'High', 'Low', 'Open', 'Volume', 'btc_close']

    # Убедимся, что колонки в правильном порядке
    df = df[expected_columns]
  
    # Подготовка данных
    df = prepare_date(df, lags=window_lag, windows_SMA=window_SMA, windows_RSI=window_RSI, windows_stats=windows_stats, windows_trend=windows_trend)

    df, _ = normalize_dataframe(df, scaler=scaler, save_scaler=False)

    # Прогноз для всех 10 последних окон
    window_size = 30
    predictions_df = make_predictions(df, model, window_size)
    predictions_df['Close'] = eth_df['close'].values[-10:]

    # Преобразуем DataFrame в список словарей для передачи в шаблон
    predictions_data = predictions_df.to_dict('records')

    # Отображаем результаты в шаблоне
    return render_template('index.html', predictions=predictions_data)
