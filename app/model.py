import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from .data_prepare import create_windows_for_trade


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout_rate=0.5):
        super(BasicBlock, self).__init__()
        # Первый Conv1d слой
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,  # Сохраняет размерность
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)  # BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)

        # Второй Conv1d слой
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Остаточное соединение
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x

        # Первый Conv1d + BatchNorm + ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Второй Conv1d + BatchNorm
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Остаточное соединение
        residual = self.shortcut(residual)
        x += residual

        # ReLU после сложения
        x = self.relu(x)

        return x


class ResNet1D(nn.Module):
    def __init__(self, input_shape, num_classes, num_blocks=[2, 2, 2], initial_channels=64, dropout_rate=0.5):
        super(ResNet1D, self).__init__()
        seq_len, num_features = input_shape  # [Seq_len, Features]

        # Начальный слой
        self.initial_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=initial_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual Blocks
        self.layer1 = self._make_layer(initial_channels, initial_channels, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(initial_channels, initial_channels * 2, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(initial_channels * 2, initial_channels * 4, num_blocks[2], stride=2, dropout_rate=dropout_rate)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Классификатор
        self.fc = nn.Linear(initial_channels * 4, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_rate):
        layers = []
        # Первый блок с изменением размерности
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, dropout_rate=dropout_rate))
        # Остальные блоки
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dropout_rate=dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, Seq_len, Features]
        x = x.permute(0, 2, 1)  # [Batch, Features, Seq_len]

        # Начальный слой
        x = self.initial_conv(x)  # [Batch, initial_channels, Seq_len / 4]

        # Residual Blocks
        x = self.layer1(x)  # [Batch, initial_channels, Seq_len / 4]
        x = self.layer2(x)  # [Batch, initial_channels * 2, Seq_len / 8]
        x = self.layer3(x)  # [Batch, initial_channels * 4, Seq_len / 16]

        # Global Average Pooling
        x = self.global_avg_pool(x)  # [Batch, initial_channels * 4, 1]
        x = x.squeeze(-1)  # [Batch, initial_channels * 4]

        # Классификация
        x = self.fc(x)  # [Batch, num_classes]

        return x


# Прогноз для всех 10 последних окон
def make_predictions(df, model, window_size=30):
    """
    Делает прогноз для всех окон длиной window_size.

    Параметры:
        df (pd.DataFrame): DataFrame с данными.
        model (torch.nn.Module): Обученная модель.
        window_size (int): Размер окна для прогноза.

    Возвращает:
        pd.DataFrame: DataFrame с результатами прогноза.
    """
    # Создаем окна для прогноза
    windows = create_windows_for_trade(df, window_size)

    # Ограничиваемся последними 10 окнами
    windows = windows[-10:]

    # Прогноз для каждого окна
    predictions = []
    probabilities_list = []
    for window in windows:
        X = torch.tensor(window.reshape(1, window_size, -1), dtype=torch.float32)
        with torch.no_grad():
            output = model(X)
            probabilities = torch.softmax(output, dim=1).numpy()[0]
            prediction = np.argmax(probabilities)
            predictions.append(prediction)
            probabilities_list.append(probabilities[prediction])

    # Создаем DataFrame с результатами
    results = pd.DataFrame({
        'timestamp': df.index[-10:],  # Время последних 10 свечей
        'Close': df['Close'].values[-10:],  # Цена закрытия
        'Prediction': predictions,  # Прогноз модели
        'Confidence': probabilities_list  # Уверенность в прогнозе
    })

    return results
