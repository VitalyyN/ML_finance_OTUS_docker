# Используем официальный образ Python
FROM python:3.13-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями
COPY requirements.txt .

# Обновляем pip до последней версии
RUN pip install --upgrade pip

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install numpy==2.2.3

# Копируем исходный код приложения
COPY . .

# Устанавливаем переменные окружения для Flask
ENV FLASK_APP=app
ENV FLASK_ENV=production

# Открываем порт 5000
EXPOSE 5000

# Команда для запуска приложения
CMD ["python", "run.py"]