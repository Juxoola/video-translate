FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Установка Python 3.11 и необходимых пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Установка системных зависимостей (ffmpeg, libsndfile1 и пр.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    libnvidia-encode-525 \

    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Задаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей и устанавливаем зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Создаем каталоги для загрузок и результатов и изменяем их права доступа
RUN mkdir -p /tmp/uploads /tmp/output && chmod -R 777 /tmp/uploads /tmp/output

# Копируем исходный код приложения
COPY . .

# Открываем порт для веб-интерфейса (Gradio по умолчанию 7860)
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Запускаем основной файл приложения
CMD ["python3", "main.py"]
