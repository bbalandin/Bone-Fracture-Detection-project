FROM python:3.12-slim

# Отключаем интерактивный режим
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Системные зависимости (нужны для OpenCV/ultralytics)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копируем весь код
COPY . .

# Загружаем модель при сборке (экономит время при старте)
RUN python download_model.py

# Делаем start.sh исполняемым
RUN chmod +x start.sh

# HF Spaces требует порт из README (app_port: 8501)
EXPOSE 8501

# Healthcheck — проверяет, что Streamlit отвечает
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Запускаем оба процесса
CMD ["./start.sh"]