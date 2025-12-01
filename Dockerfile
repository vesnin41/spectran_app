# Dockerfile для XRD Explorer
FROM python:3.11-slim

# Базовые инструменты для сборки научных пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# По умолчанию запускаем интерактивный режим
CMD ["python", "-m", "src.cli_interactive"]
