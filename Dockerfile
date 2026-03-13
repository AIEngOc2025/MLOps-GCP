FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système si nécessaire
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injecte la variable d'environnement PORT
# On utilise 'exec' pour que les signaux système (SIGTERM) soient bien reçus
CMD exec uvicorn src.main:app --host 0.0.0.0 --port $PORT