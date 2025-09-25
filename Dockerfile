# Dockerfile per Diabetes Readmission Prediction Pipeline
FROM python:3.9-slim

# Informazioni del container
LABEL maintainer="Diabetes Prediction Pipeline"
LABEL version="3.0.0"
LABEL description="Complete pipeline for diabetes hospital readmission prediction"

# Imposta directory di lavoro
WORKDIR /app

# Installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# Installa dipendenze Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il progetto
COPY . .

# Crea directory per output se non esistono
RUN mkdir -p outputs/datasets_clean/cluster/terzo_metodo
RUN mkdir -p outputs/results
RUN mkdir -p outputs/models

# Imposta permessi
RUN chmod +x test/simple_test.py

# Espone porta per eventuali API future
EXPOSE 8000

# Variabili d'ambiente
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Comando di default per eseguire test
CMD ["python", "test/simple_test.py"]