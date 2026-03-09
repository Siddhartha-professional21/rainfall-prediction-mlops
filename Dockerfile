FROM python:3.12-slim

RUN apt-get update && apt-get install -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    pandas \
    scikit-learn \
    lightgbm \
    joblib

CMD ["python", "src/predict.py"]