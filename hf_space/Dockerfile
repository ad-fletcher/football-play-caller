FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py environment.py app.py validation.py play_outcome_model.py ./
COPY outcome_classifier.joblib model_features.json label_encoders.json ./
COPY yards_q*.joblib ./

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
