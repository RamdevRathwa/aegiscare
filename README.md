# AegisCare – Patient Risk & Outcome Prediction

ML-powered predictive analytics for healthcare data, with explainable AI insights and a doctor-facing dashboard.

# docker-compose.yml

version: '3.8'
services:
api:
build: ./api
ports:

- "8000:8000"
  app:
  build: ./app
  ports:
- "8501:8501"
  depends_on:
- api
