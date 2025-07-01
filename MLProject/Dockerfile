# Use Python 3.10 as base image
FROM python:3.10-slim

# Install MLflow
RUN pip install mlflow

# Copy model artifacts (to be populated by mlflow models build-docker)
COPY ./mlruns /mlruns

# Expose port for MLflow model serving
EXPOSE 5000

# Command to serve the model (adjusted by mlflow models build-docker)
CMD ["mlflow", "models", "serve", "-m", "/mlruns/0/$RUN_ID/artifacts/random_forest_model", "--port", "5000"]
