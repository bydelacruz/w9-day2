import os
import time

import joblib
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .metrics import REQUEST_COUNT, REQUEST_LATENCY

# Define the Pydantic model for input validation


class PredictionInput(BaseModel):
    x1: float
    x2: float


class PredictionOutput(BaseModel):
    score: float
    model_version: str


# Initialize FastAPI app
app = FastAPI(title="Model Serving API")

# Global variable to hold the model
model = None
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "baseline.joblib"
)
MODEL_VERSION = "v1.0"


@app.on_event("startup")
def load_model():
    """
    Load the LogisticRegression model from disk at startup.
    """
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(
                f"Warning: Model not found at {
                    MODEL_PATH
                }. Prediction endpoint will fail."
            )
    except Exception as e:
        print(f"Error loading model: {e}")


@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """
    Middleware to track request count and latency for Prometheus metrics.
    """
    method = request.method
    endpoint = request.url.path

    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
        return response
    except Exception as e:
        # Count internal server errors
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
        raise e
    finally:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Predict endpoint using the loaded LogisticRegression model.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Prepare features: reshape to 2D array as required by scikit-learn
    features = [[input_data.x1, input_data.x2]]

    try:
        # Get probability of class 1 (positive class)
        # predict_proba returns [[prob_class_0, prob_class_1]]
        score = model.predict_proba(features)[0][1]

        return {"score": round(score, 4), "model_version": MODEL_VERSION}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
