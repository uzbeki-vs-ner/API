"""
FastAPI application for NER service using GLiNER model.
"""

import os
import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from app.model_service import NERModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model_service = None

# Model path from environment variable or default
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/case-solution/model")


# Pydantic models for request/response validation
class PredictRequestItem(BaseModel):
    """Single item in a prediction request."""
    hash: str = Field(..., description="Unique identifier for the text")
    text: str = Field(..., description="Input text for NER")
    
    @validator('hash')
    def hash_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('hash must not be empty')
        return v
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('text must not be empty')
        return v


class Entity(BaseModel):
    """Entity in the prediction response."""
    label: str = Field(..., description="Entity type: ORG, NAME, or GEO")
    start: int = Field(..., ge=0, description="Start position in text")
    end: int = Field(..., ge=0, description="End position in text")


class PredictResponseItem(BaseModel):
    """Single item in a prediction response."""
    hash: str = Field(..., description="Unique identifier for the text")
    entities: List[Entity] = Field(default_factory=list, description="List of entities")


class PredictResponse(BaseModel):
    """Prediction response envelope."""
    data: List[PredictResponseItem] = Field(..., description="List of prediction results")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_service
    
    logger.info("Starting NER service with GLiNER model")
    logger.info(f"Model path: {MODEL_PATH}")
    
    # Check if model directory exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model directory does not exist: {MODEL_PATH}")
        model_service = None
    else:
        # Load model
        try:
            model_service = NERModelService(MODEL_PATH)
            model_service.load()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            model_service = None
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down NER service")


app = FastAPI(
    title="NER Service with GLiNER",
    description="Named Entity Recognition service using GLiNER model",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model_service
    
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet"
        )
    
    return HealthResponse(status="ok")


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(request: List[PredictRequestItem]):
    """Predict entities in a batch of texts."""
    global model_service
    
    # Check if model is loaded
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet"
        )
    
    # Validate request
    if not request:
        raise HTTPException(
            status_code=400,
            detail="Request must not be empty"
        )
    
    # Check for duplicate hashes
    hashes = [item.hash for item in request]
    if len(hashes) != len(set(hashes)):
        raise HTTPException(
            status_code=400,
            detail="All hashes must be unique within a request"
        )
    
    try:
        # Perform prediction for each text
        results = []
        for item in request:
            entities = model_service.predict(item.text)
            
            results.append(
                PredictResponseItem(
                    hash=item.hash,
                    entities=entities
                )
            )
        
        return PredictResponse(data=results)
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during prediction"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )
