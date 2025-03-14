"""
LogGuardian API server using FastAPI.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Import LogGuardian components
from logguardian.pipeline import LogGuardian

# Define models for API requests/responses
class LogEntry(BaseModel):
    message: str
    timestamp: Optional[str] = None
    source: Optional[str] = None


class LogSequence(BaseModel):
    logs: List[LogEntry]
    window_size: Optional[int] = 100
    stride: Optional[int] = None


class DetectionRequest(BaseModel):
    logs: List[LogEntry]
    window_size: Optional[int] = 100
    stride: Optional[int] = None
    batch_size: Optional[int] = 16
    raw_output: Optional[bool] = False


class DetectionResponse(BaseModel):
    results: List[Dict[str, Any]]
    inference_time: float
    processing_time: float


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    uptime: float


# Create FastAPI app
app = FastAPI(
    title="LogGuardian API",
    description="API for anomaly detection in system logs using large language models",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables
start_time = time.time()
detector = None
config = {}

# Authentication dependency
async def authenticate(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """
    Authenticate API requests.
    
    Args:
        credentials: HTTP bearer token
        
    Returns:
        True if authenticated
    """
    if not os.getenv("ENABLE_AUTH", "false").lower() in ("true", "1", "yes"):
        return True
    
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Bearer authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In production, implement proper authentication here
    # This is a placeholder for demonstration purposes
    valid_token = os.getenv("API_TOKEN", "logguardian-demo-token")
    if credentials.credentials != valid_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return True


@app.on_event("startup")
async def startup_event():
    """
    Initialize the API server.
    """
    global detector, config
    
    # Load configuration
    config_path = os.getenv("MODEL_CONFIG_PATH", "config/model_config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            config = {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        config = {}
    
    # Initialize LogGuardian
    try:
        logger.info("Initializing LogGuardian...")
        detector = LogGuardian(config=config)
        logger.info("LogGuardian initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing LogGuardian: {e}")
        detector = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    global start_time
    
    return {
        "status": "ok" if detector is not None else "error",
        "version": "0.1.0",
        "uptime": time.time() - start_time,
    }


@app.post("/detect", response_model=DetectionResponse, dependencies=[Depends(authenticate)])
async def detect_anomalies(request: DetectionRequest):
    """
    Detect anomalies in log sequences.
    
    Args:
        request: Detection request
        
    Returns:
        Detection results
    """
    global detector
    
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="LogGuardian not initialized",
        )
    
    # Prepare input
    log_messages = [log.message for log in request.logs]
    
    # Record processing time
    start_process_time = time.time()
    
    # Run detection
    try:
        # Time inference
        start_inference_time = time.time()
        results = detector.detect(
            log_messages,
            window_size=request.window_size,
            stride=request.stride,
            batch_size=request.batch_size,
            raw_output=True,
        )
        inference_time = time.time() - start_inference_time
        
        # Enhance results with source information if available
        for i, result in enumerate(results):
            if i < len(request.logs) and request.logs[i].source:
                result["source"] = request.logs[i].source
            if i < len(request.logs) and request.logs[i].timestamp:
                result["timestamp"] = request.logs[i].timestamp
        
        # Calculate total processing time
        processing_time = time.time() - start_process_time
        
        return {
            "results": results,
            "inference_time": inference_time,
            "processing_time": processing_time,
        }
    except Exception as e:
        logger.exception(f"Error in anomaly detection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}",
        )


@app.post("/batch-detect", dependencies=[Depends(authenticate)])
async def batch_detect_anomalies(
    request: DetectionRequest, background_tasks: BackgroundTasks
):
    """
    Run anomaly detection as a background task for large log sequences.
    
    Args:
        request: Detection request
        background_tasks: FastAPI background tasks
        
    Returns:
        Task ID for later result retrieval
    """
    # This would normally implement a background processing system
    # For simplicity, we're not implementing the full background task system here
    
    return {
        "task_id": "demo-task-id",
        "status": "accepted",
        "message": "Background processing demo - implementation would use Redis/Celery in production",
    }


@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """
    Add metrics middleware.
    
    Args:
        request: HTTP request
        call_next: Next middleware
        
    Returns:
        HTTP response
    """
    # This would normally implement Prometheus metrics
    # For simplicity, we're just recording basic timing
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    return response


def start_server():
    """
    Start the API server.
    """
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("API_WORKERS", "1"))
    
    logger.info(f"Starting LogGuardian API server on {host}:{port} with {workers} workers")
    
    uvicorn.run(
        "logguardian.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    start_server()