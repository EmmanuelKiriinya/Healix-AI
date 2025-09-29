from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from app.core.config import settings
from app.api.endpoints import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create logger
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Healix AI - Multimodal Skin Condition Detection",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - Essential for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins - adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    os.makedirs("app/data", exist_ok=True)
    os.makedirs("app/data/models", exist_ok=True)
    logger.info("Healix AI Backend started successfully")
    logger.info(f"API Documentation available at: http://localhost:{settings.PORT}/docs")

@app.get("/")
async def root():
    return {
        "message": "Welcome to Healix AI Backend",
        "version": settings.VERSION,
        "docs": "/docs",
        "endpoints": {
            "image_prediction": "/api/v1/predict/image",
            "voice_transcription": "/api/v1/transcribe/voice",
            "llm_consultation": "/api/v1/consult/llm",
            "combined_assessment": "/api/v1/assess/combined",
            "health_check": "/api/v1/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )