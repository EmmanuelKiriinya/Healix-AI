from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # API Configuration
    APP_NAME: str = "Healix AI Backend"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model Paths
    MODEL_PATH: str = "app/models/final_model_weights.pth"
    MODEL_METADATA_PATH: str = "app/models/final_model_meta.json"
    
    # Data Paths
    KNOWLEDGE_BASE_PATH: str = "app/data/knowledge_base"
    VECTOR_STORE_PATH: str = "app/data/vector_store"
    
    # API Keys
    ASSEMBLYAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # Voice Processing
    MAX_AUDIO_SIZE: int = 25 * 1024 * 1024
    SUPPORTED_AUDIO_FORMATS: List[str] = ["mp3", "wav", "m4a", "webm", "mp4"]
    
    # Model Configuration
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3
    
    # LLM Configuration
    LLM_MODEL: str = "llama-3.1-8b-instant"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

settings = Settings()