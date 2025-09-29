from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class PredictionResult(BaseModel):
    condition: str
    confidence: float
    rank: int

class ImagePredictionResponse(BaseModel):
    predictions: List[PredictionResult]
    all_probabilities: Dict[str, float]
    model_type: str

class SymptomInput(BaseModel):
    symptoms: str = Field(..., min_length=1, max_length=1000)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, pattern="^(male|female|other)$")

class VoiceInput(BaseModel):
    audio_data: Optional[bytes] = None
    filename: Optional[str] = None

class VoiceTranscriptionResponse(BaseModel):
    success: bool
    text: str
    error: Optional[str] = None
    confidence: Optional[float] = None

class RAGQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(3, ge=1, le=20)

class LLMRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = None
    image_predictions: Optional[List[PredictionResult]] = None
    symptoms: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    success: bool = True

class CombinedAssessmentRequest(BaseModel):
    image_data: Optional[bytes] = None
    symptoms: Optional[str] = None
    voice_input: Optional[VoiceInput] = None
    age: Optional[int] = None
    gender: Optional[str] = None

class CombinedAssessmentResponse(BaseModel):
    image_predictions: Optional[List[PredictionResult]] = None
    transcribed_symptoms: Optional[str] = None
    symptom_analysis: Optional[Dict] = None
    medical_context: Optional[Dict] = None
    final_assessment: Optional[Dict] = None
    recommendations: List[str]