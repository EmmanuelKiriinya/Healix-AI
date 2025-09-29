from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from typing import Optional
import logging
from app.models.schemas import (
    ImagePredictionResponse, SymptomInput, RAGQuery,
    CombinedAssessmentRequest, CombinedAssessmentResponse,
    VoiceTranscriptionResponse, LLMRequest, LLMResponse
)
from app.models.skin_model import skin_model
from app.services.voice_service import voice_service
from app.services.rag_service import rag_service
from app.services.llm_service import llm_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def root():
    return {
        "message": "Healix AI Backend API",
        "version": "1.0.0",
        "endpoints": {
            "image_prediction": "/api/v1/predict/image",
            "voice_transcription": "/api/v1/transcribe/voice",
            "llm_consultation": "/api/v1/consult/llm",
            "combined_assessment": "/api/v1/assess/combined",
            "health_check": "/api/v1/health"
        }
    }

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "skin_model": skin_model.model is not None,
            "voice_service": voice_service.client is not None,
            "llm_service": llm_service.client is not None,
            "rag_service": rag_service.collection is not None
        }
    }

@router.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_from_image(file: UploadFile = File(...)):
    """Predict skin conditions from uploaded image"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "File must be an image")
        
        image_data = await file.read()
        result = skin_model.predict(image_data)
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(500, f"Prediction failed: {str(e)}")

@router.post("/transcribe/voice", response_model=VoiceTranscriptionResponse)
async def transcribe_voice(file: UploadFile = File(...)):
    """Transcribe voice audio to text"""
    try:
        if not file.filename or not voice_service.is_audio_file(file.filename):
            raise HTTPException(400, "Unsupported audio format. Supported formats: mp3, wav, m4a, webm, mp4")
        
        audio_data = await file.read()
        
        result = await voice_service.transcribe_audio(audio_data, file.filename)
        
        # If the service returned an error, raise it as an HTTP exception
        if not result["success"]:
            raise HTTPException(400, result["error"])
            
        return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        raise HTTPException(500, f"Transcription failed: {str(e)}")

@router.post("/consult/llm", response_model=LLMResponse)
async def consult_llm(request: LLMRequest):
    """Consult LLM with medical context"""
    try:
        response = llm_service.generate_response(
            request.query, 
            request.context, 
            request.image_predictions, 
            request.symptoms
        )
        return LLMResponse(response=response)
    except Exception as e:
        logger.error(f"LLM consultation failed: {str(e)}")
        return LLMResponse(
            response=f"LLM consultation failed: {str(e)}",
            success=False
        )

@router.post("/assess/combined", response_model=CombinedAssessmentResponse)
async def combined_assessment(
    image_data: Optional[bytes] = File(None),
    symptoms: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    gender: Optional[str] = Form(None)
):
    """Complete multimodal assessment"""
    try:
        image_predictions = None
        medical_context = None
        final_assessment = None
        
        # Process image if provided
        if image_data:
            image_result = skin_model.predict(image_data)
            image_predictions = image_result["predictions"]
            
            # Query medical knowledge based on top prediction
            if image_predictions:
                top_condition = image_predictions[0]["condition"]
                medical_context = rag_service.query_knowledge_base(
                    f"{top_condition} symptoms treatment diagnosis", 
                    max_results=3
                )
        
        # Generate final assessment using LLM if we have data
        if image_predictions or symptoms:
            context_text = "\n".join([str(ctx) for ctx in medical_context]) if medical_context else ""
            
            assessment = llm_service.generate_response(
                query="Provide a comprehensive medical assessment based on the available information",
                context=context_text,
                image_predictions=image_predictions,
                symptoms=symptoms
            )
            
            final_assessment = {
                "assessment": assessment,
                "generated_by": "Groq LLM"
            }
        
        # Analyze symptoms if provided
        symptom_analysis = None
        if symptoms:
            symptom_analysis = {
                "text": symptoms,
                "age": age,
                "gender": gender,
                "processed": True
            }
        
        return CombinedAssessmentResponse(
            image_predictions=image_predictions,
            symptom_analysis=symptom_analysis,
            medical_context=medical_context,
            final_assessment=final_assessment,
            recommendations=[
                "Consult with a healthcare professional for accurate diagnosis",
                "Monitor symptoms and seek medical attention if condition worsens",
                "Follow up with a dermatologist for specialized care"
            ]
        )
        
    except Exception as e:
        logger.error(f"Combined assessment error: {str(e)}")
        raise HTTPException(500, f"Combined assessment failed: {str(e)}")

@router.get("/model/info")
async def get_model_info():
    """Get model information"""
    return skin_model.get_model_info()