from groq import Groq
from typing import List, Optional, Dict
from app.core.config import settings
import logging
from app.models.schemas import PredictionResult  # Import the schema

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize Groq client"""
        try:
            if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_groq_api_key_here":
                self.client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq client initialized successfully")
            else:
                logger.warning("GROQ_API_KEY not set. LLM service will be disabled.")
                self.client = None
        except Exception as e:
            logger.error(f"Error initializing Groq client: {str(e)}")
            self.client = None
    
    def generate_response(self, query: str, context: Optional[str] = None, 
                         image_predictions: Optional[List[PredictionResult]] = None, 
                         symptoms: Optional[str] = None) -> str:
        """Generate response using Groq LLM with context"""
        if not self.client:
            return "LLM service not configured. Please set GROQ_API_KEY."
        
        try:
            # Construct prompt with context
            prompt = self._construct_prompt(query, context, image_predictions, symptoms)
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=settings.LLM_MODEL,
                temperature=0.1,
                max_tokens=500
            )
            
            response = chat_completion.choices[0].message.content
            return response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def _construct_prompt(self, query: str, context: Optional[str], 
                         image_predictions: Optional[List[PredictionResult]], 
                         symptoms: Optional[str]) -> str:
        """Construct prompt for medical consultation"""
        prompt_parts = []
        
        # Add context if available
        if context:
            prompt_parts.append(f"MEDICAL CONTEXT:\n{context}\n")
        
        # Add image predictions if available
        if image_predictions:
            preds_text = ", ".join([
                f"{p.condition} ({p.confidence:.2f})"  # Access attributes directly
                for p in image_predictions[:3]
            ])
            prompt_parts.append(f"IMAGE ANALYSIS PREDICTIONS: {preds_text}\n")
        
        # Add symptoms if available
        if symptoms:
            prompt_parts.append(f"PATIENT REPORTED SYMPTOMS: {symptoms}\n")
        
        # Add the main query
        prompt_parts.append(f"QUERY: {query}\n")
        
        # Add instructions
        prompt_parts.append("""
        INSTRUCTIONS:
        - Provide a professional medical assessment
        - Recommend professional medical consultation
        - List potential differential diagnoses
        - Suggest next steps or recommendations
        - Cite sources from medical context
        - Use patient-friendly language
        - Include AI assistance disclaimers
        """)
        
        return "\n".join(prompt_parts)

# Global LLM service instance
llm_service = LLMService()