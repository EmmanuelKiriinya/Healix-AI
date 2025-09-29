import assemblyai as aai
import aiohttp
import aiofiles
import os
import tempfile
from typing import Optional, Dict
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        self.client = None
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize AssemblyAI client"""
        try:
            if settings.ASSEMBLYAI_API_KEY:
                aai.settings.api_key = settings.ASSEMBLYAI_API_KEY
                self.client = aai
                logger.info("AssemblyAI client initialized successfully")
            else:
                logger.warning("ASSEMBLYAI_API_KEY not set. Voice service will be disabled.")
                self.client = None
        except Exception as e:
            logger.error(f"Error initializing AssemblyAI client: {str(e)}")
            self.client = None
    
    async def transcribe_audio(self, audio_data: bytes, filename: str) -> Dict:
        """Transcribe audio file to text using AssemblyAI"""
        if not self.client:
            return {
                "success": False,
                "error": "Voice service not configured. Please set ASSEMBLYAI_API_KEY.",
                "text": ""
            }
        
        try:
            # Check file size
            if len(audio_data) > settings.MAX_AUDIO_SIZE:
                return {
                    "success": False,
                    "error": f"Audio file too large. Maximum size is {settings.MAX_AUDIO_SIZE} bytes.",
                    "text": ""
                }

            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(tmp_file_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    return {
                        "success": False,
                        "error": transcript.error,
                        "text": ""
                    }
                else:
                    return {
                        "success": True,
                        "text": transcript.text,
                        "confidence": transcript.confidence if hasattr(transcript, 'confidence') else None,
                        "words": [word.text for word in transcript.words] if hasattr(transcript, 'words') else []
                    }
                    
            finally:
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    def is_audio_file(self, filename: str) -> bool:
        """Check if file is a supported audio format"""
        if not filename:
            return False
        ext = filename.lower().split('.')[-1]
        return ext in settings.SUPPORTED_AUDIO_FORMATS

# Global voice service instance
voice_service = VoiceService()