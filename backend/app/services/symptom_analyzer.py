from typing import Dict, List
import re
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class SymptomAnalyzer:
    def __init__(self):
        self.common_skin_terms = [
            'itch', 'rash', 'redness', 'swelling', 'pain', 'burning',
            'blister', 'scale', 'dry', 'oily', 'peeling', 'discoloration'
        ]
    
    def analyze_symptoms(self, symptoms_text: str, age: int = None, gender: str = None) -> Dict:
        """Analyze symptoms text for key patterns"""
        try:
            symptoms_lower = symptoms_text.lower()
            
            # Extract key symptoms
            found_symptoms = []
            for term in self.common_skin_terms:
                if term in symptoms_lower:
                    found_symptoms.append(term)
            
            # Extract duration if mentioned
            duration_patterns = [
                r'for (\d+ days?)', r'for (\d+ weeks?)', r'for (\d+ months?)',
                r'since (\w+ \d+)', r'since (\d+/\d+/\d+)'
            ]
            
            duration = None
            for pattern in duration_patterns:
                match = re.search(pattern, symptoms_lower)
                if match:
                    duration = match.group(1)
                    break
            
            # Extract severity if mentioned
            severity = "moderate"
            if any(word in symptoms_lower for word in ['severe', 'extreme', 'unbearable']):
                severity = "severe"
            elif any(word in symptoms_lower for word in ['mild', 'slight', 'minor']):
                severity = "mild"
            
            return {
                "symptoms_found": found_symptoms,
                "symptom_count": len(found_symptoms),
                "duration": duration,
                "severity": severity,
                "age": age,
                "gender": gender,
                "processed": True
            }
            
        except Exception as e:
            logger.error(f"Symptom analysis error: {str(e)}")
            return {
                "symptoms_found": [],
                "symptom_count": 0,
                "duration": None,
                "severity": "unknown",
                "age": age,
                "gender": gender,
                "processed": False,
                "error": str(e)
            }
    
    def generate_search_queries(self, symptoms_text: str, age: int = None, gender: str = None) -> List[str]:
        """Generate search queries for medical knowledge base"""
        analysis = self.analyze_symptoms(symptoms_text, age, gender)
        
        queries = []
        if analysis['symptoms_found']:
            base_query = " ".join(analysis['symptoms_found'][:3])
            queries.append(f"{base_query} skin condition")
            
            if age and gender:
                queries.append(f"{base_query} {gender} age {age}")
            elif gender:
                queries.append(f"{base_query} {gender}")
            
            if analysis['duration']:
                queries.append(f"{base_query} duration {analysis['duration']}")
        
        return queries

# Global symptom analyzer instance
symptom_analyzer = SymptomAnalyzer()