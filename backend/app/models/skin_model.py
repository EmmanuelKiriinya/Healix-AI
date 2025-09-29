import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import json
import numpy as np
from typing import List, Dict
import logging
import os
from app.core.config import settings

logger = logging.getLogger(__name__)

class SkinConditionModel:
    def __init__(self):
        self.model = None
        self.classes = []
        self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self) -> None:
        """Load PyTorch model from .pth file and metadata from JSON"""
        try:
            # Load metadata
            if not os.path.exists(settings.MODEL_METADATA_PATH):
                raise FileNotFoundError(f"Metadata file not found: {settings.MODEL_METADATA_PATH}")
            
            with open(settings.MODEL_METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            
            self.classes = metadata.get('class_names', ['Acne', 'Eczema', 'Impetigo', 'Psoriasis', 'Tinea'])
            logger.info(f"Loaded classes: {self.classes}")
            
            # Load model architecture
            self.model = self._create_model_architecture(metadata)
            
            # Load weights
            if not os.path.exists(settings.MODEL_PATH):
                raise FileNotFoundError(f"Model weights not found: {settings.MODEL_PATH}")
            
            state_dict = torch.load(settings.MODEL_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Create transforms
            self.transform = self._create_transforms(metadata)
            
            logger.info(f"PyTorch model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.setup_fallback_model()
    
    def _create_model_architecture(self, metadata: Dict) -> nn.Module:
        """Create model architecture based on metadata"""
        arch = metadata.get('model_architecture', 'resnet34')
        
        if arch == 'resnet34':
            model = models.resnet34(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(self.classes))
        else:
            model = self._create_simple_cnn(len(self.classes))
        
        return model
    
    def _create_simple_cnn(self, num_classes: int) -> nn.Module:
        """Create a simple CNN as fallback"""
        return nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    
    def _create_transforms(self, metadata: Dict):
        """Create image transforms based on metadata"""
        mean = metadata.get('normalization_mean', [0.485, 0.456, 0.406])
        std = metadata.get('normalization_std', [0.229, 0.224, 0.225])
        input_size = metadata.get('input_size', [224, 224])
        
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def setup_fallback_model(self):
        """Setup fallback if model loading fails"""
        logger.warning("Using fallback model configuration")
        self.classes = ['Acne', 'Eczema', 'Impetigo', 'Psoriasis', 'Tinea']
        self.model = self._create_simple_cnn(len(self.classes))
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_data: bytes) -> torch.Tensor:
        """Preprocess image for model inference"""
        try:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            if self.transform:
                return self.transform(img).unsqueeze(0)
            else:
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                return transform(img).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ValueError(f"Invalid image data: {str(e)}")
    
    def predict(self, image_data: bytes) -> Dict:
        """Predict skin conditions from image data"""
        try:
            input_tensor = self.preprocess_image(image_data)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            
            # Filter predictions by confidence threshold
            confident_indices = np.where(probabilities >= settings.DEFAULT_CONFIDENCE_THRESHOLD)[0]
            
            if len(confident_indices) > 0:
                top_indices = confident_indices[np.argsort(probabilities[confident_indices])[::-1][:3]]
            else:
                top_indices = np.argsort(probabilities)[::-1][:1]
            
            predictions = []
            
            for i, idx in enumerate(top_indices):
                predictions.append({
                    "condition": self.classes[idx],
                    "confidence": float(probabilities[idx]),
                    "rank": i + 1
                })
            
            return {
                "predictions": predictions,
                "all_probabilities": {
                    cls_name: float(probabilities[i]) 
                    for i, cls_name in enumerate(self.classes)
                },
                "model_type": "trained" if hasattr(self, 'model_loaded') else "fallback"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_loaded": self.model is not None,
            "classes": self.classes,
            "device": str(self.device),
            "model_type": "pytorch",
            "status": "operational"
        }

# Global model instance
skin_model = SkinConditionModel()