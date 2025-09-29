from PIL import Image
import io
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def validate_image(image_data: bytes) -> bool:
    """Validate image data"""
    try:
        img = Image.open(io.BytesIO(image_data))
        img.verify()
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        return False

def get_image_info(image_data: bytes) -> dict:
    """Get image information"""
    try:
        img = Image.open(io.BytesIO(image_data))
        return {
            "format": img.format,
            "size": img.size,
            "mode": img.mode,
            "width": img.width,
            "height": img.height
        }
    except Exception as e:
        logger.error(f"Error getting image info: {str(e)}")
        return {}

def resize_image(image_data: bytes, size: Tuple[int, int] = (224, 224)) -> bytes:
    """Resize image to target size"""
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        
        output = io.BytesIO()
        img.save(output, format='JPEG')
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise