"""
Prediction history service
Manages storage and retrieval of prediction history
"""
import json
import os
from datetime import datetime
import base64
from PIL import Image
from io import BytesIO
import logging

from config import HISTORY_FILE, MAX_HISTORY_ITEMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoryService:
    """Manages prediction history storage and retrieval"""
    
    def __init__(self):
        self.history_file = HISTORY_FILE
        self._ensure_history_file()
    
    def _ensure_history_file(self):
        """Create history file if it doesn't exist"""
        if not os.path.exists(self.history_file):
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump([], f)
    
    def _image_to_base64(self, image_path, max_size=(150, 150)):
        """
        Convert image to base64 thumbnail for storage
        
        Args:
            image_path: Path to the image
            max_size: Maximum thumbnail size
            
        Returns:
            Base64 encoded image string
        """
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            return None
    
    def add_prediction(self, image_path, prediction_result, gradcam_image=None):
        """
        Add a new prediction to history
        
        Args:
            image_path: Path to the uploaded image
            prediction_result: Dictionary containing prediction results
            gradcam_image: Base64 encoded Grad-CAM overlay image
            
        Returns:
            Updated history list
        """
        try:
            # Load existing history
            history = self.get_history()
            
            # Create thumbnail of original image
            thumbnail = self._image_to_base64(image_path)
            
            # Create history entry
            entry = {
                'id': datetime.now().strftime('%Y%m%d%H%M%S%f'),
                'timestamp': datetime.now().isoformat(),
                'thumbnail': thumbnail,
                'predicted_class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'all_predictions': prediction_result['all_predictions'],
                'gradcam': gradcam_image
            }
            
            # Add to beginning of history
            history.insert(0, entry)
            
            # Limit history size
            history = history[:MAX_HISTORY_ITEMS]
            
            # Save updated history
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info(f"Added prediction to history: {entry['predicted_class']}")
            return history
            
        except Exception as e:
            logger.error(f"Error adding prediction to history: {str(e)}")
            return []
    
    def get_history(self, limit=None):
        """
        Retrieve prediction history
        
        Args:
            limit: Maximum number of items to return (None for all)
            
        Returns:
            List of history entries
        """
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
            
            if limit:
                history = history[:limit]
            
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []
    
    def clear_history(self):
        """Clear all prediction history"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump([], f)
            logger.info("Prediction history cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing history: {str(e)}")
            return False
    
    def get_prediction_by_id(self, prediction_id):
        """
        Get a specific prediction by ID
        
        Args:
            prediction_id: ID of the prediction
            
        Returns:
            Prediction entry or None if not found
        """
        history = self.get_history()
        for entry in history:
            if entry['id'] == prediction_id:
                return entry
        return None


# Global history service instance
_history_instance = None


def get_history_service():
    """Get or create the global history service instance"""
    global _history_instance
    if _history_instance is None:
        _history_instance = HistoryService()
    return _history_instance


def add_to_history(image_path, prediction_result, gradcam_image=None):
    """
    Convenience function to add a prediction to history
    
    Args:
        image_path: Path to the uploaded image
        prediction_result: Dictionary containing prediction results
        gradcam_image: Base64 encoded Grad-CAM overlay image
        
    Returns:
        Updated history list
    """
    service = get_history_service()
    return service.add_prediction(image_path, prediction_result, gradcam_image)


def get_prediction_history(limit=None):
    """
    Convenience function to get prediction history
    
    Args:
        limit: Maximum number of items to return
        
    Returns:
        List of history entries
    """
    service = get_history_service()
    return service.get_history(limit)
