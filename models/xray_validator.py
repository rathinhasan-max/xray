"""
X-ray image validation service
Validates whether an uploaded image is a chest X-ray using multiple approaches
"""
import numpy as np
import cv2
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XRayValidator:
    """Validates if an image is a chest X-ray"""
    
    def __init__(self):
        self.min_confidence = 0.5
    
    def validate_image(self, image_path):
        """
        Validate if the image is a chest X-ray
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with validation result and confidence score
        """
        try:
            # Use rule-based validation
            is_valid, confidence = self._rule_based_validation(image_path)
            
            result = {
                'is_xray': is_valid,
                'confidence': confidence,
                'method': 'rule_based'
            }
            
            logger.info(f"X-ray validation: {is_valid} (confidence: {confidence:.2%})")
            return result
            
        except Exception as e:
            logger.error(f"Error during X-ray validation: {str(e)}")
            return {
                'is_xray': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _rule_based_validation(self, image_path):
        """
        Rule-based validation using image characteristics
        
        Chest X-rays typically have:
        - Grayscale or near-grayscale appearance
        - High contrast in certain regions
        - Specific aspect ratio (roughly portrait)
        - Dark background with lighter central region
        
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return False, 0.0
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Check 1: Grayscale similarity (STRICT CHECK)
            # X-rays should have minimal color variation. If it's too colorful, reject immediately.
            color_variance = np.std(img, axis=2).mean()
            if color_variance > 15.0:  # Threshold for "too colorful"
                logger.info(f"Image rejected due to high color variance: {color_variance:.2f}")
                return False, 0.0
                
            grayscale_score = 1.0 - min(color_variance / 15.0, 1.0)
            
            # Check 2: Contrast analysis
            # X-rays typically have good contrast
            contrast = gray.std()
            contrast_score = min(contrast / 80.0, 1.0)
            
            # Check 3: Aspect ratio
            # Chest X-rays are typically portrait or square
            height, width = gray.shape
            aspect_ratio = height / width
            aspect_score = 1.0 if 0.8 <= aspect_ratio <= 1.5 else 0.5
            
            # Check 4: Edge density
            # X-rays have characteristic edge patterns
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            edge_score = min(edge_density / 0.15, 1.0)
            
            # Check 5: Brightness distribution
            # X-rays typically have darker edges and lighter center
            center_region = gray[height//4:3*height//4, width//4:3*width//4]
            edge_region = np.concatenate([
                gray[:height//4, :].flatten(),
                gray[3*height//4:, :].flatten(),
                gray[:, :width//4].flatten(),
                gray[:, 3*width//4:].flatten()
            ])
            
            center_brightness = center_region.mean()
            edge_brightness = edge_region.mean()
            brightness_diff = center_brightness - edge_brightness
            brightness_score = min(max(brightness_diff / 50.0, 0.0), 1.0)
            
            # Check 6: Symmetry (New)
            # Chest X-rays are roughly symmetric
            # Resize to small square to ignore minor details/pathologies
            small = cv2.resize(gray, (64, 64))
            flipped = cv2.flip(small, 1)
            diff = cv2.absdiff(small, flipped)
            symmetry_score = 1.0 - (np.mean(diff) / 255.0)
            # Boost symmetry score if it's reasonably high (X-rays are usually > 0.8)
            if symmetry_score > 0.8:
                symmetry_score = min(symmetry_score * 1.1, 1.0)
            # Check 7: Chest Structure Analysis (New)
            # Detect specific X-ray features: dark corners (shoulders/background) and central spine
            structure_score = 0.0
            
            # 7a. Check top corners (should be dark in X-rays)
            h, w = gray.shape
            corner_size = min(h, w) // 8
            top_left = gray[:corner_size, :corner_size].mean()
            top_right = gray[:corner_size, -corner_size:].mean()
            corners_brightness = (top_left + top_right) / 2.0
            # If corners are very bright (e.g. > 100), it's likely not an X-ray (sky, ceiling)
            corners_score = 1.0 - min(corners_brightness / 100.0, 1.0)
            
            # 7b. Vertical Profile (Spine vs Lungs)
            # Center strip (spine) should be brighter than side strips (lungs)
            mid_x = w // 2
            strip_w = w // 6
            spine_region = gray[:, mid_x-strip_w//2 : mid_x+strip_w//2].mean()
            left_lung = gray[:, :strip_w].mean()
            right_lung = gray[:, -strip_w:].mean()
            lungs_brightness = (left_lung + right_lung) / 2.0
            
            # Spine should be brighter than lungs
            if spine_region > lungs_brightness:
                spine_score = 1.0
            else:
                spine_score = 0.3  # Penalty if spine is darker than lungs
                
            structure_score = (corners_score * 0.6 + spine_score * 0.4)

            # Weighted combination of all scores
            weights = {
                'grayscale': 0.20,
                'contrast': 0.10,
                'aspect': 0.10,
                'edges': 0.10,
                'brightness': 0.15,
                'symmetry': 0.15,
                'structure': 0.20  # High weight for structural correctness
            }
            
            confidence = (
                weights['grayscale'] * grayscale_score +
                weights['contrast'] * contrast_score +
                weights['aspect'] * aspect_score +
                weights['edges'] * edge_score +
                weights['brightness'] * brightness_score +
                weights['symmetry'] * symmetry_score +
                weights['structure'] * structure_score
            )
            
            # Threshold
            self.min_confidence = 0.55
            
            is_valid = confidence >= self.min_confidence
            
            logger.debug(f"Validation scores - Grayscale: {grayscale_score:.2f}, "
                        f"Contrast: {contrast_score:.2f}, Aspect: {aspect_score:.2f}, "
                        f"Edges: {edge_score:.2f}, Brightness: {brightness_score:.2f}, "
                        f"Symmetry: {symmetry_score:.2f}")
            
            return bool(is_valid), float(confidence)
            
        except Exception as e:
            logger.error(f"Error in rule-based validation: {str(e)}")
            return False, 0.0


# Global validator instance
_validator_instance = None


def get_validator():
    """Get or create the global validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = XRayValidator()
    return _validator_instance


def validate_xray(image_path):
    """
    Convenience function to validate an X-ray image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Validation result dictionary
    """
    validator = get_validator()
    return validator.validate_image(image_path)
