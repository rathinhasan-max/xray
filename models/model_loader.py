"""
Model loader and prediction service for ResNet50V2 chest X-ray classification
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import os
import logging

from config import MODEL_PATH, CLASS_LABELS, IMG_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChestXRayModel:
    """Wrapper class for the ResNet50V2 chest X-ray disease detection model"""
    
    def __init__(self):
        self.model = None
        self.class_labels = CLASS_LABELS
        
    def load_model(self):
        """Load the pre-trained ResNet50V2 model"""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model file not found at {MODEL_PATH}. "
                    f"Please place your ResNet50V2 model (.h5 file) in the models/ directory."
                )
            
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Define custom TrueDivide layer for compatibility
            class TrueDivide(keras.layers.Layer):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                
                def __call__(self, *args, **kwargs):
                    # Helper to convert scalars to tensors
                    def to_tensor(x):
                        if isinstance(x, (int, float)):
                            return tf.constant(x, dtype=tf.float32)
                        elif isinstance(x, (list, tuple)):
                            return [to_tensor(i) for i in x]
                        return x

                    # Convert all arguments
                    new_args = [to_tensor(arg) for arg in args]
                    new_kwargs = {k: to_tensor(v) for k, v in kwargs.items()}
                    
                    # If we have more than 1 positional arg, move the second one to kwargs 'y'
                    # This satisfies Keras 3 strictness about positional args being KerasTensors
                    if len(new_args) > 1:
                        inputs = new_args[0]
                        y = new_args[1]
                        new_kwargs['y'] = y
                        new_args = [inputs]
                    
                    return super().__call__(*new_args, **new_kwargs)
                    
                def call(self, inputs, y=None):
                    # Handle inputs being passed as a list/tuple (when y is None)
                    if y is None:
                        if isinstance(inputs, (list, tuple)):
                            x, y = inputs
                        else:
                            # Should not happen if logic is correct
                            raise ValueError(f"Unexpected inputs to TrueDivide: {inputs}")
                    else:
                        x = inputs
                        
                    return tf.math.divide(x, y)
                    
                def get_config(self):
                    return super().get_config()

            # Define custom Sub layer for compatibility
            class Sub(keras.layers.Layer):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                
                def __call__(self, *args, **kwargs):
                    # Helper to convert scalars to tensors
                    def to_tensor(x):
                        if isinstance(x, (int, float)):
                            return tf.constant(x, dtype=tf.float32)
                        elif isinstance(x, (list, tuple)):
                            return [to_tensor(i) for i in x]
                        return x

                    print(f"DEBUG: Sub called with args: {args}", flush=True)
                    
                    # Convert all arguments
                    new_args = [to_tensor(arg) for arg in args]
                    new_kwargs = {k: to_tensor(v) for k, v in kwargs.items()}
                    
                    # If we have more than 1 positional arg, move the second one to kwargs 'y'
                    if len(new_args) > 1:
                        inputs = new_args[0]
                        y = new_args[1]
                        new_kwargs['y'] = y
                        new_args = [inputs]
                    
                    return super().__call__(*new_args, **new_kwargs)
                    
                def call(self, inputs, y=None):
                    if y is None:
                        if isinstance(inputs, (list, tuple)):
                            x, y = inputs
                        else:
                            raise ValueError(f"Unexpected inputs to Sub: {inputs}")
                    else:
                        x = inputs
                    return tf.math.subtract(x, y)

                def get_config(self):
                    return super().get_config()
            
            # Generic custom layer for debugging and fixing inputs
            class GenericOp(keras.layers.Layer):
                def __init__(self, op_name, op_func, **kwargs):
                    super().__init__(**kwargs)
                    self.op_name = op_name
                    self.op_func = op_func
                
                def __call__(self, *args, **kwargs):
                    def to_tensor(x):
                        if isinstance(x, (int, float)):
                            return tf.constant(x, dtype=tf.float32)
                        elif isinstance(x, (list, tuple)):
                            return [to_tensor(i) for i in x]
                        return x

                    new_args = [to_tensor(arg) for arg in args]
                    new_kwargs = {k: to_tensor(v) for k, v in kwargs.items()}
                    
                    if len(new_args) > 1:
                        inputs = new_args[0]
                        y = new_args[1]
                        new_kwargs['y'] = y
                        new_args = [inputs]
                    
                    return super().__call__(*new_args, **new_kwargs)
                    
                def call(self, inputs, y=None):
                    if y is None:
                        if isinstance(inputs, (list, tuple)):
                            x, y = inputs
                        else:
                            # Fallback for single input (shouldn't happen for binary ops)
                            return inputs
                    else:
                        x = inputs
                    return self.op_func(x, y)

                def get_config(self):
                    return super().get_config()

            # Define specific classes for registration
            class BiasAdd(GenericOp):
                def __init__(self, **kwargs):
                    super().__init__("BiasAdd", tf.nn.bias_add, **kwargs)

            class Multiply(GenericOp):
                def __init__(self, **kwargs):
                    super().__init__("Multiply", tf.math.multiply, **kwargs)

            class Add(GenericOp):
                def __init__(self, **kwargs):
                    super().__init__("Add", tf.math.add, **kwargs)
            
            class Subtract(GenericOp):
                def __init__(self, **kwargs):
                    super().__init__("Subtract", tf.math.subtract, **kwargs)

            # Define custom objects for loading
            custom_objects = {
                'TrueDivide': TrueDivide, 
                'Sub': Sub,
                'BiasAdd': BiasAdd,
                'Multiply': Multiply,
                'Add': Add,
                'Subtract': Subtract
            }
            
            with keras.utils.custom_object_scope(custom_objects):
                self.model = keras.models.load_model(MODEL_PATH)
                
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for model prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for prediction
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to model input size
            img = img.resize(IMG_SIZE)
            
            # Convert to array
            img_array = np.array(img)
            
            # Ensure image is float32
            img_array = img_array.astype('float32')
            
            # Use ResNet50V2 specific preprocessing (scales to -1 to 1)
            img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image_path):
        """
        Perform prediction on a chest X-ray image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results with confidence scores
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            
            # Preprocess image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get confidence scores for all classes
            confidence_scores = predictions[0]
            
            # Get predicted class
            predicted_class_idx = np.argmax(confidence_scores)
            predicted_class = self.class_labels[predicted_class_idx]
            predicted_confidence = float(confidence_scores[predicted_class_idx])
            
            # Create results dictionary
            results = {
                'predicted_class': predicted_class,
                'predicted_class_index': int(predicted_class_idx),
                'confidence': predicted_confidence,
                'all_predictions': {
                    label: float(score) 
                    for label, score in zip(self.class_labels, confidence_scores)
                }
            }
            
            logger.info(f"Prediction: {predicted_class} ({predicted_confidence:.2%})")
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise


# Global model instance
_model_instance = None


def get_model():
    """Get or create the global model instance"""
    global _model_instance
    if _model_instance is None:
        _model_instance = ChestXRayModel()
        _model_instance.load_model()
    return _model_instance


def load_model():
    """Initialize and load the model"""
    return get_model()


def predict_image(image_path):
    """
    Convenience function to predict on an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Prediction results dictionary
    """
    model = get_model()
    return model.predict(image_path)
