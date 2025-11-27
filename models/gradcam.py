"""
Grad-CAM (Gradient-weighted Class Activation Mapping) visualization generator
Highlights regions of the X-ray image that influenced the model's prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import base64
from io import BytesIO
import logging

from config import IMG_SIZE, GRADCAM_LAYER_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradCAM:
    """Generate Grad-CAM heatmaps for model interpretability"""
    
    def __init__(self, model, layer_name=GRADCAM_LAYER_NAME):
        """
        Initialize Grad-CAM generator
        
        Args:
            model: Trained Keras model
            layer_name: Name of the convolutional layer to use for Grad-CAM
        """
        self.model = model
        self.layer_name = layer_name
        self.nested_model = None
        
        # Check if layer exists in top model
        if self._layer_exists(self.model, layer_name):
            logger.info(f"Layer {layer_name} found in top-level model")
            return
            
        # Check for nested model (common in transfer learning)
        logger.info(f"Layer {layer_name} not found in top level. Checking nested models...")
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # It's a nested model (Functional or Sequential)
                if self._layer_exists(layer, layer_name):
                    logger.info(f"Layer {layer_name} found in nested model: {layer.name}")
                    self.nested_model = layer
                    return
        
        # If still not found, try to find *any* conv layer
        logger.warning(f"Layer {layer_name} not found. Searching for alternative...")
        self.layer_name = self._find_last_conv_layer()
        logger.warning(f"Using alternative layer: {self.layer_name}")

    def _layer_exists(self, model_obj, layer_name):
        """Check if a layer exists in a specific model object"""
        try:
            model_obj.get_layer(layer_name)
            return True
        except:
            return False
    
    def _find_last_conv_layer(self):
        """Find the last convolutional layer in the model (recursive)"""
        # Check nested models first as they likely contain the feature extractor
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for sub_layer in reversed(layer.layers):
                    if 'conv' in sub_layer.name.lower() or 'block' in sub_layer.name.lower():
                        if 'out' in sub_layer.name.lower() or 'add' in sub_layer.name.lower():
                             # Prefer output blocks
                             self.nested_model = layer
                             return sub_layer.name
        
        # Check top level
        for layer in reversed(self.model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
                
        return None
    
    def generate_heatmap(self, img_array, pred_index):
        """
        Generate Grad-CAM heatmap
        """
        try:
            if self.layer_name is None:
                logger.warning("No convolutional layer available for Grad-CAM")
                return np.zeros(IMG_SIZE, dtype=np.uint8)
            
            # Different approach for nested vs non-nested models
            if self.nested_model:
                # For nested models, we need to be more careful about graph construction
                # We'll use a functional approach that traces through the actual computation
                
                # Get the target layer from the nested model
                target_layer = self.nested_model.get_layer(self.layer_name)
                
                # Create a new model that goes: top_model_input -> nested_model -> target_layer_output
                # We do this by calling the nested model and extracting the intermediate output
                @tf.function
                def get_conv_output_and_prediction(inputs):
                    # This function will be traced by GradientTape
                    # We need to manually call layers to get intermediate outputs
                    
                    # Process through the layers before nested_model
                    x = inputs
                    for layer in self.model.layers:
                        if layer == self.nested_model:
                            # Now we're at the nested model - get the target layer output
                            # We need to create a sub-model from nested_model input to target layer
                            intermediate_model = keras.models.Model(
                                inputs=self.nested_model.input,
                                outputs=target_layer.output
                            )
                            conv_output = intermediate_model(x)
                            # Continue with the rest of the nested model
                            x = self.nested_model(x)
                            break
                        else:
                            x = layer(x)
                    
                    # Continue with remaining layers after nested_model
                    found_nested = False
                    for layer in self.model.layers:
                        if found_nested:
                            x = layer(x)
                        elif layer == self.nested_model:
                            found_nested = True
                    
                    return conv_output, x
                
                # Use GradientTape with the custom function
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = get_conv_output_and_prediction(img_array)
                    class_channel = predictions[:, pred_index]
                
            else:
                # For non-nested models, use the standard approach
                grad_model = keras.models.Model(
                    inputs=[self.model.inputs],
                    outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
                )
                
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    class_channel = predictions[:, pred_index]
            
            # Extract the gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is None:
                logger.warning("Gradients are None - graph may be disconnected")
                return np.zeros(IMG_SIZE, dtype=np.uint8)
            
            # Compute the guided gradients
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the feature map by the gradients
            conv_outputs = conv_outputs[0]
            pooled_grads = pooled_grads.numpy()
            conv_outputs = conv_outputs.numpy()
            
            for i in range(len(pooled_grads)):
                conv_outputs[:, :, i] *= pooled_grads[i]
            
            # Create the heatmap
            heatmap = np.mean(conv_outputs, axis=-1)
            
            # Normalize the heatmap
            heatmap = np.maximum(heatmap, 0)  # ReLU
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            
            # Resize heatmap to match input image size
            heatmap = cv2.resize(heatmap, IMG_SIZE)
            
            # Convert to uint8
            heatmap = np.uint8(255 * heatmap)
            
            return heatmap
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM heatmap: {str(e)}")
            import traceback
            traceback.print_exc()
            return np.zeros(IMG_SIZE, dtype=np.uint8)
    
    def overlay_heatmap(self, heatmap, original_image_path, alpha=0.4):
        """
        Overlay heatmap on original image
        
        Args:
            heatmap: Grad-CAM heatmap array
            original_image_path: Path to the original image
            alpha: Transparency of the heatmap overlay (0-1)
            
        Returns:
            Base64 encoded image string for web display
        """
        try:
            # Load original image
            img = Image.open(original_image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize(IMG_SIZE)
            img = np.array(img)
            
            # Apply colormap to heatmap (use JET colormap for medical imaging)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
            
            # Convert to PIL Image
            overlayed_img = Image.fromarray(overlayed.astype('uint8'))
            
            # Convert to base64 for web display
            buffered = BytesIO()
            overlayed_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Error overlaying heatmap: {str(e)}")
            return None


def generate_gradcam(model, image_path, img_array, pred_index):
    """
    Convenience function to generate Grad-CAM visualization
    
    Args:
        model: Trained Keras model
        image_path: Path to the original image
        img_array: Preprocessed image array
        pred_index: Index of the predicted class
        
    Returns:
        Base64 encoded Grad-CAM overlay image
    """
    try:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate_heatmap(img_array, pred_index)
        overlay = gradcam.overlay_heatmap(heatmap, image_path)
        return overlay
    except Exception as e:
        logger.error(f"Error in Grad-CAM generation: {str(e)}")
        return None
