import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.model_loader import get_model
from models.gradcam import GradCAM
from config import IMG_SIZE

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys

def debug_gradcam():
    with open('debug_log.txt', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        
        print("Loading model...")
        try:
            model_wrapper = get_model()
            model = model_wrapper.model
            
            # Create a dummy image (random noise or black)
            img_array = np.random.random((1, 224, 224, 3)).astype('float32')
            img_array = tf.keras.applications.resnet_v2.preprocess_input(img_array * 255)
            
            print("\nInitializing GradCAM...")
            gradcam = GradCAM(model)
            print(f"Target Layer: {gradcam.layer_name}")
            if gradcam.nested_model:
                print(f"Nested Model: {gradcam.nested_model.name}")
            
            print("\nGenerating Heatmap...")
            
            # We need to manually replicate the logic inside generate_heatmap to debug it
            pred_index = 0 # Dummy class index
            
            if gradcam.nested_model:
                target_layer = gradcam.nested_model.get_layer(gradcam.layer_name)
                target_output = target_layer.output
                
                print(f"Constructing grad_model with inputs={model.inputs} and outputs=[{target_output}, {model.output}]")
                
                grad_model = keras.models.Model(
                    inputs=model.inputs,
                    outputs=[target_output, model.output]
                )
            else:
                grad_model = keras.models.Model(
                    inputs=[model.inputs],
                    outputs=[model.get_layer(gradcam.layer_name).output, model.output]
                )
                
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_channel = predictions[:, pred_index]
            
            print(f"Predictions shape: {predictions.shape}")
            print(f"Conv outputs shape: {conv_outputs.shape}")
            
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is None:
                print("!!! GRADIENTS ARE NONE !!!")
                print("This means the graph is disconnected or the target layer does not contribute to the output.")
            else:
                print(f"Gradients shape: {grads.shape}")
                print(f"Gradients mean: {np.mean(grads)}")
                print(f"Gradients max: {np.max(grads)}")
                
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                print(f"Pooled grads shape: {pooled_grads.shape}")
                
                conv_outputs = conv_outputs[0]
                heatmap = np.mean(conv_outputs * pooled_grads.numpy(), axis=-1)
                heatmap = np.maximum(heatmap, 0)
                
                print(f"Heatmap mean: {np.mean(heatmap)}")
                print(f"Heatmap max: {np.max(heatmap)}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

if __name__ == "__main__":
    debug_gradcam()
