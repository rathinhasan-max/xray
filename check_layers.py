import os
import tensorflow as tf
from tensorflow import keras
from config import MODEL_PATH
from models.model_loader import ChestXRayModel

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def check_model_layers():
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        loader = ChestXRayModel()
        loader.load_model()
        model = loader.model
        
        print("\nModel loaded successfully.")
        print("-" * 50)
        
        print(f"Model Type: {type(model)}")
        print(f"Total Layers: {len(model.layers)}")
        
        print("\n--- Layer List ---")
        for i, layer in enumerate(model.layers):
            print(f"{i}: {layer.name} ({layer.__class__.__name__})")
            
            # If it's a functional model inside, print its layers too
            if hasattr(layer, 'layers') and len(layer.layers) > 1:
                print(f"   [NESTED MODEL DETECTED] - {len(layer.layers)} layers")
                print("   Last 5 layers of nested model:")
                for sub_layer in layer.layers[-5:]:
                    print(f"     - {sub_layer.name} ({sub_layer.__class__.__name__})")

    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model_layers()
