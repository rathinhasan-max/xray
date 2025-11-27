# 🫁 Chest X-Ray Disease Detection System

An AI-powered web application for automated detection of infectious diseases in chest X-ray images using deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange.svg)

## 🎯 Features

- **Multi-Disease Detection**: Identifies 5 conditions:
  - Bacterial Pneumonia
  - Viral Pneumonia
  - COVID-19
  - Tuberculosis
  - Normal (Healthy)

- **X-Ray Validation**: Automatically validates if uploaded images are chest X-rays
- **Confidence Scores**: Displays prediction confidence for all disease classes
- **Grad-CAM Visualization**: Highlights affected lung regions for interpretability
- **Prediction History**: Tracks recent predictions with thumbnails and results
- **Modern UI**: Beautiful, responsive interface with smooth animations

## 🏗️ Architecture

```
Backend: Flask REST API
Model: ResNet50V2 (TensorFlow/Keras)
Frontend: Vanilla JavaScript + Modern CSS
Visualization: Grad-CAM heatmaps
Storage: JSON-based history (easily migrated to database)
```

## 📋 Prerequisites

- Python 3.8 or higher
- 8-16 GB RAM
- GPU recommended (but not required)
- Modern web browser

## 🚀 Installation

### 1. Clone or Download the Project

```bash
cd "c:\Users\rathi\OneDrive\Desktop\Gravity Thesis"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Activate the virtual environment:**

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Model File

**IMPORTANT**: Place your trained ResNet50V2 model file in the `models/` directory:

```
models/
└── resnet50v2_chest_xray.h5
```

If your model file has a different name, update the `MODEL_PATH` in `config.py`:

```python
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'your_model_name.h5')
```

## ▶️ Running the Application

### Start the Flask Server

```bash
python app.py
```

You should see output like:
```
INFO:__main__:Initializing model...
INFO:models.model_loader:Loading model from models/resnet50v2_chest_xray.h5
INFO:models.model_loader:Model loaded successfully
INFO:__main__:Model initialized successfully
INFO:__main__:Starting Flask server on 0.0.0.0:5000
```

### Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage Guide

### 1. Upload X-Ray Image

- **Drag and drop** a chest X-ray image onto the upload zone, or
- **Click** the upload zone to browse and select a file
- Supported formats: JPEG, PNG (max 16MB)

### 2. Automatic Analysis

The system will automatically:
1. ✅ Validate if the image is a chest X-ray
2. 🔍 Perform disease classification
3. 📊 Display confidence scores for all classes
4. 🎨 Generate Grad-CAM heatmap visualization
5. 💾 Save to prediction history

### 3. View Results

- **Predicted Disease**: Displayed prominently with confidence percentage
- **Confidence Bars**: Shows probability for each disease class
- **Grad-CAM Heatmap**: Highlights lung regions that influenced the prediction
- **History Panel**: View recent predictions and click to review

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Model settings
MODEL_PATH = 'path/to/your/model.h5'
CLASS_LABELS = ['Bacterial Pneumonia', 'Covid', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Image preprocessing
IMG_SIZE = (224, 224)

# Upload limits
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# History settings
MAX_HISTORY_ITEMS = 20

# Server settings
HOST = '0.0.0.0'
PORT = 5000
```

## 📁 Project Structure

```
Gravity Thesis/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── models/
│   ├── __init__.py
│   ├── model_loader.py         # Model loading and prediction
│   ├── xray_validator.py       # X-ray image validation
│   ├── gradcam.py              # Grad-CAM visualization
│   └── resnet50v2_chest_xray.h5  # Your trained model (add this)
├── services/
│   ├── __init__.py
│   └── history_service.py      # Prediction history management
├── static/
│   ├── css/
│   │   └── style.css           # Modern UI styles
│   └── js/
│       └── app.js              # Frontend application logic
├── templates/
│   └── index.html              # Main HTML template
├── uploads/                    # Temporary upload storage
└── data/
    └── prediction_history.json # Prediction history (auto-created)
```

## 🔌 API Endpoints

### Health Check
```
GET /api/health
```

### Predict Disease
```
POST /api/predict
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
  "success": true,
  "validation": {
    "is_xray": true,
    "confidence": 0.87
  },
  "prediction": {
    "predicted_class": "Normal",
    "confidence": 0.952,
    "all_predictions": {
      "Bacterial Pneumonia": 0.012,
      "Covid": 0.008,
      "Normal": 0.952,
      "Tuberculosis": 0.015,
      "Viral Pneumonia": 0.013
    }
  },
  "gradcam": "data:image/png;base64,..."
}
```

### Get Prediction History
```
GET /api/history?limit=20

Response:
{
  "success": true,
  "history": [...],
  "count": 5
}
```

## 🐛 Troubleshooting

### Model Not Found Error

**Error**: `Model file not found at models/resnet50v2_chest_xray.h5`

**Solution**: Ensure your `.h5` model file is in the `models/` directory with the correct filename.

### TensorFlow/GPU Issues

If you encounter GPU-related errors, TensorFlow will automatically fall back to CPU. For CPU-only installation:

```bash
pip install tensorflow-cpu==2.15.0
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Change the port in `config.py`:
```python
PORT = 5001  # or any available port
```

### Image Upload Fails

- Check file size (must be < 16MB)
- Ensure file format is JPEG or PNG
- Verify the image is a valid chest X-ray

## 🎨 Customization

### Change Color Theme

Edit CSS variables in `static/css/style.css`:

```css
:root {
  --primary-blue: #00a8e8;
  --accent-teal: #00d9ff;
  /* ... modify other colors */
}
```

### Modify Disease Classes

If you retrain the model with different classes, update `config.py`:

```python
CLASS_LABELS = [
    'Your Class 1',
    'Your Class 2',
    # ... add your classes
]
```

## 📊 Model Information

- **Architecture**: ResNet50V2
- **Input Size**: 224×224×3
- **Output**: 5 disease classes
- **Framework**: TensorFlow/Keras
- **Format**: HDF5 (.h5)

## 🔒 Security Notes

- Change `SECRET_KEY` in `config.py` for production
- Implement user authentication for production deployment
- Use HTTPS in production
- Set up proper CORS policies
- Implement rate limiting for API endpoints

## 📝 License

This project is part of a thesis on "Deep Learning Approaches for Automated Detection of Infectious Diseases in Chest Radiographs."

## 👨‍💻 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration settings
3. Ensure all dependencies are installed correctly

## 🙏 Acknowledgments

- ResNet50V2 architecture by Microsoft Research
- Grad-CAM visualization technique
- Public chest X-ray datasets (NIH, COVIDx, Kaggle, etc.)

---

**Note**: This system is designed for research and educational purposes. It should not be used as a replacement for professional medical diagnosis.
