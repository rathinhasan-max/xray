# Quick Start Guide

## ✅ Setup Complete!

Your virtual environment is ready with all dependencies installed.

---

## 🚀 Running the Application

### Step 1: Add Your Model File

Place your trained ResNet50V2 model in the `models/` folder:

```
models/
└── resnet50v2_chest_xray.h5  ← Your model file here
```

### Step 2: Start the Application

**Option A: Double-click the startup script**
- Double-click `start.bat`
- The application will start automatically

**Option B: Manual start**
```bash
# Activate virtual environment
venv\Scripts\activate

# Run application
python app.py
```

### Step 3: Open in Browser

Navigate to: **http://localhost:5000**

---

## 📋 What's Installed (in venv)

✅ Flask 3.1.2 - Web framework  
✅ TensorFlow 2.20.0 - Deep learning  
✅ OpenCV 4.12.0 - Image processing  
✅ Pillow 12.0.0 - Image handling  
✅ NumPy 2.2.6 - Numerical operations  
✅ scikit-learn 1.7.2 - ML utilities  

All packages are installed **inside the virtual environment** - no global installation!

---

## 🎯 Features

- **5 Disease Detection**: Bacterial Pneumonia, COVID-19, Normal, Tuberculosis, Viral Pneumonia
- **X-Ray Validation**: Automatically rejects non-X-ray images
- **Confidence Scores**: Shows probability for each disease class
- **Grad-CAM Visualization**: Highlights affected lung regions
- **Prediction History**: Tracks recent predictions
- **Modern UI**: Beautiful, responsive interface

---

## 🔧 Troubleshooting

**Issue**: "Model file not found"
- **Solution**: Add your `.h5` model file to the `models/` directory

**Issue**: "Module not found"
- **Solution**: Make sure virtual environment is activated
  ```bash
  venv\Scripts\activate
  ```

**Issue**: "Port 5000 already in use"
- **Solution**: Change port in `config.py`:
  ```python
  PORT = 5001  # or any available port
  ```

---

## 📁 Project Structure

```
Gravity Thesis/
├── venv/                    ← Virtual environment (don't modify)
├── app.py                   ← Main Flask application
├── config.py                ← Configuration settings
├── start.bat                ← Quick start script
├── models/
│   ├── model_loader.py
│   ├── xray_validator.py
│   ├── gradcam.py
│   └── resnet50v2_chest_xray.h5  ← Add your model here
├── services/
│   └── history_service.py
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
└── README.md
```

---

## 🎓 For Your Thesis Defense

1. **Start the application** using `start.bat`
2. **Open http://localhost:5000** on your presentation computer
3. **Upload sample X-rays** to demonstrate predictions
4. **Show Grad-CAM visualizations** for explainability
5. **Explain confidence scores** for each disease class

---

## 💡 Tips

- Test the application before your defense
- Prepare 5-6 sample X-ray images
- Know your model's accuracy metrics
- Practice the demonstration flow
- Have backup screenshots ready

---

**Everything is ready! Just add your model file and run `start.bat`** 🚀
