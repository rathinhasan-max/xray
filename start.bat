@echo off
echo ============================================================
echo Chest X-Ray Disease Detection - Starting Application
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [OK] Virtual environment activated
echo.

REM Check if model file exists
if not exist "models\resnet50v2_chest_xray.h5" (
    echo [WARNING] Model file not found!
    echo.
    echo Please add your trained model file to:
    echo   models\resnet50v2_chest_xray.h5
    echo.
    echo If your model has a different name, update config.py
    echo.
    pause
    exit /b 1
)

echo [OK] Model file found
echo.

REM Start Flask application
echo Starting Flask server...
echo.
echo Once started, open your browser and go to:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ============================================================
start http://localhost:5000
python app.py

pause
