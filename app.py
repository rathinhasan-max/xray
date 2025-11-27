"""
Flask application for Chest X-Ray Disease Detection
Main server with API endpoints for image upload, prediction, and history
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import logging
from datetime import datetime

from config import (
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_FILE_SIZE,
    SECRET_KEY, DEBUG, HOST, PORT
)
from models.model_loader import get_model, predict_image
from models.xray_validator import validate_xray
from models.gradcam import generate_gradcam
from services.history_service import add_to_history, get_prediction_history

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Enable CORS
CORS(app)

# Global model instance (loaded on startup)
model = None


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def cleanup_old_uploads():
    """Clean up old uploaded files to save space"""
    try:
        upload_dir = app.config['UPLOAD_FOLDER']
        if not os.path.exists(upload_dir):
            return
        
        # Delete files older than 1 hour
        current_time = datetime.now().timestamp()
        for filename in os.listdir(upload_dir):
            if filename == '.gitkeep':
                continue
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > 3600:  # 1 hour
                    os.remove(filepath)
                    logger.info(f"Cleaned up old file: {filename}")
    except Exception as e:
        logger.error(f"Error cleaning up uploads: {str(e)}")


@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts image upload, validates it's an X-ray, and returns prediction results
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        # Step 1: Validate if it's an X-ray image
        validation_result = validate_xray(filepath)
        
        if not validation_result['is_xray']:
            # Clean up the file
            os.remove(filepath)
            return jsonify({
                'error': 'The uploaded image does not appear to be a chest X-ray',
                'validation_confidence': validation_result['confidence'],
                'suggestion': 'Please upload a valid chest X-ray image'
            }), 400
        
        logger.info(f"X-ray validation passed (confidence: {validation_result['confidence']:.2%})")
        
        # Step 2: Perform disease prediction
        prediction_result = predict_image(filepath)
        
        logger.info(f"Prediction: {prediction_result['predicted_class']} "
                   f"({prediction_result['confidence']:.2%})")
        
        # Step 3: Generate Grad-CAM visualization
        gradcam_image = None
        try:
            img_array = model.preprocess_image(filepath)
            gradcam_image = generate_gradcam(
                model.model,
                filepath,
                img_array,
                prediction_result['predicted_class_index']
            )
            logger.info("Grad-CAM visualization generated")
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {str(e)}")
            # Continue without Grad-CAM if it fails
        
        # Step 4: Add to prediction history
        try:
            add_to_history(filepath, prediction_result, gradcam_image)
            logger.info("Prediction added to history")
        except Exception as e:
            logger.error(f"Error adding to history: {str(e)}")
            # Continue even if history fails
        
        # Step 5: Prepare response
        response = {
            'success': True,
            'validation': {
                'is_xray': validation_result['is_xray'],
                'confidence': validation_result['confidence']
            },
            'prediction': {
                'predicted_class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'all_predictions': prediction_result['all_predictions']
            },
            'gradcam': gradcam_image,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up old files
        cleanup_old_uploads()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred during prediction',
            'details': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    try:
        limit = request.args.get('limit', type=int, default=20)
        history = get_prediction_history(limit=limit)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return jsonify({
            'error': 'Error retrieving history',
            'details': str(e)
        }), 500


@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large',
        'max_size': f'{MAX_FILE_SIZE / (1024 * 1024):.0f}MB'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500


def initialize_model():
    """Initialize the model on application startup"""
    global model
    try:
        logger.info("Initializing model...")
        model = get_model()
        logger.info("Model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        logger.warning("Application will start but predictions will fail until model is loaded")
        return False


if __name__ == '__main__':
    # Initialize model before starting server
    initialize_model()
    
    # Start Flask server
    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
