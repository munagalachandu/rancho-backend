from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from fer import FER
import logging
import os

app = Flask(__name__)

# Production-friendly CORS configuration
# Allow all origins in production or specify your frontend URLs
CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins for mobile apps
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FER model (lazy loading for faster startup)
detector = None

def get_detector():
    """Lazy load the FER model"""
    global detector
    if detector is None:
        logger.info("Loading FER model...")
        detector = FER(mtcnn=True)
        logger.info("FER model loaded successfully")
    return detector

@app.route('/', methods=['GET'])
def home():
    """Root endpoint"""
    return jsonify({
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/detect_emotion": "POST - Detect emotion from image"
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Emotion detection API is running"
    }), 200

@app.route('/detect_emotion', methods=['POST', 'OPTIONS'])
def detect_emotion():
    """
    Endpoint to detect emotion from an image
    Expects JSON with base64 encoded image
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Remove data URI prefix if present
        if ',' in image_data:
            header, encoded = image_data.split(",", 1)
        else:
            encoded = image_data
        
        # Decode base64 image
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags=cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        logger.info("Image decoded successfully. Shape: %s", img.shape)
        
        # Get detector instance
        fer_detector = get_detector()
        
        # Detect emotions
        result = fer_detector.detect_emotions(img)
        logger.info("Detection result length: %d", len(result))
        
        if result and len(result) > 0:
            # Get top emotion
            top_emotion, score = fer_detector.top_emotion(img)
            
            # Get all emotions for the first face
            all_emotions = result[0]['emotions']
            
            logger.info("Top emotion: %s (score: %.2f)", top_emotion, score)
            
            return jsonify({
                "success": True,
                "emotion": top_emotion,
                "confidence": float(score),
                "all_emotions": all_emotions,
                "faces_detected": len(result)
            }), 200
        else:
            logger.info("No face detected in image")
            return jsonify({
                "success": False,
                "emotion": "neutral",
                "message": "No face detected in the image",
                "faces_detected": 0
            }), 200
            
    except base64.binascii.Error as e:
        logger.error("Base64 decoding error: %s", str(e))
        return jsonify({"error": "Invalid base64 image data"}), 400
    
    except Exception as e:
        logger.error("Error during processing: %s", str(e))
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    # In production, Render will use gunicorn, so debug should be False
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
