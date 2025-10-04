from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
from fer import FER
import logging

app = Flask(__name__)

CORS(app, 
     resources={r"/*": {"origins": "*"}},  # Allow all origins for mobile apps
     supports_credentials=True,
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the FER model
detector = FER(mtcnn=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Emotion detection API is running"}), 200

@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    """
    Endpoint to detect emotion from an image
    Expects JSON with base64 encoded image
    """
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
        
        # Detect emotions
        result = detector.detect_emotions(img)
        logger.info("Detection result length: %d", len(result))
        
        if result and len(result) > 0:
            # Get top emotion
            top_emotion, score = detector.top_emotion(img)
            
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

if __name__ == '__main__':
    # For production, use a proper WSGI server like gunicorn
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(host='0.0.0.0', port=5000)
