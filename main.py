from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fer import FER
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()
emotion_detector = FER(mtcnn=True)

# Utility: classify high-risk states like panic
def classify_emotion(emotion: str, score: float):
    if emotion == "fear" and score > 0.6:
        return "Possible Panic/Anxiety"
    elif emotion == "angry" and score > 0.6:
        return "High Stress"
    elif emotion == "sad" and score > 0.6:
        return "Low Mood/Depression Risk"
    elif emotion == "happy":
        return "Positive/Calm"
    else:
        return "Neutral/Uncertain"

@app.post("/detect-emotion/")
async def detect_emotion(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    frame = np.array(image)

    # Detect emotions
    result = emotion_detector.detect_emotions(frame)

    if not result:
        return JSONResponse({"status": "no_face_detected"})

    emotions = result[0]["emotions"]
    dominant_emotion = max(emotions, key=emotions.get)
    score = emotions[dominant_emotion]

    classification = classify_emotion(dominant_emotion, score)

    return {
        "dominant_emotion": dominant_emotion,
        "confidence": score,
        "all_emotions": emotions,
        "mental_health_flag": classification
    }

# Run using: uvicorn app:app --reload