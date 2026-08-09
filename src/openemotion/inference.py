import cv2
import torch

from .config import CLASS_NAMES
from .preprocessing import INFERENCE_TRANSFORM


def predict_emotion(model, face_bgr, device):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = INFERENCE_TRANSFORM(face_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return CLASS_NAMES[prediction.item()], confidence.item()


def detect_emotions(model, detector, frame, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    results = []

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(model, face, device)

        results.append({
            "box": [int(x), int(y), int(w), int(h)],
            "emotion": emotion,
            "confidence": float(confidence),
        })

    return results
