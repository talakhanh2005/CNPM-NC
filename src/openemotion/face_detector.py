import cv2


def create_face_detector():
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    if detector.empty():
        raise RuntimeError("Could not load OpenCV Haar cascade.")
    return detector
