# OpenEmotion architecture

```text
FER-2013
   |
   v
ImageFolder dataset
   |
   v
Resize + augmentation
   |
   v
EfficientNet-B0
   |
   v
7-class emotion logits
   |
   v
Softmax probabilities

Live mode:

Webcam
   |
   v
OpenCV frame capture
   |
   v
Haar face detection
   |
   v
Face crop
   |
   v
OpenEmotion classifier
   |
   v
Emotion + confidence
```

The notebook remains the primary experiment/training record. The Python package provides reusable inference and model-loading components.
