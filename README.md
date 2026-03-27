# ASL Vision — American Sign Language Recognition

Real-time hand gesture recognition for American Sign Language (A–Z) using a 
CNN (EfficientNetB3) trained on the ASL alphabet dataset.

## Dataset
- [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- 87,000 images across 29 classes (A–Z + space, delete, nothing)

## Model
- Architecture: CNN (Conv2D → MaxPool → Dense)
- Trained on Kaggle (GPU P100)
- Accuracy: ~94% on validation set

## Tech stack
- Python, TensorFlow/Keras, OpenCV, MediaPipe
- Trained on Kaggle

## Results
| Class | Precision | Recall |
|-------|-----------|--------|
| A | 0.99 | 0.98 |
| B | 0.98 | 0.99 |
...
