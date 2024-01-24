import os
from ultralytics import YOLO
import multiprocessing

def train_model():
    model = YOLO("yolov8m.pt")
    model.train(data="custom.yaml", batch=4, imgsz=640, epochs=10, workers=1)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_model()