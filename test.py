from ultralytics import YOLO
import cv2
import PIL
model = YOLO("best.pt")
model.predict(source=r"E:\yolo\val\images\08img31.png", show=True, save=True, conf=0.2, save_txt=True, save_crop=True)


cv2.waitKey(0)