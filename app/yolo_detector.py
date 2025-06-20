from ultralytics import YOLO
import cv2

model = YOLO("runs/myfood2/weights/best.pt")

def detect_food_objects(image_path):
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(label)
    return list(set(detections))
