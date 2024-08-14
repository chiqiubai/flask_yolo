# yolo.py
from ultralytics import YOLO

model = YOLO('yolov8n.pt')


def process_frame(frame):
    results = model(frame)
   
    if isinstance(results, list):
        return results[0]
    return results
