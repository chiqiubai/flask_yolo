# video_processing.py
import cv2
import time
from yolo import process_frame


def capture_frames(video_source, frame_callback, stop_event):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        # 处理帧
        results = process_frame(frame)

        # 回调处理结果
        frame_callback(results)
        # 每分钟截取一帧
        time.sleep(1)

    cap.release()
