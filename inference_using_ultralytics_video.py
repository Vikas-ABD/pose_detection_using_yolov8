import cv2
import random
import pathlib
from ultralytics import YOLO
import numpy as np

def yolov8_detection(model, frame):
    # Update object localizer
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Convert the annotated frame to a NumPy array
    return np.array(annotated_frame)

def video_inference(model_path, video_path, output_path):
    model = YOLO(model_path)

    video_cap = cv2.VideoCapture(video_path)
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while video_cap.isOpened():
        ret, frame = video_cap.read()
        if not ret:
            break

        annotated_frame = yolov8_detection(model, frame)

        # Save each frame with detections
        out.write(annotated_frame)

        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
model_path = 'yolov8x-pose.pt'
video_path = 'input.mp4'
output_path = 'output_video_with_detections.avi'
video_inference(model_path, video_path, output_path)