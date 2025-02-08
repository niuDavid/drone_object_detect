#encoding="utf-8"
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import torch
import logging
from pathlib import Path
# # Class names mapping
# classnames = {
#     0: "pedestrian",
#     1: "people",
#     2: "bicycle",
#     3: "car",
#     4: "van",
#     5: "truck",
#     6: "tricycle",
#     7: "awning-tricycle",
#     8: "bus",
#     9: "motor"
# }

# Class names mapping
classnames = {
    0: "行人",
    1: "人",
    2: "自行车",
    3: "汽车",
    4: "面包车",
    5: "卡车",
    6: "三轮车",
    7: "遮阳三轮车",
    8: "公共汽车",
    9: "摩托"
}

# Class colors mapping
colors = {
    0: (0, 255, 0),  # Blue for pedestrian
    1: (255, 0, 0),  # Green for person
    2: (0, 0, 255),  # Red for bicycle
    3: (255, 255, 0),  # Cyan for car
    4: (255, 0, 255),  # Magenta for van
    5: (0, 255, 255),  # Yellow for truck
    6: (128, 0, 128),  # Purple for tricycle
    7: (128, 128, 0),  # Olive for awning-tricycle
    8: (0, 128, 128),  # Teal for bus
    9: (255, 0, 0)  # Gray for motor
}

# Load YOLOv8 model
model = YOLO('weights/best_track.pt')
# model = YOLO('best_track.pt')
# model.half()

#model = YOLO('yolov8n.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model is running on: {device}")

"可用摄像头"
cap =cv2.VideoCapture("videos/small_test5.mp4")
# print(cv2.VideoCapture(stream_url).read())
# print(cv2.VideoCapture(stream_url).isOpened)
original_fps = cap.get(cv2.CAP_PROP_FPS)
print("Original Video FPS:", original_fps)
print("Original Video Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Original Video Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#
# Track history storage
frame_count1 = 0
track_history = defaultdict(lambda: [])
start_time_fir = time.time()
# Loop through video frames
while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()

    #print('frame:',frame.shape)

    if success:

        frame_count1 += 1
        if frame_count1 % 1 != 0:
            continue
        #frame = cv2.resize(frame, (1920, 1080))
        # Run YOLOv8 tracking on the frame
        #results = model.track(frame,persist=True,classes=[2,3,4,5,6,7,8,9],imgsz=[1440,1088],conf=0.01)
        #results = model.track(frame, persist=True, imgsz=[1440, 1088], conf=0.01)
        results = model.predict(frame, conf=0.25, iou=0.1)
        print("******加载成功88888888")
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            classes = results[0].boxes.cls.int().cpu().tolist()
            confidences = results[0].boxes.conf.cpu().numpy()


            for box, cls, conf in zip(boxes, classes, confidences):
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                class_name = classnames.get(cls, "未知")
                label = f"类别: {class_name} 置信度: {conf:.2f}"
                # label = f"类别: {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

            # cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
            #             (0, 0, 255), 2)
        total_time = time.time() - start_time_fir
        print("Total video playback time:", total_time)

        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time
        print(f"Processed FPS: {fps:.2f}")
        # Display the frame
        cv2.imshow("YOLOv8 Tracking", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()

