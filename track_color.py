#encoding="utf-8"
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import torch

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
    9: (128, 128, 128)  # Gray for motor
}

# Load YOLOv8 model
model = YOLO('best_track.pt')
#model = YOLO('yolov8s.pt')
# model.half()

#model = YOLO('yolov8n.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Model is running on: {device}")
# Open video file or webcam



"可用摄像头"
cap =cv2.VideoCapture("small_test5.mp4")

# print(cv2.VideoCapture(stream_url).read())
# print(cv2.VideoCapture(stream_url).isOpened)
original_fps = cap.get(cv2.CAP_PROP_FPS)
print("Original Video FPS:", original_fps)
print("Original Video Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Original Video Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# 获取视频的宽、高、帧率等属性
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# # 创建一个 VideoWriter 对象，指定输出文件名、编码格式、帧率和帧尺寸
# output_path = "output/output_video_11_6_2.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID'，根据需要调整编码格式
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Track history storage
track_history = defaultdict(lambda: [])
start_time_fir = time.time()
# Loop through video frames
while cap.isOpened():
    start_time = time.time()
    success, frame = cap.read()

    #print('frame:',frame.shape)

    if success:
        #frame = cv2.resize(frame, (1920, 1080))
        # Run YOLOv8 tracking on the frame
        #results = model.track(frame,persist=True,classes=[2,3,4,5,6,7,8,9],imgsz=[1440,1088],conf=0.01)
        #results = model.track(frame, persist=True, imgsz=[1440, 1088], conf=0.01)
        results = model.track(frame, persist=True, classes=[2, 3, 4, 5, 6, 7, 8, 9],conf=0.1,iou=0.8)
        #print('result99888:',results[0])

        # Check if there are any detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get boxes, track IDs, and classes
            boxes = results[0].boxes.xywh.cpu().numpy()
            #print(boxes)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            # Draw boxes and labels
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                class_name = classnames.get(cls, "Unknown")
                label = f"ID: {track_id} {class_name}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

        # out.write(frame)
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
# out.release()
cv2.destroyAllWindows()


