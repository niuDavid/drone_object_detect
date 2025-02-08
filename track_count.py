from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time
import torch
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Chinese class names
# chinese_names = ['行人', '人', '自行车', '汽车', '面包车', '卡车', '三轮车', '遮阳三轮车', '公共汽车', '摩托车']
#
#
# def en_to_ch(chinese_names, weights):
#     # 读取权重文件
#     weights_dict = torch.load(weights)
#     # 将原来的英文标签，替换为中文标签
#     weights_dict['model'].names = chinese_names
#     # 最后保存到原文件中
#     torch.save(weights_dict, weights)
#
#
# model_path = Path("E:\\small_object\\ultralytics\\best_track.pt")
# en_to_ch(chinese_names, model_path)

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
    0: (0, 255, 0),  # Green for pedestrian
    1: (255, 0, 0),  # Blue for person
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
model = YOLO('weights/best_track.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Model is running on: {device}")


# def count_vehicles(classes):
#     """Count the number of vehicles in the given classes list."""
#     vehicle_classes = [2, 3, 4, 5, 6, 7, 8, 9]  # Classes considered as vehicles
#     vehicle_count = sum(cls in vehicle_classes for cls in classes)
#     return vehicle_count

def people_count(classes):
    """Count the number of vehicles in the given classes list."""
    people_classes = [0,1]  # Classes considered as vehicles
    people_count = sum(cls in people_classes for cls in classes)
    return people_count


def process_video(video_path):
    # Open video file or webcam
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    logging.info(f"Original Video FPS: {original_fps}")
    logging.info(f"Original Video Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    logging.info(f"Original Video Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    #track_history = defaultdict(lambda: [])
    start_time_fir = time.time()
    frame_count1=0


    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()

        if success:
            frame_count1 += 1
            if frame_count1 % 1 != 0:
                continue

            frame = cv2.resize(frame, (1280, 720))
            results = model.track(frame, persist=True, classes=[0,1], imgsz=960, conf=0.01)
            #results = model.track(frame, persist=True, imgsz=960, conf=0.01)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                classes = results[0].boxes.cls.int().cpu().tolist()

                vehicle_count = people_count(classes)

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    x, y, w, h = box
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), colors[cls], 2)
                    class_name = classnames.get(cls, "Unknown")
                    label = f"ID: {track_id} {class_name}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], 2)

                cv2.putText(frame, f"people Count: {vehicle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)

                print(vehicle_count)
            total_time = time.time() - start_time_fir
            logging.info(f"Total video playback time: {total_time}")

            elapsed_time = time.time() - start_time
            fps = 1 / elapsed_time
            logging.info(f"Processed FPS: {fps:.2f}")

            cv2.imshow("YOLOv8 Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# Process the video
process_video("videos/people3.mp4")


