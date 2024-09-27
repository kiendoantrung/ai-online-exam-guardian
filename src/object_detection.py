import time
import cv2

from ultralytics import YOLO

model = YOLO('./weights/yolov8n.pt')

def detect_objects(image):
    results = model(image)
    
    detected_objects = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if (class_name == 'person') or (class_name == 'cell phone' and conf > 0.5):
                detected_objects.append((x1, y1, x2, y2, conf, cls))
    
    return detected_objects

def warning(detected_objects):
    class_counts = {}
    for obj in detected_objects:
        cls = obj[5]
        class_name = model.names[cls]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    phone_warning = ''
    person_warning = ''
    
    if 'cell phone' in class_counts:
        phone_warning = 'Warning: phone detected'
    
    person_count = class_counts.get('person', 0)
    if person_count == 0:
        person_warning = 'Warning: no person detected'
    elif person_count > 1:
        person_warning = f"Warning: {person_count} persons detected"
    
    return phone_warning, person_warning

def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return new_frame_time
