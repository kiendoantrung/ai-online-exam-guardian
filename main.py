import collections
import threading
import time
import cv2
import numpy as np
import imutils  # type: ignore

from src.pose_fuctions import *
from src.object_detection import *
from src.face_functions import *
from src.send_email import *
from dotenv import load_dotenv
from playsound import playsound
import mediapipe as mp  # Assuming mediapipe is used

# Initialize mediapipe modules
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

load_dotenv()

# Initialize video captures
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap2 = cv2.VideoCapture(1)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize audio playing flag
audio_playing = False
audio_lock = threading.Lock()

# Create a shared event to signal threads to stop
stop_event = threading.Event()

def play_audio_alert():
    global audio_playing
    with audio_lock:
        if not audio_playing:
            audio_playing = True
            try:
                playsound('audio.mp3')
            except Exception as e:
                print(f"Audio playback error: {e}")
            audio_playing = False

def face_system():
    global audio_playing
    prev_frame_time = 0
    output = None
    codec = cv2.VideoWriter_fourcc(*'XVID')
    frame_rec = 0
    recording = False
    new_dir = True
    eyes_movements, head_movements, mouth_movements, hand_movements = [], [], [], []
    warnings = [""]
    warning_count = 0
    collection_warnings = []
    
    try:
        with mp_hands.Hands(model_complexity=0) as hands:
            with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
                while cap.isOpened() and not stop_event.is_set():
                    success, frame = cap.read()
                    if not success:
                        print('Ignore empty camera frame!')
                        break

                    image = cv2.flip(frame, 1)
                    image, results = mediapipe_detection(image, face_mesh)
                    image, hand_results = mediapipe_detection(image, hands)
                    
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            lmks = face_landmarks.landmark

                            # Extract relevant landmarks for eyes, head, and mouth
                            eyes_landmarks = [lmks[263], lmks[473], lmks[362], lmks[133], lmks[468], lmks[33]]
                            eyes_movement = get_eyes_movement(eyes_landmarks)
                            eyes_movements.append(eyes_movement)
                            eyes_movements = eyes_movements[-21:]

                            face_keypoints = [lmks[1], lmks[33], lmks[263], lmks[61], lmks[291], lmks[199]]
                            head_movement = get_head_movement(image, face_keypoints)
                            head_movements.append(head_movement)
                            head_movements = head_movements[-21:]

                            mouth_movement = get_mouth_movement(lmks[13], lmks[14])
                            mouth_movements.append(mouth_movement)
                            mouth_movements = mouth_movements[-21:]

                            if len(eyes_movements) == 21:
                                warn_eyes = warning_3s(eyes_movements)
                                warn_head = warning_3s(head_movements)
                                warn_mouth = warning_3s(mouth_movements)
                                warnings.extend([warn_eyes, warn_head, warn_mouth])

                            draw_face_landmarks(image, face_landmarks)
                            face_occlusion_points = [lmks[13], lmks[8]]

                            if hand_results.multi_hand_landmarks:
                                for hand_landmarks in hand_results.multi_hand_landmarks:
                                    hand_lmks = hand_landmarks.landmark
                                    hand_limit = get_limit_hand_coordinate(hand_lmks)
                                    hand_movement = get_hand_movement(face_occlusion_points, hand_limit)
                                    hand_movements.append(hand_movement)
                                    hand_movements = hand_movements[-21:]

                                    if len(hand_movements) == 21:
                                        warn_hand = warning_3s(hand_movements)
                                        warnings.append(warn_hand)
                                    draw_hand_landmarks(image, hand_landmarks)
                            else:
                                hand_movements.append("")
                                hand_movements = hand_movements[-21:]

                    detected_objects = detect_objects(image)
                    
                    warning_info, warnings = warning_display(warnings)
                    phone_warning, person_warning = warning(detected_objects)
                    
                    if person_warning:
                        cv2.putText(image, person_warning, (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (45, 255, 255), 2,
                                    cv2.LINE_AA)     
                    if phone_warning:               
                        cv2.putText(image, phone_warning, (7, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (45, 255, 255), 2,
                                    cv2.LINE_AA)
                    if warning_info:
                        cv2.putText(image, warning_info, (7, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (45, 255, 255), 2,
                                    cv2.LINE_AA)
                    
                    current_warnings = [phone_warning, person_warning, warning_info]
                    for warning_msg in current_warnings:
                        if warning_msg:
                            collection_warnings.append(warning_msg)
                            recording = True
                            warning_count += 1
                            if warning_count % 10 == 0:
                                email_thread = threading.Thread(target=send_warning_email, args=(22006, collection_warnings.copy()))
                                email_thread.start()
                                collection_warnings.clear()
                                audio_thread = threading.Thread(target=play_audio_alert)
                                audio_thread.start()
                            if new_dir:
                                vidOutDir = f"data/external/video_log/{time.strftime('%Y%m%d-%H%M%S')}.avi"
                                output = cv2.VideoWriter(vidOutDir, codec, 15, (1280, 720))
                                new_dir = False
                    
                    prev_frame_time = show_fps(image, prev_frame_time)
                        
                    if recording:
                        if frame_rec <= 600:
                            output.write(image)
                            frame_rec += 1
                        else:
                            frame_rec = 0
                            recording = False
                            output.release()
                            new_dir = True

                    for obj in detected_objects:
                        x1, y1, x2, y2, conf, cls = obj
                        class_name = model.names[cls]
                        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(image, f"{class_name}: {conf:.2f}", (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cv2.imshow("Front camera", image)
                    
                    # Check for 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
    except Exception as e:
        print(f"Error in face_system: {e}")
    finally:
        cap.release()
        cv2.destroyWindow("Front camera")
        print("Face system terminated.")

def pose_system():
    prev_frame_time = 0
    input_sequence = []
    predictions = []

    frame_rec = 0
    output = None
    codec = cv2.VideoWriter_fourcc(*'XVID')

    recording = False
    new_dir = True

    try:
        with mp_pose.Pose() as pose:
            while cap2.isOpened() and not stop_event.is_set():
                # Read feed
                ret, frame = cap2.read()
                if not ret:
                    print("Can't get frame!")
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, pose)

                # Draw landmarks
                if results.pose_landmarks:
                    draw_landmarks(image, results)

                frame_landmarks = get_frame_landmarks(results)
                input_sequence.append(frame_landmarks)
                input_sequence = input_sequence[-30:]
                if len(input_sequence) == 30:
                    # Use ONNX runtime for inference
                    input_data = np.expand_dims(input_sequence, axis=0).astype(np.float32)
                    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
                    ort_outs = ort_session.run(None, ort_inputs)
                    res = ort_outs[0][0]
                    cheating_prob = round(res[1], 2)

                    if cheating_prob > 0.8:
                        predictions.append(1)
                        cv2.putText(image, "Cheating probs: " + str(cheating_prob), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        predictions.append(0)
                        cv2.putText(image, "Cheating probs: " + str(cheating_prob), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                    (255, 0, 0), 2, cv2.LINE_AA)

                    predictions = predictions[-20:]
                    prediction_dict = dict(collections.Counter(predictions))
                    if 1 in prediction_dict and prediction_dict[1] > 15:
                        cv2.putText(image, "Warning: suspicious behavior", (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (45, 255, 255),
                                    2, cv2.LINE_AA)
                        recording = True
                        if new_dir:
                            vidOutDir = f"data/external/video_log/{time.strftime('%Y%m%d-%H%M%S')}-suspicious_action.avi"
                            output = cv2.VideoWriter(vidOutDir, codec, 12, (1280, 720))
                            new_dir = False

                # Show fps
                prev_frame_time = show_fps(image, prev_frame_time)

                # Handle recording
                if recording:
                    if frame_rec <= 600:
                        output.write(image)
                        frame_rec += 1
                    else:
                        frame_rec = 0
                        recording = False
                        new_dir = True
                        output.release()

                # Show to screen
                cv2.imshow('OpenCV Feed', imutils.resize(image, width=1000))

                # Check for 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break
    except Exception as e:
        print(f"Error in pose_system: {e}")
    finally:
        cap2.release()
        cv2.destroyWindow("OpenCV Feed")
        print("Pose system terminated.")

# Create and start threads for face and pose systems
t1 = threading.Thread(target=face_system)
t2 = threading.Thread(target=pose_system)
t1.start()
t2.start()

# Wait for both threads to finish
t1.join()
t2.join()

# Ensure all OpenCV windows are closed
cv2.destroyAllWindows()
print("All systems terminated gracefully.")
