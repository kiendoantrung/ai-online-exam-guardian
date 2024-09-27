import cv2
import numpy as np
import time
import mediapipe as mp
import onnxruntime as ort 
actions = np.array(['non_cheating', 'cheating'])
mp_pose = mp.solutions.pose

onnx_model_path = './weights/pose_weights.onnx' 
ort_session = ort.InferenceSession(onnx_model_path)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                  
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def get_scaled_landmarks(landmarks, dimension):
    if dimension == '2d':
        return [[int(landmark.x * 1280), int(landmark.y * 720)] for landmark in landmarks]
    elif dimension == 'both':
        return (
            [[int(landmark.x * 1280), int(landmark.y * 720)] for landmark in landmarks],
            [[int(landmark.x * 1280), int(landmark.y * 720), landmark.z] for landmark in landmarks]
        )
    return []

def draw_landmarks(image, results):
    lmks = results.pose_landmarks.landmark
    pose_landmarks = [lmks[0], lmks[11], lmks[12], lmks[13], lmks[14], lmks[15], lmks[16],
                      lmks[23], lmks[24], lmks[19], lmks[20]]
    pose_landmarks = get_scaled_landmarks(pose_landmarks, '2d')

    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[2]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[3]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[3]), tuple(pose_landmarks[5]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[2]), tuple(pose_landmarks[4]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[4]), tuple(pose_landmarks[6]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[1]), tuple(pose_landmarks[7]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[2]), tuple(pose_landmarks[8]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[7]), tuple(pose_landmarks[8]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[5]), tuple(pose_landmarks[9]), (255, 255, 255), 2)
    cv2.line(image, tuple(pose_landmarks[6]), tuple(pose_landmarks[10]), (255, 255, 255), 2)
    for lm in pose_landmarks:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 4, (0, 0, 255), -1)

def show_fps(image, prev_frame_time):
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time))
    cv2.putText(image, f"fps: {fps}", (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)
    return new_frame_time

def get_joint_angle(a, b, c):
    angle = np.abs(np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x))
    if angle > np.pi:
        angle = 2 * np.pi - angle
    return angle

def get_all_angles(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    right_elbow_angle = get_joint_angle(right_shoulder, right_elbow, right_wrist)
    right_shoulders_angle = get_joint_angle(right_elbow, right_shoulder, left_shoulder)
    left_elbow_angle = get_joint_angle(left_shoulder, left_elbow, left_wrist)
    left_shoulders_angle = get_joint_angle(left_elbow, left_shoulder, right_shoulder)
    nose_angle = get_joint_angle(left_shoulder, nose, right_shoulder)
    left_ear_angle = get_joint_angle(left_shoulder, left_ear, right_shoulder)
    right_ear_angle = get_joint_angle(left_shoulder, right_ear, right_shoulder)
    angles = [right_elbow_angle, right_shoulders_angle, left_elbow_angle, left_shoulders_angle,
              nose_angle, left_ear_angle, right_ear_angle]
    return angles

def get_frame_landmarks(results):
    if not results.pose_landmarks or not results.pose_world_landmarks:
        return np.zeros(191)  # 4*23 + 4*23 + 7

    size_landmarks = np.array([[res.x, res.y, res.z, res.visibility]
                               for res in results.pose_landmarks.landmark[:23]]).flatten()
    world_landmarks = np.array([[res.x, res.y, res.z, res.visibility]
                                for res in results.pose_world_landmarks.landmark[:23]]).flatten()
    angles = np.array(get_all_angles(results.pose_landmarks.landmark))

    return np.concatenate([size_landmarks, world_landmarks, angles])