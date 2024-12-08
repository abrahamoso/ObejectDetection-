import cv2
import torch
import serial
import time
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
from threading import Thread
import queue

class VideoStreamThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.q = queue.Queue(maxsize=2)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    break
                while self.q.full():
                    self.q.get()
                self.q.put(frame)
            else:
                time.sleep(0.001)
    
    def read(self):
        return self.q.get() if not self.q.empty() else None
    
    def stop(self):
        self.stopped = True
        self.cap.release()

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Reduced for better performance
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_gesture(self, hand_landmarks):
        # Get important finger landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        wrist = hand_landmarks.landmark[0]

        # Get finger bases for better relative measurements
        index_base = hand_landmarks.landmark[5]
        middle_base = hand_landmarks.landmark[9]
        ring_base = hand_landmarks.landmark[13]
        pinky_base = hand_landmarks.landmark[17]

        # Define threshold for finger raised/lowered
        RAISED_THRESHOLD = 0.1  # Adjusted threshold
        
        # Helper function to check if finger is raised
        def is_finger_raised(tip, base):
            return (base.y - tip.y) > RAISED_THRESHOLD

        # Check hand position
        if wrist.y > index_tip.y:  # Hand is raised
            # LEFT gesture (thumb pointing left)
            if (thumb_tip.x < wrist.x - 0.1 and 
                thumb_tip.y < wrist.y and 
                not is_finger_raised(index_tip, index_base)):
                return "LEFT"
            
            # RIGHT gesture (thumb pointing right)
            elif (thumb_tip.x > wrist.x + 0.1 and 
                  thumb_tip.y < wrist.y and 
                  not is_finger_raised(index_tip, index_base)):
                return "RIGHT"
        
        # STOP gesture (index and pinky up, others down)
        if (is_finger_raised(index_tip, index_base) and 
            not is_finger_raised(middle_tip, middle_base) and 
            not is_finger_raised(ring_tip, ring_base) and 
            is_finger_raised(pinky_tip, pinky_base)):
            return "STOP"
        
        # UP gesture (index and middle up, others down)
        elif (is_finger_raised(index_tip, index_base) and 
              is_finger_raised(middle_tip, middle_base) and 
              not is_finger_raised(ring_tip, ring_base) and 
              not is_finger_raised(pinky_tip, pinky_base)):
            return "UP"
        
        # DOWN gesture (only index up)
        elif (is_finger_raised(index_tip, index_base) and 
              not is_finger_raised(middle_tip, middle_base) and 
              not is_finger_raised(ring_tip, ring_base) and 
              not is_finger_raised(pinky_tip, pinky_base)):
            return "DOWN"

        return None

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        gesture = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2),
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                gesture = self.detect_gesture(hand_landmarks)
                if gesture:
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
        return frame, gesture

    def close(self):
        self.hands.close()


class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Custom connection style
        self.connection_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),  # Green color
            thickness=2
        )
        
        # Custom landmark style
        self.landmark_spec = self.mp_draw.DrawingSpec(
            color=(255, 255, 255),  # White color
            thickness=2,
            circle_radius=2
        )

    def detect_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Create custom connections list (including head)
            connections = [
                # Head to shoulders
                (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.LEFT_SHOULDER.value),
                (self.mp_pose.PoseLandmark.NOSE.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                
                # Torso
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value),
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_HIP.value),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
                (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_HIP.value),
                
                # Arms
                (self.mp_pose.PoseLandmark.LEFT_SHOULDER.value, self.mp_pose.PoseLandmark.LEFT_ELBOW.value),
                (self.mp_pose.PoseLandmark.LEFT_ELBOW.value, self.mp_pose.PoseLandmark.LEFT_WRIST.value),
                (self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value, self.mp_pose.PoseLandmark.RIGHT_ELBOW.value),
                (self.mp_pose.PoseLandmark.RIGHT_ELBOW.value, self.mp_pose.PoseLandmark.RIGHT_WRIST.value),
                
                # Legs
                (self.mp_pose.PoseLandmark.LEFT_HIP.value, self.mp_pose.PoseLandmark.LEFT_KNEE.value),
                (self.mp_pose.PoseLandmark.LEFT_KNEE.value, self.mp_pose.PoseLandmark.LEFT_ANKLE.value),
                (self.mp_pose.PoseLandmark.RIGHT_HIP.value, self.mp_pose.PoseLandmark.RIGHT_KNEE.value),
                (self.mp_pose.PoseLandmark.RIGHT_KNEE.value, self.mp_pose.PoseLandmark.RIGHT_ANKLE.value),
            ]
            
            # Draw connections
            for connection in connections:
                start_point = (int(landmarks[connection[0]].x * frame.shape[1]),
                             int(landmarks[connection[0]].y * frame.shape[0]))
                end_point = (int(landmarks[connection[1]].x * frame.shape[1]),
                           int(landmarks[connection[1]].y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, self.connection_spec.color, self.connection_spec.thickness)
            
            # Draw landmarks for body and head
            body_landmarks = [
                self.mp_pose.PoseLandmark.NOSE.value,  # Head point
                self.mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                self.mp_pose.PoseLandmark.LEFT_ELBOW.value,
                self.mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                self.mp_pose.PoseLandmark.LEFT_WRIST.value,
                self.mp_pose.PoseLandmark.RIGHT_WRIST.value,
                self.mp_pose.PoseLandmark.LEFT_HIP.value,
                self.mp_pose.PoseLandmark.RIGHT_HIP.value,
                self.mp_pose.PoseLandmark.LEFT_KNEE.value,
                self.mp_pose.PoseLandmark.RIGHT_KNEE.value,
                self.mp_pose.PoseLandmark.LEFT_ANKLE.value,
                self.mp_pose.PoseLandmark.RIGHT_ANKLE.value,
            ]
            
            for landmark_id in body_landmarks:
                landmark = landmarks[landmark_id]
                point = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                cv2.circle(frame, point, self.landmark_spec.circle_radius, 
                          self.landmark_spec.color, self.landmark_spec.thickness)

        return frame

    def close(self):
        self.pose.close()

def main():
    # Initialize settings
    CAMERA_INDEX = 1
    
    # Initialize video stream
    print("Starting video stream...")
    vs = VideoStreamThread(CAMERA_INDEX).start()
    time.sleep(1.0)

def main():
    CAMERA_INDEX = 1
    PROCESS_EVERY_N_FRAMES = 2  # Only process every 2nd frame
    
    print("Starting video stream...")
    vs = VideoStreamThread(CAMERA_INDEX).start()
    time.sleep(1.0)

    pose_detector = PoseDetector()
    hand_detector = HandGestureDetector()

    print("Loading YOLO model...")
    try:
        model = YOLO('yolov5n.pt')  # Using the smallest YOLO model
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"YOLOv5 model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    frame_times = []
    start_time = time.time()
    frames_processed = 0
    last_gesture = None
    gesture_duration = 0
    last_person_box = None  # Cache the last detected person location

    try:
        while True:
            loop_start = time.time()
            
            frame = vs.read()
            if frame is None:
                continue

            try:
                # Only run YOLO and pose detection every N frames
                if frames_processed % PROCESS_EVERY_N_FRAMES == 0:
                    results = model(frame, stream=True)
                    person_detected = False
                    
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            if int(box.cls.cpu().numpy()[0]) == 0:  # Person detected
                                last_person_box = box.xyxy[0].cpu().numpy()  # Cache the box
                                frame = pose_detector.detect_pose(frame)
                                person_detected = True
                                break
                        if person_detected:
                            break
                
                # Always process hand gestures (they're faster)
                frame, gesture = hand_detector.process_frame(frame)
                
                # Draw the last known person location
                if last_person_box is not None:
                    x1, y1, x2, y2 = map(int, last_person_box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if gesture:
                    if gesture != last_gesture:
                        print(f"Detected gesture: {gesture}")
                        last_gesture = gesture
                        gesture_duration = time.time()
                elif last_gesture and time.time() - gesture_duration > 2:
                    last_gesture = None

                # Calculate and display FPS less frequently
                frames_processed += 1
                if frames_processed % 30 == 0:
                    fps = frames_processed / (time.time() - start_time)
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Detection with Gestures", frame)

            except Exception as e:
                print(f"Error during detection: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        vs.stop()
        cv2.destroyAllWindows()
        pose_detector.close()
        hand_detector.close()

if __name__ == "__main__":
    main()