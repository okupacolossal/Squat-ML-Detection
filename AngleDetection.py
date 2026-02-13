#IMPORTS
import cv2
import mediapipe as mp
import math
import os
import numpy as np
import time
from scipy.signal import savgol_filter
import pandas as pd
from collections import deque
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#WHILE LOOP VAR
running = True

def train_grading_model(csv_path='data.csv'):
    """
    Train a linear regression model on your rep dataset
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Features (X) and target (y)
    features = ['start_angle', 'bottom_angle', 'end_angle', 'descent_time', 'ascent_time', 'depth', 'velocity_smoothness']
    X = df[features]
    y = df['rank']
    
    # Train the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the model
    with open('rep_grading_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved!")
    print(f"Model R² score: {model.score(X, y):.4f}")
    
    return model

# Train once at startup
grading_model = train_grading_model()

class Session: 
    def __init__(self):
        self.reps = {}
        self.reps_number = 0
        self.reps_angles = {}

        self.list_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VideoExamples')
        
        # Check if directory exists
        if not os.path.exists(self.list_path):
            print(f"Error: Directory '{self.list_path}' not found!")
            self.videos = []
            self.video_list = []
        else:
            self.video_list = os.listdir(self.list_path)
            self.videos = [os.path.join(self.list_path, video) for video in self.video_list]
        
        self.index = 0

        if self.videos:
            self.cap = cv2.VideoCapture(self.videos[self.index])
            if not self.cap.isOpened():
                print(f"Error: Could not open video {self.videos[self.index]}")
        else:
            print("No videos found in VideoExamples directory")
            self.cap = None
        
        self.fps = 16
        self.running = True
        cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.last_bodyresult = None
        self.frame_count = 0

        self.current_rep = self.new_rep()
        
    def clean_up(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
    
    def next_video(self):
        """Switch to next video in the list"""
        if self.videos:
            self.index = (self.index + 1) % len(self.videos)
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.videos[self.index])
            print(f"Switched to video: {self.video_list[self.index]}")
            self.current_rep = self.new_rep(0)


    def new_rep(self, number=1):
        """Start a new rep"""
        print('Initiated rep number:', self.reps_number)
        rep = Rep(self.reps_number)
        self.reps[self.reps_number] = rep
        self.reps_number += number
        return rep
    
    def save_reps_to_csv(self):
        """Save all completed reps to CSV file"""
        data = []
        for rep_num, rep in self.reps.items():
            # Only save reps that have been completed (have end_time)
            if rep.end_time is not None:
                data.append({
                    'rep_number': rep.rep_number,
                    'start_angle': rep.start_angle,
                    'bottom_angle': rep.bottom_angle,
                    'end_angle': rep.end_angle,
                    'descent_time': rep.descent_time,
                    'ascent_time': rep.ascent_time,
                    'depth': rep.depth,
                    'velocity_smoothness': rep.velocity_smoothness
                })
        
        if data:
            df = pd.DataFrame(data)
            filename = f'reps_data_{time.time():.0f}.csv'
            df.to_csv(filename, index=False)
            print(f"✓ Saved {len(data)} reps to {filename}")
            return filename
        else:
            print("No completed reps to save")
            return None

class Rep:
    def __init__(self, rep_number):
        self.rep_number = rep_number
        self.angles = deque(maxlen=25)
        
        self.start_time = None
        self.start_angle = None
        self.bottom_time = None
        self.bottom_angle = None
        self.end_time = None
        self.end_angle = None

        self.state = None # states: still, descending, ascending
        self.still_timer = None

        self.ascent_time = None
        self.descent_time = None
        self.depth = None
        self.velocity_smoothness = None

# ---------------------------
# Helper Functions
# ---------------------------

def grade_rep(rep, model):

    features_dict = {
        'start_angle': [rep.start_angle],
        'bottom_angle': [rep.bottom_angle],
        'end_angle': [rep.end_angle],
        'descent_time': [rep.descent_time],
        'ascent_time': [rep.ascent_time],
        'depth': [rep.depth],
        'velocity_smoothness': [rep.velocity_smoothness]
    }

    x = pd.DataFrame(features_dict)
    grade = model.predict(x)[0]

    return grade
def calculate_angle(a, b, c):
    """
    Calculate angle between three points
    a, b, c are landmarks (each with x, y, z attributes)
    """
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def draw_landmarks(frame, landmarks, h, w, connections=None):
    """
    Draw landmarks and connections on frame
    """
    if not landmarks:
        return frame
    
    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_pos = (int(start.x * w), int(start.y * h))
            end_pos = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
    
    # Draw landmarks
    for landmark in landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
    
    return frame

def put_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale=0.7, text_color=(255, 255, 255), 
                            bg_color=(0, 0, 0), thickness=2, padding=5):
    """
    Draw text with a background rectangle for visibility
    """
    x, y = position
    
    # Get text size
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Create rectangle coordinates with padding
    rect_x1 = x - padding
    rect_y1 = y - text_size[1] - padding
    rect_x2 = x + text_size[0] + padding
    rect_y2 = y + padding
    
    # Draw background rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    return frame

def average(values: list, number: float = 2):
    return sum(values) / number

def frame_state_management(session, angle: int):
    """
    Manage frame state and rep counting logic
    """
    rep = session.current_rep
    if len(rep.angles) >= 3:

        # still logic
        if abs(max(rep.angles) - min(rep.angles)) <= 10:
            if not rep.still_timer:
                rep.still_timer = time.time()
            elif (time.time() - rep.still_timer) >= 0.2 and rep.state != 'still':
                rep.state = 'still'
                print('Still time started at', session.frame_count, 'while user is', rep.state)
        # movement logic
        else:
            rep.still_timer = None
            # started descending

            if rep.state == 'still':
                if angle < min(list(rep.angles)) - 6:
                    rep.state = 'descending'
                elif angle > max(list(rep.angles)) + 6:
                    rep.state = 'ascending'

            if not rep.start_time and angle < average(list(rep.angles)[-3:]):
                rep.start_time = time.time()
                rep.start_angle = angle
                rep.state = 'descending'
                print(f"Rep {rep.rep_number} started at frame {session.frame_count} with angle {angle}º")
            # ascending logic, set bottom
            elif len(rep.angles) > 10 and rep.state == 'descending' and angle > average(list(rep.angles)[-10:], 10) and angle <= 125:
                rep.state = 'ascending'
                rep.bottom_time = time.time()
                rep.bottom_angle = angle
                print(f"Rep {rep.rep_number} bottom reached at frame {session.frame_count}, {angle}º")
    
            if ( rep.state == 'still' and rep.bottom_time != None ) or (rep.state == 'ascending' and angle >= rep.start_angle - 5 and ( rep.state == 'still' or angle < min(rep.angles) - 3 )):
                rep.end_time = time.time()
                rep.end_angle = angle

                rep.ascent_time = round((rep.end_time - rep.bottom_time), 2)
                rep.descent_time = round((rep.bottom_time - rep.start_time), 2)
                rep.depth = int(180 - rep.bottom_angle)
                rep.velocity_smoothness = int(np.std(np.diff(rep.angles)))
                velocity_smoothness = rep.velocity_smoothness
                
                print(f'---- REP {rep.rep_number} completed at frame {session.frame_count} ----')
                print('Ascent time:', rep.ascent_time)
                print('Descent time:', rep.descent_time)
                print('Depth:', rep.depth)
                print('Velocity smoothness', velocity_smoothness)
                print('Grade:', grade_rep(rep, grading_model))
                print('---------------------------------------------')


                session.current_rep = session.new_rep()


    rep.angles.append(angle)


    pass
# ---------------------------
# Main loop
# ---------------------------

session = Session()

if not session.cap or not session.cap.isOpened():
    print("Could not initialize video capture. Exiting.")
else:
    while session.cap.isOpened() and session.running:
        ret, frame = session.cap.read()

        if not ret:
            print(f"Reached end of video. Looping...")
            session.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        h, w, _ = frame.shape
        session.frame_count += 1
        
        # Process pose
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bodyresult = session.pose.process(rgb_frame)
        
        if bodyresult.pose_world_landmarks:
            session.last_bodyresult = bodyresult

        # Draw on frame if we have results
        if session.last_bodyresult and session.last_bodyresult.pose_landmarks:
            lm = session.last_bodyresult.pose_world_landmarks.landmark
            lm2d = session.last_bodyresult.pose_landmarks.landmark
            
            # Draw pose skeleton
            frame = draw_landmarks(frame, lm2d, h, w, 
                                  mp.solutions.pose.POSE_CONNECTIONS)
            
            # Example: Calculate and display knee angles
            left_knee_angle = calculate_angle(lm[23], lm[25], lm[27])  # hip, knee, ankle
            right_knee_angle = calculate_angle(lm[24], lm[26], lm[28])

            frame_state_management(session, (round(((right_knee_angle + left_knee_angle)/2), 1)))
            
            frame = put_text_with_background(frame, f'L Knee: {left_knee_angle:.1f}°', (10, 30),
                                            text_color=(0, 255, 0), bg_color=(0, 0, 0))
            frame = put_text_with_background(frame, f'R Knee: {right_knee_angle:.1f}°', (10, 70),
                                            text_color=(0, 255, 0), bg_color=(0, 0, 0))
            
            # Display squat state
            state = session.current_rep.state
            state_color = (255, 255, 255)  # White for 'still'
            if state == 'descending':
                state_color = (0, 165, 255)  # Orange
            elif state == 'ascending':
                state_color = (0, 255, 0)  # Green
            
            frame = put_text_with_background(frame, f'State: {state.upper() if state else "NEUTRAL"}', (10, 110),
                                            font_scale=0.8, text_color=state_color, bg_color=(0, 0, 0), thickness=2)
            frame = put_text_with_background(frame, f'Rep: {session.current_rep.rep_number}', (10, 150),
                                            text_color=(255, 255, 0), bg_color=(0, 0, 0))
        
        # Display frame info
        frame = put_text_with_background(frame, f'Frame: {session.frame_count}', (10, h - 20),
                                        text_color=(255, 255, 0), bg_color=(0, 0, 0))
        frame = put_text_with_background(frame, f'Video: {session.video_list[session.index] if session.video_list else "None"}', 
                                        (10, h - 50), font_scale=0.5, text_color=(200, 200, 200), bg_color=(0, 0, 0))
        
        cv2.imshow('Pose', frame)
        
        # Handle key presses
        key = cv2.waitKey(session.fps) & 0xFF
        if key == ord('q'):
            session.running = False
            print("Quit requested")
        elif key == ord('n'):
            session.next_video()
            session.frame_count = 0
        elif key == ord('p'):
            print("Paused - press any key to continue")
            cv2.waitKey(0)
        elif key == ord('s'):
            filename = f'pose_screenshot_{time.time():.0f}.png'
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('b'):
            session.save_reps_to_csv()

# ---------------------------
# Cleanup
# ---------------------------
session.clean_up()
print("Session ended.")