import cv2
import mediapipe as mp
import math

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Process hands and pose
    handsresult = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bodyresult = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Draw hand bounding box
    if handsresult.multi_hand_landmarks:
        hand = handsresult.multi_hand_landmarks[0]
        wristpoints = [
            hand.landmark[mp_hands.HandLandmark.THUMB_CMC],
            hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP],
            hand.landmark[mp_hands.HandLandmark.PINKY_MCP],
            hand.landmark[mp_hands.HandLandmark.WRIST],
            hand.landmark[mp_hands.HandLandmark.RING_FINGER_MCP],
            hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        ]
        x_vals = [p.x for p in wristpoints]
        y_vals = [p.y for p in wristpoints]
        min_x, max_x = int(min(x_vals) * w), int(max(x_vals) * w)
        min_y, max_y = int(min(y_vals) * h), int(max(y_vals) * h)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Draw pose landmarks and visualize elbow angle
    if bodyresult.pose_landmarks:
        landmarks = bodyresult.pose_landmarks.landmark

        lms = {
            'left_shoulder': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            'left_elbow': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            'left_wrist': landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
            'right_shoulder': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            'right_elbow': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            'right_wrist': landmarks[mp_pose.PoseLandmark.RIGHT_WRIST],
        }

        # Draw all landmarks as green circles
        for lm in lms.values():
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1) 

        # Check if the left arm is visible
        if lms['left_elbow'].visibility > 0.2:
            # Get the direction of upper arm/lower arm vectors
            upper_arm = (lms['left_elbow'].x - lms['left_shoulder'].x) * w, (lms['left_elbow'].y - lms['left_shoulder'].y) * h
            lower_arm = (lms['left_elbow'].x - lms['left_wrist'].x) * w, (lms['left_elbow'].y - lms['left_wrist'].y) * h
            
            diffx = lower_arm[0] - upper_arm[0]
            diffy = lower_arm[1] - upper_arm[1]
            angle_radians = math.atan2(diffy, diffx)
            angle_degrees = math.degrees(angle_radians)
            print(f'Left elbow angle: {angle_degrees:.2f} degrees')
    # Show frame
    cv2.imshow('Frame', frame)

cap.release()
cv2.destroyAllWindows()
