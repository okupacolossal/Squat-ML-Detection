#IMPORTS
import cv2
import mediapipe as mp
import math
import os
import numpy as np

#WHILE LOOP VAR
running = True

#VIDEOS
listpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VideoExamples')
videolist = os.listdir(listpath)
videos = []
playingindex = 0

#REPS
reps = {}
reps_number = 0
squat_stage = 'still'
last_knee_angles = []
rep_completed = False
hit_bottom = False

#JOIN VIDEOS IN THE LIST
for i in videolist:
    fullvideo = os.path.join(listpath, i)
    videos.append(fullvideo)

#CAPTURE VIDEOS
cap = cv2.VideoCapture(videos[playingindex])
mp_pose = mp.solutions.pose

#POSE DETECTION
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#CREATE WINDOW
cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)


#FUNCTIONS
def drawVisibility(frame, landmark, result, w, h, name):
    lm = result[landmark]
    x, y = int(lm.x * w), int(lm.y * h)
    cv2.circle(frame, (x, y), 5, (0, 0, 0), -1, lineType=cv2.LINE_AA)

def getLandmarks():
    return {
            'LFOOT': mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
            'RFOOT': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
            'LHEEL': mp_pose.PoseLandmark.LEFT_HEEL.value,
            'RHEEL': mp_pose.PoseLandmark.RIGHT_HEEL.value,
            'LANKLE': mp_pose.PoseLandmark.LEFT_ANKLE.value,
            'RANKLE': mp_pose.PoseLandmark.RIGHT_ANKLE.value,

            # Knees
            'LKNEE': mp_pose.PoseLandmark.LEFT_KNEE.value,
            'RKNEE': mp_pose.PoseLandmark.RIGHT_KNEE.value,

            # Hips / pelvis
            'LHIP': mp_pose.PoseLandmark.LEFT_HIP.value,
            'RHIP': mp_pose.PoseLandmark.RIGHT_HIP.value,

            # Shoulders
            'LSHOULDER': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            'RSHOULDER': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,

            # Head / alignment
            'LEAR': mp_pose.PoseLandmark.LEFT_EAR.value,
            'REAR': mp_pose.PoseLandmark.RIGHT_EAR.value,
            'NOSE': mp_pose.PoseLandmark.NOSE.value,

            # Wrists (optional)
            'LWRIST': mp_pose.PoseLandmark.LEFT_WRIST.value,
            'RWRIST': mp_pose.PoseLandmark.RIGHT_WRIST.value
    }

def drawSkeleton(frame, lm2d, w, h):
    # Define connections (pairs of landmarks)
    connections = [
        # Torso
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),

        # Arms
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),

        # Legs
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),

        # Feet
        (mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL),
        (mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.LEFT_FOOT_INDEX),
        (mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL),
        (mp_pose.PoseLandmark.RIGHT_HEEL, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX),

        # Head
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_EAR),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_EAR),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_EAR),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_EAR)
    ]

    for start, end in connections:
        p1 = lm2d[start.value]
        p2 = lm2d[end.value]
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)

        # Draw smooth anti-aliased line
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3, lineType=cv2.LINE_AA)

def calculate3PAngles(mainpoint, point1, point2, lm):
    
    mainpoint, point1, point2 = lm[mainpoint], lm[point1], lm[point2]
    dir1 = (
        point1.x - mainpoint.x,
        point1.y - mainpoint.y,
        point1.z - mainpoint.z
    )
    dir2 = (
        point2.x - mainpoint.x,
        point2.y - mainpoint.y,
        point2.z - mainpoint.z
    )

    dotproduct = dir1[0] * dir2[0] + dir1[1] * dir2[1] + dir1[2] * dir2[2]

    crossproduct = np.cross(dir1, dir2)
    magnitudedir1 = math.sqrt(dir1[0] ** 2 + dir1[1] ** 2 + dir1[2] ** 2)
    magnitudedir2 = math.sqrt(dir2[0] ** 2 + dir2[1] ** 2 + dir2[2] ** 2)

    angle = math.acos(dotproduct / (magnitudedir1 * magnitudedir2))
    angleInDegrees = angle * (180 / math.pi)
    flexion = 180 - angleInDegrees

    return flexion

def drawAssignAngles(point, angle, frame, marks):
    x, y  = int(marks[point].x * w), int(marks[point].y * h)
    cv2.putText(frame, f'{int(angle)}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f'{int(angle)}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

def torsoLogic(neededLandmarks):
    torso_vector_x = lm[neededLandmarks['RSHOULDER']].x - lm[neededLandmarks['RHIP']].x
    torso_vector_y = lm[neededLandmarks['RSHOULDER']].y - lm[neededLandmarks['RHIP']].y
    torso_vector_z = lm[neededLandmarks['RSHOULDER']].z - lm[neededLandmarks['RHIP']].z
    torsov = np.array([torso_vector_x, torso_vector_y, torso_vector_z])
    torsomagnitude = np.linalg.norm(torsov)
    torso_normalize = torsov / torsomagnitude
    torsoup_vector = np.array([0, -1, 0])
    torsoalignment = np.dot(torso_normalize, torsoup_vector)    
    torsoangle = int((math.acos(torsoalignment) * (180 / math.pi)))
    torsox = (((lm2d[neededLandmarks['RSHOULDER']].x + (lm2d[neededLandmarks['LSHOULDER']].x)) / 2) * w)
    torsoy = (((lm2d[neededLandmarks['RSHOULDER']].y + (lm2d[neededLandmarks['LSHOULDER']].y)) / 2) * h)
    return torsoangle, (torsox, torsoy)

def calcAngles():
    CalculatePoints = {
        'L_KNEE': (neededLandmarks['LKNEE'], neededLandmarks['LHIP'], neededLandmarks['LANKLE']),
        'R_KNEE': (neededLandmarks['RKNEE'], neededLandmarks['RHIP'], neededLandmarks['RANKLE']),
        'L_DORSIFLEX': (neededLandmarks['LANKLE'], neededLandmarks['LFOOT'], neededLandmarks['LKNEE']),
        'R_DORSIFLEX': (neededLandmarks['RANKLE'], neededLandmarks['RFOOT'], neededLandmarks['RKNEE']),
        'L_HIP': (neededLandmarks['LHIP'], neededLandmarks['LSHOULDER'], neededLandmarks['LKNEE']),
        'R_HIP': (neededLandmarks['RHIP'], neededLandmarks['RSHOULDER'], neededLandmarks['RKNEE']),
    }
    Angles = {}
    for key, point in CalculatePoints.items():
        angle = int(calculate3PAngles((point[0]), point[1], point[2], lm))
        Angles[key] = angle, point[0]

        if key == 'L_DORSIFLEX':
            Angles[key] = abs(angle - 90), point[0]
        if key == 'R_DORSIFLEX':
            Angles[key] = abs(angle - 90), point[0]
        
    #Angles['TORSO'] = torsoLogic(lm)
    
    justAngles = {}
    for key, (angle, point) in Angles.items():
        drawAssignAngles(point, angle, frame, lm2d)
        justAngles[key] = angle
    return justAngles

def checkState(angles):
    
    global hit_bottom, rep_counted, reps_number
    
    # Initialize storage for current rep if needed
    if reps_number not in reps:
        reps[reps_number] = {}
    
    # Store angle data
    for name, angle in angles.items():
        if name not in reps[reps_number]:
            reps[reps_number][name] = []
        reps[reps_number][name].append(angle)
    
    # Determine current movement stage
    ss = None
    if last_knee_angles:
        if angles['L_KNEE'] > max(last_knee_angles) + 3:
            # Knee angle increasing = descending into squat
            ss = 'descending'
            # Reset flags when starting a new descent (only if previous rep was fully completed)
            if hit_bottom and rep_counted:
                hit_bottom = False
                rep_counted = False
                print(f'Starting new rep {reps_number}')
        elif angles['L_KNEE'] < min(last_knee_angles) - 3:
            # Knee angle decreasing = ascending from squat
            if not hit_bottom:
                ss = 'ascending'
                hit_bottom = True
                print(f'Rep {reps_number}: Hit the bottom')
            else:
                ss = 'ascending'
        elif max(last_knee_angles) - min(last_knee_angles) <= 6:
            ss = 'still'
    
    # Update knee angle history
    last_knee_angles.append(angles['L_KNEE'])
    if len(last_knee_angles) > 10:
        last_knee_angles.pop(0)
    
    # Count rep when returning near starting position
    if (hit_bottom and 
        not rep_counted and 
        ss == 'ascending' and
        len(reps[reps_number]['L_KNEE']) > 20):
        
        # Get starting angle (average of first 15 frames)
        starting_angles = reps[reps_number]['L_KNEE'][:15]
        avg_starting_angle = sum(starting_angles) / len(starting_angles)
        
        # Check if we're back near starting position
        if angles['L_KNEE'] >= avg_starting_angle - 15:
            print(f'✓ REP {reps_number + 1} COMPLETED!')
            reps_number += 1
            rep_counted = True

def drawHUD():
    global running, playingindex, cap  # allow modifying these g

    # Display current video and squat stage
    cv2.putText(frame, f'Video: {playingindex + 1}/{len(videos)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, squat_stage, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, str(reps_number), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Pose', frame)

    # Handle key presses
    key = cv2.waitKey(20) & 0xFF
    if key == ord('q'):
        running = False  # exit main loop
    elif key == ord('d'):
        playingindex = (playingindex + 1) % len(videos)
        cap.release()
        cap = cv2.VideoCapture(videos[playingindex])
    elif key == ord('a'):
        playingindex = (playingindex - 1) % len(videos)
        cap.release()
        cap = cv2.VideoCapture(videos[playingindex])

# ---------------------------
# Main loop
# ---------------------------
last_bodyresult = None

while cap.isOpened() and running:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (900, 600))

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h, w, _ = frame.shape
    bodyresult = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if bodyresult.pose_world_landmarks:
        last_bodyresult = bodyresult

    if last_bodyresult and last_bodyresult.pose_world_landmarks:
        lm = last_bodyresult.pose_world_landmarks.landmark
        lm2d = last_bodyresult.pose_landmarks.landmark

        # Draw skeleton lines
        drawSkeleton(frame, lm2d, w, h)

        # Draw visible landmark dots
        neededLandmarks = getLandmarks()

        for i, landmark in neededLandmarks.items():
            drawVisibility(frame, landmark, last_bodyresult.pose_landmarks.landmark, w, h, i) 
        
        angles = calcAngles()
        passedStage = checkState(angles)


        
    

    drawHUD()
# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()
pose.close()
