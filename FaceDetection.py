import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0, # 0 for speed, 1 for better accuracy
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
connections = mp_hands.HAND_CONNECTIONS

w, h, fps = 640, 480, 60

running = True
BORDER_COLOR = [5, 158, 0] 
BORDER_WIDTH = 5 
win_name = 'EXERCIFY'

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS, fps)

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def draw_hand_landmarks(hand_landmarks, frame):
    for con in connections:
        idx1 = con[0]
        idx2 = con[1]

        p1 = (int(hand_landmarks.landmark[idx1].x * w), int(hand_landmarks.landmark[idx1].y * h))
        p2 = (int(hand_landmarks.landmark[idx2].x * w), int(hand_landmarks.landmark[idx2].y * h))

        cv2.line(frame, p1, p2, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, p1, 5, (0,0,0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(frame, p2, 5, (0,0,0), thickness=2, lineType=cv2.LINE_AA)

def check_gesture(hand_landmarks, frame):
    p1 = (int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h))
    if hand_landmarks.landmark[11].y > hand_landmarks.landmark[12].y:
        cv2.putText(frame, 'Middle finger', p1, cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)



while cap.isOpened() and running:
    ret, frame = cap.read() 
    if not ret:
        continue
    
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            draw_hand_landmarks(hand_landmarks, frame)
            check_gesture(hand_landmarks, frame)

    custom_frame = cv2.copyMakeBorder(
        frame, 
        BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, 
        cv2.BORDER_CONSTANT, 
        value=BORDER_COLOR
    )
    
    cv2.imshow(win_name, custom_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False 

cap.release()
cv2.destroyAllWindows()