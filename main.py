import cv2
import mediapipe as mp
import screen_brightness_control as sbc

mp_hands = mp.solutions.hands

FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]


def is_hand_open(landmarks):
    fingers_up = 0
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip].y < landmarks[pip].y:
            fingers_up += 1
    if landmarks[4].x < landmarks[3].x:
        fingers_up += 1
    return fingers_up >= 4


cap = cv2.VideoCapture(0)
current_brightness = -1

print("Running. Press Ctrl+C to stop.")

try:
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0].landmark
                target = 100 if is_hand_open(lm) else 0
                if target != current_brightness:
                    sbc.set_brightness(target)
                    current_brightness = target
                    print(f"Brightness -> {target}%")
finally:
    cap.release()
