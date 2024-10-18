import cv2
import mediapipe as mp
from gestures import Gestures


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def main():
    draw = mp.solutions.drawing_utils
    camera = cv2.VideoCapture(0)
    gestures = Gestures(mp_hands.HandLandmark, mp_hands.HandLandmark.INDEX_FINGER_TIP, True)

    try:
        while camera.isOpened():
            ret, frame = camera.read()

            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proccessed = hands.process(frame_rgb)

            landmarks = []
            if proccessed.multi_hand_landmarks:
                hand_landmarks = proccessed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y))
            
            gestures.detect_gesture(frame, landmarks, proccessed)

            cv2.imshow("Camera Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.getWindowProperty('Camera Mouse', cv2.WND_PROP_VISIBLE) < 1:        
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
  main()