from typing import cast
import cv2
import mediapipe as mp
from gestures import Gestures
from utils import draw_landmarks
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
from mediapipe.tasks.python.components.processors import ClassifierOptions

# mediapipe config
# ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_hand_detection_confidence=0.7,
    num_hands=1)
    # canned_gesture_classifier_options=ClassifierOptions(category_allowlist=["None", "Closed_Fist", "Pointing_Up"]))

def main():
    with GestureRecognizer.create_from_options(options) as recognizer:
        camera = cv2.VideoCapture(0)
        gestures = Gestures(True)
        
        try:
            while camera.isOpened():
                retrieved, frame = camera.read()
                if not retrieved:
                    break
                
                timestamp = round(camera.get(cv2.CAP_PROP_POS_MSEC))
                frame = cv2.flip(frame, 1)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                result = cast(GestureRecognizerResult, recognizer.recognize_for_video(mp_image, timestamp))

                annotated_image = draw_landmarks(mp_image.numpy_view(), result)
                gestures.detect(annotated_image, result)

                cv2.imshow("Camera Mouse Debug", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                if (cv2.waitKey(1) & 0xFF == ord('q')):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()
