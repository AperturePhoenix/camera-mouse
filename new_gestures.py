from typing import List, cast
from mediapipe.tasks.python.components.containers.category import Category

import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
import numpy as np
import pyautogui

screen_width, screen_height = pyautogui.size()


def dist(landmarks: List[NormalizedLandmark], i: int, j: int) -> float:
    a = landmarks[i]
    b = landmarks[j]
    return np.hypot(a.x - b.x, a.y - b.y)


def is_finger_bent(landmarks: List[NormalizedLandmark], PIP_INDEX: int) -> bool:
    return landmarks[PIP_INDEX].y > landmarks[PIP_INDEX + 1]


class Gestures():
    debug: bool
    curr_pos: NormalizedLandmark = None
    prev_pos: NormalizedLandmark = None
    frame: np.ndarray

    def __init__(self, debug = True) -> None:
        self.debug = debug

    def detect(self, frame: np.ndarray, result: GestureRecognizerResult):
        if not result.hand_landmarks:
            self.curr_pos = None
            self.prev_pos = None
            self.frame = None
            return
        

        self.frame = frame
        self.curr_pos = result.hand_landmarks[0][4]
        self.scroll(result)

        self.prev_pos = self.curr_pos

    def scroll(self, result: GestureRecognizerResult):
        if not self.prev_pos:
            return

        if result.gestures:
            gesture = cast(Category, result.gestures[0][0])
            if not gesture.category_name == "Closed_Fist" or gesture.score < 0.8:
                return
            
            amount = round((self.curr_pos.x - self.prev_pos.x) * screen_width)
            if abs(amount) < 2:
                amount = 0
            pyautogui.scroll(amount * 10)

            if self.debug:
                cv2.putText(self.frame, f"Scoll: {amount}", (15, 25), cv2.FONT_HERSHEY_DUPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)
