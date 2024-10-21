from typing import List, cast
from mediapipe.tasks.python.components.containers.category import Category

import cv2
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
import numpy as np
import pyautogui

FONT_SIZE = 0.75
FONT_THICKNESS = 1
TEXT_COLOR = (88, 205, 54)
screen_width, screen_height = pyautogui.size()


def deadzone(value: int, deadzone: int = 4) -> int:
    if abs(value) < deadzone:
        return 0
    return value


def dist(landmarks: List[NormalizedLandmark], i: int, j: int) -> float:
    a = landmarks[i]
    b = landmarks[j]
    return np.hypot(a.x - b.x, a.y - b.y)


def is_touching(landmarks: List[NormalizedLandmark], i: int, j: int, max_dist: float = 7.5) -> bool:
    return np.interp(dist(landmarks, i, j), [0, 1], [0, 100]) <= max_dist


class Gestures():
    curr_pos: NormalizedLandmark = None
    prev_pos: NormalizedLandmark = None
    frame: np.ndarray

    active: bool = False
    has_toggle: bool = False
    has_left_click: bool = False
    has_right_click: bool = False

    debug: bool

    def __init__(self, debug = False) -> None:
        self.debug = debug

    def detect(self, frame: np.ndarray, result: GestureRecognizerResult):
        if not result.hand_landmarks:
            self.curr_pos = None
            self.prev_pos = None
            self.frame = None
            return

        self.frame = frame
        self.curr_pos = result.hand_landmarks[0][8]
        self.toggle_active(result)
        if self.active and self.prev_pos and result.gestures:
            self.scroll(result)
            self.move_mouse(result)
            self.left_click(result)
            self.right_click(result)

        self.prev_pos = self.curr_pos

    def toggle_active(self, result: GestureRecognizerResult):
        if is_touching(result.hand_landmarks[0], 4, 20, 3):
            if not self.has_toggle:
                self.active = not self.active
                self.has_toggle = True
            return True
        else:
            self.has_toggle = False

        if self.debug:
            if self.active:
                cv2.putText(self.frame, "Active", (15, 25), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            else:
                cv2.putText(self.frame, "Inactive", (15, 25), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, (255, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
    
    def move_mouse(self, result: GestureRecognizerResult):
        gesture = cast(Category, result.gestures[0][0])
        if not gesture.category_name == "Pointing_Up" or gesture.score < 0.8 or not is_touching(result.hand_landmarks[0], 4, 10, 3):
            return
        
        x = deadzone((self.curr_pos.x - self.prev_pos.x) * screen_width, 3)
        y = deadzone((self.curr_pos.y - self.prev_pos.y) * screen_width, 3)
        pyautogui.moveRel(x * 2, y * 4, _pause=False)
        
        if self.debug:
            cv2.putText(self.frame, f"Mouse: {x}", (15, 50), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            cv2.putText(self.frame, f"Mouse: {y}", (15, 75), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
    
    def left_click(self, result: GestureRecognizerResult):
        gesture = cast(Category, result.gestures[0][0])
        if not gesture.category_name == "Pointing_Up" or gesture.score < 0.8:
            return
        
        if is_touching(result.hand_landmarks[0], 4, 11, 3):
            if not self.has_left_click:
                    pyautogui.leftClick()
                    self.has_left_click = True
            if self.debug:
                cv2.putText(self.frame, "Left Click", [10, 50], cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        else:
            self.has_left_click = False

    def right_click(self, result: GestureRecognizerResult):
        gesture = cast(Category, result.gestures[0][0])
        if not gesture.category_name == "Pointing_Up" or gesture.score < 0.8:
            return
        
        if is_touching(result.hand_landmarks[0], 4, 12, 3):
            if not self.has_right_click:
                    pyautogui.rightClick()
                    self.has_right_click = True
            if self.debug:
                cv2.putText(self.frame, "Right Click", [10, 50], cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        else:
            self.has_right_click = False

    def scroll(self, result: GestureRecognizerResult):
        gesture = cast(Category, result.gestures[0][0])
        if not gesture.category_name == "Closed_Fist" or gesture.score < 0.8 or not is_touching(result.hand_landmarks[0], 4, 8):
            return
        
        amount = deadzone(round((self.prev_pos.x - self.curr_pos.x) * screen_width))
        pyautogui.scroll(amount * 9)

        if self.debug:
            cv2.putText(self.frame, f"Scoll: {amount}", (15, 50), cv2.FONT_HERSHEY_DUPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
