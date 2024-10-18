import cv2
from utils import get_distance, get_angle
import pyautogui

screen_width, screen_height = pyautogui.size()

def normalize(value: int, norm_value = 4) -> int:
    if abs(value) < norm_value:
        return 0
    return value

def move_mouse(prev, curr, x_scale = 2, y_scale = 3):
    if prev is not None and curr is not None:
        x = normalize(int((curr.x - prev.x) * screen_width))
        y = normalize(int((curr.y - prev.y) * screen_height))
        try:
            pyautogui.moveRel(x * x_scale, y * y_scale, _pause=False)
        except:
            print("Tried to move out of bounds")

def scroll(prev, curr, scale = 8):
    if prev is not None and curr is not None:
        x = normalize(int((curr.x - prev.x) * screen_width))
        pyautogui.scroll(x * scale)

class Gestures():
    def __init__(self, hand_landmarks, position_index, debug = False) -> None:
        self.hand_landmarks = hand_landmarks
        self.is_active = False
        self.previous_pos = None
        self.curr_pos = None
        self.position_index = position_index
        self.debug = debug

        self.has_toggle = False
        self.has_left_click = False
        self.has_right_click = False

    def find_position(self, processed):
        """Find finger position"""
        if processed.multi_hand_landmarks:
            hand_landmarks = processed.multi_hand_landmarks[0]
            position = hand_landmarks.landmark[self.position_index]
            return position
        return None
    
    def is_touching(self, landmarks, target = 3):
        dist = get_distance(landmarks)
        return dist <= target
    
    def is_angle(self, landmarks, target, range):
        angle = get_angle(*landmarks)
        if (angle > 180):
            angle = 360 - angle
        return abs(target - angle) <= range
    
    def toggle_active(self, landmarks):
        is_touching = self.is_touching([landmarks[self.hand_landmarks.THUMB_TIP], landmarks[self.hand_landmarks.PINKY_TIP]])
        if is_touching:
            if not self.has_toggle:
                self.is_active = not self.is_active
                self.has_toggle = True
        else:
            self.has_toggle = False
    
    def left_click(self, frame, landmarks):
        is_touching = self.is_touching([landmarks[self.hand_landmarks.THUMB_TIP], landmarks[self.hand_landmarks.MIDDLE_FINGER_DIP]], 3.5)
        is_bent = self.is_angle([landmarks[2], landmarks[3], landmarks[4]], 110, 15)
        if is_touching and is_bent:
            if not self.has_left_click:
                pyautogui.leftClick()
                self.has_left_click = True
            if self.debug:
                cv2.putText(frame, "Left Click", [25, 75], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.has_left_click = False

    def right_click(self, frame, landmarks):
        is_touching = self.is_touching([landmarks[self.hand_landmarks.THUMB_TIP], landmarks[self.hand_landmarks.MIDDLE_FINGER_TIP]], 2)
        if is_touching:
            if not self.has_right_click:
                pyautogui.rightClick()
                self.has_right_click = True
            if self.debug:
                cv2.putText(frame, "Right Click", [25, 75], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            self.has_right_click = False
    
    def move_rel(self, frame, landmarks):
        if self.previous_pos is None:
            return
        
        is_touching = self.is_touching([landmarks[self.hand_landmarks.THUMB_TIP], landmarks[self.hand_landmarks.MIDDLE_FINGER_PIP]], 3.5)
        if is_touching:
            move_mouse(self.previous_pos, self.curr_pos)
            if self.debug:
                cv2.putText(frame, "Move Mouse", [25, 75], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    def scroll(self, frame, landmarks):
        if self.previous_pos is None:
            return
        
        is_touching = self.is_touching([landmarks[self.hand_landmarks.THUMB_TIP], landmarks[self.hand_landmarks.INDEX_FINGER_PIP]])
        if is_touching:
            scroll(self.previous_pos, self.curr_pos)
            if self.debug:
                cv2.putText(frame, "Scroll", [25, 75], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    def detect_gesture(self, frame, landmarks, processed):
        if self.debug:
            if self.is_active:
                cv2.putText(frame, "Active", [25, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Inactive", [25, 50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(landmarks) >= 21:
            self.curr_pos = self.find_position(processed)

            self.toggle_active(landmarks)

            if self.is_active:
                self.left_click(frame, landmarks)
                self.right_click(frame, landmarks)
                self.move_rel(frame, landmarks)
                self.scroll(frame, landmarks)

            self.previous_pos = self.curr_pos
