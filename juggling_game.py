import sys
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget,QToolButton,QSizePolicy, 
                             QFrame, QSizePolicy, QButtonGroup, QRadioButton, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist

# --- EXE PATH HELPER ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- GAME CONFIGURATION ---
LEVEL_CONFIG = {
    1: { "balls": 1, "gravity": 0.12, "base_speed": 2.5, "desc": "Level 1: Warm Up", "details": "1 Ball  •  Slow" },
    2: { "balls": 1, "gravity": 0.18, "base_speed": 4.0, "desc": "Level 2: Standard", "details": "1 Ball  •  Normal" },
    3: { "balls": 2, "gravity": 0.25, "base_speed": 5.0, "desc": "Level 3: Double Trouble", "details": "2 Balls  •  Fast" },
    4: { "balls": 2, "gravity": 0.35, "base_speed": 6.5, "desc": "Level 4: Expert Mode", "details": "2 Balls  •  Extreme" }
}

# --- STYLESHEET ---
STYLESHEET = """
    QWidget { font-family: 'Segoe UI', sans-serif; }
    
    QLabel#Header { font-size: 36px; font-weight: bold; color: #00fff5; background-color: transparent; }
    QLabel#SubHeader { font-size: 20px; color: #e0e0e0; background-color: transparent; font-weight: bold; }
    QLabel#NormalText { font-size: 18px; color: #ffffff; background-color: transparent; }
    QLabel#TimeLabel { font-size: 48px; font-weight: bold; color: #ffcc00; background-color: transparent; }
    
    QLabel#GameOverTitle { font-size: 60px; font-weight: bold; color: #ff2a6d; }
    QLabel#FinalScore { font-size: 40px; font-weight: bold; color: #00ff00; }
    QLabel#FinalStats { font-size: 24px; color: #ffffff; }

    /* Standard Buttons (Used in Setup) */
    QPushButton {
        background-color: rgba(0, 0, 0, 180);
        color: #00fff5;
        border: 2px solid #00fff5;
        border-radius: 20px;
        padding: 10px;
        font-size: 18px; font-weight: bold;
    }
    QPushButton:hover { background-color: #00fff5; color: #000000; border: 2px solid #ffffff; }
    QPushButton:pressed { background-color: #00cccc; margin-top: 2px; }
    QPushButton:disabled { background-color: rgba(50, 50, 50, 150); color: #666666; border: 2px solid #444444; }
    
    /* Image Buttons (Transparent backgrounds for PNGs) */
    QPushButton.ImgBtn {
        background-color: transparent;
        border: none;
    }
    QPushButton.ImgBtn:hover {
        background-color: rgba(255, 255, 255, 30);
        border-radius: 10px;
    }
    QPushButton.ImgBtn:pressed {
        padding-top: 5px;
    }

    QPushButton#SecondaryBtn { border: 2px solid #888888; color: #cccccc; background-color: rgba(0,0,0,150); }
    QPushButton.SmallBtn { border-radius: 10px; padding: 5px; font-size: 24px; min-width: 50px; max-width: 50px; background-color: rgba(0,0,0,180); color: #ffcc00; border: 2px solid #ffcc00; }

    QRadioButton { color: white; font-size: 24px; spacing: 15px; background-color: transparent; padding: 10px; }
    QRadioButton::indicator { width: 25px; height: 25px; border-radius: 12px; border: 2px solid #00fff5; }
    QRadioButton::indicator:checked { background-color: #00fff5; border: 2px solid white; }

    QFrame#HUD { background-color: transparent; border: none; }
    QLabel#ScoreLabel { color: #00ff00; font-size: 36px; font-weight: bold; }
    QLabel#LevelLabel { color: #ffcc00; font-size: 36px; font-weight: bold; }
    QLabel#TimerLabel { color: #00fff5; font-size: 36px; font-weight: bold; }
    
    QFrame#Panel { background-color: rgba(0,0,0,180); border-radius: 20px; border: 1px solid #444; }
"""

class GameLogic:
    def __init__(self, width=1280, height=720):
        self.mp_hands = mp.solutions.hands
        # self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.hands = self.mp_hands.Hands(
            model_complexity=0,            # Lighter model = faster/more stable for motion
            max_num_hands=2,
            min_detection_confidence=0.65, # Slightly lower to recapture fast hands
            min_tracking_confidence=0.65   # Higher for stability
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.width = width
        self.height = height
        self.active = False
        self.paused = False
        self.calibration_active = False
        self.calibration_mode = "Standing"
        self.game_over = False
        self.calibration_y = height - 100 
        self.play_area_min_x = 0
        self.play_area_max_x = width
        self.calibrated = False

        self.smooth_alpha = 0.35          # Smoothing factor (0.2-0.5: higher = more responsive)
        self.hand_states = {              # Persistent state per hand
            'Left': {'pos': None, 'prev_pos': None, 'vel': np.array([0., 0.]), 'last_update': 0},
            'Right': {'pos': None, 'prev_pos': None, 'vel': np.array([0., 0.]), 'last_update': 0}
        }
        self.max_dropout_time = 0.25
        
        # NEW: Calibration locking system
        self.calibration_locked = False
        self.calibration_lock_start_time = None
        self.calibration_lock_duration = 3.0  # 3 seconds
        self.temp_calibration_y = None
        self.temp_play_area_min_x = None
        self.temp_play_area_max_x = None
        self.calibration_progress = 0.0  # 0.0 to 1.0
        
        self.current_level = 1
        self.score = 0
        self.total_airtime = 0
        self.selected_duration = 120
        # self.time_remaining = 120
        self.elapsed_time = 0
        self.last_time_check = 0
        self.balls = []
        self.prev_hand_positions = {}
        self.last_spawn_time = 0
        self.hand_radius = 50
        self.push_force = 8
        self.max_push_angle = 35

    def reset_game(self, level):
        self.current_level = level
        self.score = 0
        self.balls = []
        self.total_airtime = 0
        self.game_over = False
        self.paused = False
        self.last_spawn_time = time.time()
        self.prev_hand_positions = {}
        # self.time_remaining = self.selected_duration
        self.elapsed_time = 0
        self.last_time_check = time.time()

    def reset_calibration(self):
        """Reset calibration state to start over"""
        self.calibrated = False
        self.calibration_locked = False
        self.calibration_lock_start_time = None
        self.temp_calibration_y = None
        self.temp_play_area_min_x = None
        self.temp_play_area_max_x = None
        self.calibration_progress = 0.0

    def calibrate(self, hand_landmarks_list):
        """
        Continuous calibration with 3-second lock requirement
        Returns: (detected_both_hands, is_locked, progress)
        """
        if len(hand_landmarks_list) < 2:
            # Reset if hands not detected
            self.calibration_lock_start_time = None
            self.calibration_progress = 0.0
            return False, False, 0.0
        
        h, w = self.height, self.width
        wrist_1_x = hand_landmarks_list[0].landmark[0].x * w
        wrist_1_y = hand_landmarks_list[0].landmark[0].y * h
        wrist_2_x = hand_landmarks_list[1].landmark[0].x * w
        wrist_2_y = hand_landmarks_list[1].landmark[0].y * h
        
        # Calculate current boundary values
        avg_y = (wrist_1_y + wrist_2_y) / 2
        current_y = int(avg_y) - 50
        min_x = min(wrist_1_x, wrist_2_x)
        max_x = max(wrist_1_x, wrist_2_x)
        
        if self.calibration_mode == "Sitting":
            current_min_x = int(min_x)
            current_max_x = int(max_x)
        else:
            span = max_x - min_x
            center = (min_x + max_x) / 2
            current_min_x = int(max(0, center - (span * 0.75)))
            current_max_x = int(min(self.width, center + (span * 0.75)))
        
        # Check if hands are steady (similar to previous position)
        if self.temp_calibration_y is not None:
            y_diff = abs(current_y - self.temp_calibration_y)
            x_min_diff = abs(current_min_x - self.temp_play_area_min_x)
            x_max_diff = abs(current_max_x - self.temp_play_area_max_x)
            
            # If hands moved significantly, reset timer
            if y_diff > 20 or x_min_diff > 30 or x_max_diff > 30:
                self.calibration_lock_start_time = None
                self.calibration_progress = 0.0
        
        # Update temporary values
        self.temp_calibration_y = current_y
        self.temp_play_area_min_x = current_min_x
        self.temp_play_area_max_x = current_max_x
        
        # Start or continue lock timer
        current_time = time.time()
        if self.calibration_lock_start_time is None:
            self.calibration_lock_start_time = current_time
            self.calibration_progress = 0.0
        else:
            elapsed = current_time - self.calibration_lock_start_time
            self.calibration_progress = min(elapsed / self.calibration_lock_duration, 1.0)
            
            # Lock achieved!
            if self.calibration_progress >= 1.0 and not self.calibration_locked:
                self.calibration_locked = True
                self.calibration_y = self.temp_calibration_y
                # Expand play area bit wider than selected (e.g., +50px each side)
                self.play_area_min_x = max(0, self.temp_play_area_min_x - 50)
                self.play_area_max_x = min(self.width, self.temp_play_area_max_x + 50)
                return True, True, 1.0
        
        return True, self.calibration_locked, self.calibration_progress

    def confirm_calibration(self):
        """Confirm and finalize the locked calibration"""
        if self.calibration_locked:
            self.calibrated = True
            return True
        return False

    def spawn_ball(self):
        config = LEVEL_CONFIG[self.current_level]
        if len(self.balls) < config["balls"]:
            safe_min = self.play_area_min_x + 30
            safe_max = self.play_area_max_x - 30
            if safe_max <= safe_min: safe_min, safe_max = 50, self.width - 50
            x = random.randint(safe_min, safe_max)
            ball = {'pos': np.array([x, 50], dtype=float), 'vel': np.array([0, config["base_speed"]], dtype=float), 'radius': 25, 'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)), 'airtime': 0, 'last_hit_time': 0}
            self.balls.append(ball)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        h, w, c = frame.shape
        hand_centers = []
        
        # Draw calibration guides
        if self.calibration_active:
            # Use temp values during calibration, locked values after
            if self.calibration_locked:
                y_line = self.calibration_y
                min_line = self.play_area_min_x
                max_line = self.play_area_max_x
                line_color = (0, 255, 0)  # Green when locked
                line_thickness = 4
            elif self.temp_calibration_y is not None:
                y_line = self.temp_calibration_y
                min_line = self.temp_play_area_min_x
                max_line = self.temp_play_area_max_x
                line_color = (0, 255, 255)  # Yellow when tracking
                line_thickness = 2
            else:
                y_line = self.calibration_y
                min_line = self.play_area_min_x
                max_line = self.play_area_max_x
                line_color = (128, 128, 128)  # Gray when waiting
                line_thickness = 2
            
            cv2.line(frame, (0, y_line), (w, y_line), line_color, line_thickness)
            cv2.line(frame, (min_line, 0), (min_line, h), line_color, line_thickness)
            cv2.line(frame, (max_line, 0), (max_line, h), line_color, line_thickness)
            
            # Draw progress bar if locking in progress
            if self.calibration_progress > 0 and not self.calibration_locked:
                bar_width = 400
                bar_height = 40
                bar_x = (w - bar_width) // 2
                bar_y = h - 100
                
                # Background
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
                # Progress
                progress_width = int(bar_width * self.calibration_progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 255), -1)
                # Border
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
                
                # Countdown text
                remaining = max(0, self.calibration_lock_duration - (self.calibration_progress * self.calibration_lock_duration))
                text = f"Hold Steady: {remaining:.1f}s"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(frame, text, (text_x, bar_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Status text
            if self.calibration_locked:
                status_text = "✓ LOCKED! Click CONFIRM to proceed"
                text_color = (0, 255, 0)
            else:
                status_text = f"Mode: {self.calibration_mode} - Hold both hands steady"
                text_color = (255, 255, 255)
            
            cv2.putText(frame, status_text, (min_line + 10, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        elif self.calibrated:
            # Show locked boundaries during game
            # cv2.line(frame, (0, self.calibration_y), (w, self.calibration_y), (0, 255, 255), 2)
            cv2.line(frame, (self.play_area_min_x, 0), (self.play_area_min_x, h), (255, 0, 0), 2)
            cv2.line(frame, (self.play_area_max_x, 0), (self.play_area_max_x, h), (255, 0, 0), 2)
        
        # if results.multi_hand_landmarks:
        #     for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        #         self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        #         if self.calibration_active:
        #             # During calibration, just draw hands
        #             self.calibrate(results.multi_hand_landmarks)
                
        #         elif self.active and not self.paused:
        #             indices = [0, 5, 9, 13, 17]
        #             cx = int(np.mean([hand_landmarks.landmark[i].x * w for i in indices]))
        #             cy = int(np.mean([hand_landmarks.landmark[i].y * h for i in indices]))
        #             center = (cx, cy)
        #             hand_centers.append(center)
        #             if hand_idx in self.prev_hand_positions:
        #                 prev = self.prev_hand_positions[hand_idx]
        #                 velocity = np.array(center) - np.array(prev)
        #             else: velocity = np.array([0,0])
        #             self.prev_hand_positions[hand_idx] = center
        #             self.check_collisions(center, velocity)
        #             cv2.circle(frame, center, self.hand_radius, (0, 255, 0), 2)
        if results.multi_hand_landmarks and results.multi_handedness:
            current_time = time.time()
            
            # Draw ALL detected hands safely (0, 1, or 2)
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            if self.calibration_active:
                detected, locked, progress = self.calibrate(results.multi_hand_landmarks)
            
            # Now process each hand with handedness
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand_info.classification[0].label  # 'Left' or 'Right'
                
                if label not in self.hand_states:
                    continue
                    
                state = self.hand_states[label]
                state['last_update'] = current_time
                
                # Raw center from stable landmarks
                indices = [0, 5, 9, 13, 17]
                raw_pos = np.array([
                    np.mean([hand_landmarks.landmark[i].x * w for i in indices]),
                    np.mean([hand_landmarks.landmark[i].y * h for i in indices])
                ])
                
                # EMA Smoothing
                if state['pos'] is None:
                    state['pos'] = raw_pos
                else:
                    state['pos'] = (self.smooth_alpha * raw_pos + 
                                (1 - self.smooth_alpha) * state['pos'])
                
                # Velocity smoothing
                dt = max(current_time - state.get('prev_time', current_time), 1/60.0)
                if state['prev_pos'] is not None:
                    new_vel = (state['pos'] - state['prev_pos']) / dt
                    state['vel'] = 0.6 * state['vel'] + 0.4 * new_vel
                
                state['prev_pos'] = state['pos'].copy()
                state['prev_time'] = current_time
                
                center = tuple(state['pos'].astype(int))
                self.check_collisions(center, state['vel'])
                cv2.circle(frame, center, self.hand_radius, (0, 255, 0), 3) # Draw smoothed center
        if self.active and not self.paused:
            current_time = time.time()
            for label, state in self.hand_states.items():
                if (state['pos'] is not None and 
                    (current_time - state['last_update']) < self.max_dropout_time):
                    # Predict position during brief dropout
                    dt = current_time - state['last_update']
                    predicted_pos = state['pos'] + state['vel'] * dt
                    predicted_pos = np.clip(predicted_pos, [0, 0], [w, h])  # Clamp to screen
                    
                    pred_center = tuple(predicted_pos.astype(int))
                    
                    # Use int() for radius
                    pred_radius = int(self.hand_radius * 0.8)   # or round(self.hand_radius * 0.8)
                    
                    self.check_collisions(pred_center, state['vel'] * 0.7)  # Reduced force
                    cv2.circle(frame, pred_center, pred_radius, (0, 180, 0), 2)  # Faint predicted
        if self.active and not self.paused and not self.game_over:
            current_time = time.time()
            # if current_time - self.last_time_check >= 1.0:
            #     self.time_remaining -= 1
            #     self.last_time_check = current_time
            #     if self.time_remaining <= 0:
            #         self.game_over = True
            #         self.time_remaining = 0
            if current_time - self.last_time_check >= 1.0:
                self.elapsed_time += 1
                self.last_time_check = current_time
                if self.elapsed_time >= self.selected_duration:
                    self.game_over = True
            self.update_physics()
            self.spawn_ball()
            self.draw_balls(frame)
        return frame

    def check_collisions(self, hand_center, hand_vel):
        current_time = time.time()
        for ball in self.balls:
            dist = np.linalg.norm(ball['pos'] - np.array(hand_center))
            if dist < (ball['radius'] + self.hand_radius) and (current_time - ball['last_hit_time']) > 0.3:
                direction = ball['pos'] - np.array(hand_center)
                norm = np.linalg.norm(direction)
                if norm > 0: direction /= norm
                angle = np.degrees(np.arctan2(direction[0], -direction[1]))
                angle = np.clip(angle, -self.max_push_angle, self.max_push_angle)
                rad = np.radians(angle)
                direction = np.array([np.sin(rad), -np.cos(rad)])
                speed = np.linalg.norm(hand_vel)
                force = self.push_force + min(speed * 0.3, 10.0)
                ball['vel'] = direction * force
                ball['last_hit_time'] = current_time
                self.score += 1

    def update_physics(self):
        config = LEVEL_CONFIG[self.current_level]
        balls_to_remove = []
        for i, ball in enumerate(self.balls):
            ball['vel'][1] += config['gravity']
            ball['pos'] += ball['vel']
            if ball['pos'][0] < self.play_area_min_x + ball['radius']:
                ball['vel'][0] *= -0.8
                ball['pos'][0] = self.play_area_min_x + ball['radius']
            elif ball['pos'][0] > self.play_area_max_x - ball['radius']:
                ball['vel'][0] *= -0.8
                ball['pos'][0] = self.play_area_max_x - ball['radius']
            if ball['pos'][1] < ball['radius']:
                ball['vel'][1] *= -0.5
                ball['pos'][1] = ball['radius']
            if ball['pos'][1] > self.height + 50:
                balls_to_remove.append(i)
                self.total_airtime += ball['airtime']
            else: ball['airtime'] += 0.03
        for i in reversed(balls_to_remove): self.balls.pop(i)

    def draw_balls(self, frame):
        for ball in self.balls:
            pos = tuple(ball['pos'].astype(int))
            cv2.circle(frame, pos, ball['radius'], ball['color'], -1)
            cv2.circle(frame, pos, ball['radius'], (255,255,255), 2)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    calibration_signal = pyqtSignal(dict)  # NEW: Signal for calibration updates
    
    def __init__(self, game_logic):
        super().__init__()
        self.game = game_logic
        self.running = True
    
    def run(self):
        source = 0
        test_cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            source = 1
            test_cap.release()
        else:
            source = 0
        
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                processed_frame = self.game.process_frame(frame)
                
                # Emit calibration status
                if self.game.calibration_active:
                    calib_data = {
                        "locked": self.game.calibration_locked,
                        "progress": self.game.calibration_progress
                    }
                    self.calibration_signal.emit(calib_data)
                
                stats = {
                    "score": self.game.score, 
                    "level": self.game.current_level, 
                    "game_over": self.game.game_over, 
                    "calibrated": self.game.calibrated,
                    # "time": self.game.time_remaining,
                    "time": self.game.elapsed_time,
                    "airtime": self.game.total_airtime
                }
                self.stats_signal.emit(stats)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(1050, 590, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            else:
                time.sleep(0.02)
            time.sleep(0.006)
        cap.release()
    
    def stop(self):
        self.running = False
        self.wait()

class JugglingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Juggling Pro")
        self.showFullScreen()
        self.game_logic = GameLogic()
        
        self.init_sounds()
        self.last_known_score = 0
        
        self.setStyleSheet(STYLESHEET)
        
        self.central_widget = QWidget()
        self.central_widget.setObjectName("CentralScreen")
        self.setCentralWidget(self.central_widget)
        
        self.set_background_image()

        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)
        
        self.init_main_menu()           # 0
        self.init_stance_screen()       # 1
        self.init_duration_screen()     # 2
        self.init_level_screen()        # 3
        self.init_instruction_screen()  # 4
        self.init_calibration_screen()  # 5
        self.init_game_screen()         # 6
        self.init_result_screen()       # 7
        
        self.thread = VideoThread(self.game_logic)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.calibration_signal.connect(self.update_calibration_ui)  # NEW
        self.thread.start()
        
        self.stack.setCurrentIndex(0)
        self.setFocusPolicy(Qt.StrongFocus)

    def init_sounds(self):
        self.music_player = QMediaPlayer()
        self.playlist = QMediaPlaylist()
        bg_path = resource_path("assets/bg.mp3")
        if os.path.exists(bg_path):
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(bg_path)))
            self.playlist.setPlaybackMode(QMediaPlaylist.Loop)
            self.music_player.setPlaylist(self.playlist)
            self.music_player.setVolume(20)
        self.hit_player = QMediaPlayer()
        hit_path = resource_path("assets/jug_hit.mp3")
        if os.path.exists(hit_path):
            self.hit_player.setMedia(QMediaContent(QUrl.fromLocalFile(hit_path)))
            self.hit_player.setVolume(100)
        self.click_player = QMediaPlayer()
        click_path = resource_path("assets/click.mp3")
        if os.path.exists(click_path):
            self.click_player.setMedia(QMediaContent(QUrl.fromLocalFile(click_path)))
            self.click_player.setVolume(100)

    def play_click(self):
        if self.click_player.mediaStatus() != QMediaPlayer.NoMedia:
            if self.click_player.state() == QMediaPlayer.PlayingState: self.click_player.stop()
            self.click_player.play()

    def play_hit(self):
        if self.hit_player.mediaStatus() != QMediaPlayer.NoMedia:
            if self.hit_player.state() == QMediaPlayer.PlayingState: self.hit_player.stop()
            self.hit_player.play()

    def start_music(self):
        if self.music_player.playlist() is not None: self.music_player.play()

    def stop_music(self):
        self.music_player.stop()

    def set_background_image(self):
        bg_path = resource_path("assets/jug_logo.png")
        style = "#CentralScreen { background-color: #1a1a2e; }"
        if os.path.exists(bg_path):
            clean_path = bg_path.replace("\\", "/")
            style = f"#CentralScreen {{ border-image: url({clean_path}) 0 0 0 0 stretch stretch; }}"
        self.central_widget.setStyleSheet(style)

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 200))
        shadow.setOffset(2, 2)
        widget.setGraphicsEffect(shadow)

    def create_image_button(self, image_name, callback):
        btn = QToolButton()
        btn.setAttribute(Qt.WA_TranslucentBackground, True)
        btn.setFixedSize(200, 180)
        btn.setContentsMargins(0, 0, 0, 0)
        btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        btn.setAutoRaise(True)
        btn.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
            }
        """)

        path = resource_path(f"assets/{image_name}")
        if os.path.exists(path):
            icon = QIcon(path)
            btn.setIcon(icon)
            btn.setIconSize(QSize(450, 180))
            btn.setFixedSize(450, 180)
        else:
            btn.setText(image_name.upper())

        btn.setCursor(Qt.PointingHandCursor)
        btn.clicked.connect(callback)
        btn.clicked.connect(self.play_click)
        return btn

    def keyPressEvent(self, event):
        if self.stack.currentIndex() == 2:
            if event.key() == Qt.Key_Right or event.key() == Qt.Key_Up:
                self.change_duration(60)
            elif event.key() == Qt.Key_Left or event.key() == Qt.Key_Down:
                self.change_duration(-60)
        elif event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)

    def change_duration(self, amount):
        self.play_click()
        new_time = self.game_logic.selected_duration + amount
        if 120 <= new_time <= 1200:
            self.game_logic.selected_duration = new_time
            self.update_time_label()

    def update_time_label(self):
        mins = self.game_logic.selected_duration // 60
        secs = self.game_logic.selected_duration % 60
        self.lbl_time_display.setText(f"{mins:02d}:{secs:02d}")

    def init_main_menu(self):
        page = QWidget()
        page.setObjectName("MainMenuPage")

        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setStyleSheet("background: transparent;")
        top_bar.setFixedHeight(200)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(20, 10, 20, 10)
        top_layout.setSpacing(0)

        top_layout.addStretch()

        btn_exit = self.create_image_button("exits.png", self.close)
        btn_exit.setIconSize(QSize(270, 180))
        btn_exit.setFixedSize(270, 180)
        top_layout.addWidget(btn_exit)

        main_layout.addWidget(top_bar)

        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(40, 20, 40, 40)
        center_layout.setAlignment(Qt.AlignCenter)
        center_layout.setSpacing(30)

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        jug_bg_path = resource_path("assets/jug_bg.png")
        if os.path.exists(jug_bg_path):
            pixmap = QPixmap(jug_bg_path).scaled(1000, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        else:
            logo_label.setText("VIRTUAL JUGGLING PRO")
            logo_label.setStyleSheet("color: #00fff5; font-size: 60px; font-weight: bold;")

        center_layout.addWidget(logo_label)
        center_layout.addStretch()

        btn_play = self.create_image_button("play.png", lambda: self.switch_to_page(1))
        btn_play.setIconSize(QSize(650, 260))
        btn_play.setFixedSize(650, 260)
        btn_play.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        center_layout.addWidget(btn_play, 0, Qt.AlignCenter)

        center_layout.addStretch()

        main_layout.addWidget(center_widget, stretch=1)

        page.setLayout(main_layout)
        self.stack.addWidget(page)

    def init_stance_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QFrame()
        container.setObjectName("Panel")
        container.setFixedWidth(600)
        con_layout = QVBoxLayout(container)
        con_layout.setSpacing(30)
        con_layout.setContentsMargins(40,40,40,40)

        lbl_title = QLabel("STEP 1: SELECT STANCE")
        lbl_title.setObjectName("Header")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.rb_standing = QRadioButton(" Standing Mode")
        self.rb_sitting = QRadioButton(" Sitting Mode")
        self.rb_standing.setChecked(True)
        self.rb_sitting.clicked.connect(self.play_click)
        self.rb_standing.clicked.connect(self.play_click)
        
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.rb_standing)
        radio_layout.addWidget(self.rb_sitting)
        radio_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        con_layout.addWidget(lbl_title)
        con_layout.addLayout(radio_layout)

        nav_layout = QHBoxLayout()
        btn_back = QPushButton("BACK")
        btn_back.setObjectName("SecondaryBtn")
        btn_back.clicked.connect(lambda: self.switch_to_page(0))
        
        btn_next = QPushButton("NEXT")
        btn_next.clicked.connect(lambda: self.switch_to_page(2))
        
        nav_layout.addWidget(btn_back)
        nav_layout.addWidget(btn_next)
        
        con_layout.addLayout(nav_layout)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_duration_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QFrame()
        container.setObjectName("Panel")
        container.setFixedWidth(600)
        con_layout = QVBoxLayout(container)
        con_layout.setSpacing(30)
        con_layout.setContentsMargins(40,40,40,40)

        lbl_title = QLabel("STEP 2: GAME DURATION")
        lbl_title.setObjectName("Header")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_sub = QLabel("(2 min - 20 min)")
        lbl_sub.setObjectName("SubHeader")
        lbl_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)

        time_layout = QHBoxLayout()
        btn_minus = QPushButton("-")
        btn_minus.setProperty("class", "SmallBtn")
        btn_minus.clicked.connect(lambda: self.change_duration(-60))
        
        self.lbl_time_display = QLabel("02:00")
        self.lbl_time_display.setObjectName("TimeLabel")
        
        btn_plus = QPushButton("+")
        btn_plus.setProperty("class", "SmallBtn")
        btn_plus.clicked.connect(lambda: self.change_duration(60))
        
        time_layout.addStretch()
        time_layout.addWidget(btn_minus)
        time_layout.addSpacing(20)
        time_layout.addWidget(self.lbl_time_display)
        time_layout.addSpacing(20)
        time_layout.addWidget(btn_plus)
        time_layout.addStretch()

        con_layout.addWidget(lbl_title)
        con_layout.addWidget(lbl_sub)
        con_layout.addLayout(time_layout)

        nav_layout = QHBoxLayout()
        btn_back = QPushButton("BACK")
        btn_back.setObjectName("SecondaryBtn")
        btn_back.clicked.connect(lambda: self.switch_to_page(1))
        
        btn_next = QPushButton("NEXT")
        btn_next.clicked.connect(lambda: self.switch_to_page(3))
        
        nav_layout.addWidget(btn_back)
        nav_layout.addWidget(btn_next)
        con_layout.addLayout(nav_layout)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_level_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QFrame()
        container.setObjectName("Panel")
        container.setFixedWidth(600)
        con_layout = QVBoxLayout(container)
        con_layout.setSpacing(15)
        con_layout.setContentsMargins(40,40,40,40)

        lbl_title = QLabel("STEP 3: DIFFICULTY")
        lbl_title.setObjectName("Header")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        con_layout.addWidget(lbl_title)
        
        for lvl, data in LEVEL_CONFIG.items():
            btn = QPushButton(f"{data['desc']}\n{data['details']}")
            btn.setMinimumHeight(70)
            btn.clicked.connect(lambda checked, l=lvl: self.start_calibration_setup(l))
            con_layout.addWidget(btn)

        btn_back = QPushButton("BACK")
        btn_back.setObjectName("SecondaryBtn")
        btn_back.clicked.connect(lambda: self.switch_to_page(2))
        
        con_layout.addSpacing(10)
        con_layout.addWidget(btn_back)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_instruction_screen(self):
        page = QWidget()
        page.setObjectName("InstructionPage")
        
        img_path = resource_path("assets/instruction (2).png")
        if os.path.exists(img_path):
            clean_path = img_path.replace("\\", "/")
            page.setStyleSheet(f"#InstructionPage {{ border-image: url({clean_path}) 0 0 0 0 stretch stretch; }}")
        else:
            page.setStyleSheet("#InstructionPage { background-color: #1a1a2e; }")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 50)
        layout.setSpacing(20)

        layout.addStretch()

        btn_start = QPushButton("START CALIBRATION")
        btn_start.setFixedWidth(350)
        btn_start.setMinimumHeight(70)
        btn_start.setCursor(Qt.PointingHandCursor)
        btn_start.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #009999, stop:1 #005555);
                color: white;
                border: 2px solid #00fff5;
                border-radius: 35px;
                font-size: 22px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00fff5;
                color: black;
                border: 2px solid white;
            }
            QPushButton:pressed {
                background-color: #007777;
                margin-top: 2px;
            }
        """)
        btn_start.clicked.connect(lambda: self.switch_to_page(5))
        btn_start.clicked.connect(self.play_click)

        btn_back = QPushButton("BACK")
        btn_back.setFixedWidth(200)
        btn_back.setMinimumHeight(50)
        btn_back.setObjectName("SecondaryBtn")
        btn_back.clicked.connect(lambda: self.switch_to_page(3))
        btn_back.clicked.connect(self.play_click)

        layout.addWidget(btn_start, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(btn_back, 0, Qt.AlignmentFlag.AlignCenter)
        
        page.setLayout(layout)
        self.stack.addWidget(page)

    def init_calibration_screen(self):
        page = QWidget()
        page.setObjectName("CalibrationPage")

        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        video_container = QWidget()
        video_container.setStyleSheet("background-color: #0d1b2a;")
        main_layout.addWidget(video_container, stretch=1)

        self.video_label_calib = QLabel(video_container)
        self.video_label_calib.setAlignment(Qt.AlignCenter)
        self.video_label_calib.setScaledContents(True)
        self.video_label_calib.setStyleSheet("border: none;")

        top_overlay = QFrame(video_container)
        top_overlay.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(0, 31, 63, 220), stop:1 rgba(0, 15, 35, 180));
            border-bottom: 3px solid #00fff5;
            border-radius: 0 0 16px 16px;
        """)
        top_overlay.setFixedHeight(200)

        top_layout = QVBoxLayout(top_overlay)
        top_layout.setContentsMargins(30, 20, 30, 10)
        top_layout.setSpacing(8)

        lbl_title = QLabel("CALIBRATION SETUP")
        lbl_title.setStyleSheet("""
            color: #00fff5; font-size: 38px; font-weight: bold;
            background: transparent; text-shadow: 0 0 10px #00fff5;
        """)
        lbl_title.setAlignment(Qt.AlignCenter)

        self.lbl_calib_instr = QLabel(
            "Stretch BOTH arms out and hold steady for 3 seconds"
        )
        self.lbl_calib_instr.setStyleSheet("""
            color: #e0f7ff; font-size: 30px; line-height: 1.5;
            background: transparent;
        """)
        self.lbl_calib_instr.setWordWrap(True)
        self.lbl_calib_instr.setAlignment(Qt.AlignCenter)

        top_layout.addWidget(lbl_title)
        top_layout.addWidget(self.lbl_calib_instr)

        # UPDATED BOTTOM OVERLAY with TWO buttons
        bottom_overlay = QFrame(video_container)
        bottom_overlay.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(0, 15, 35, 200), stop:1 rgba(0, 31, 63, 240));
            border-top: 3px solid #00fff5;
            border-radius: 16px 16px 0 0;
        """)
        bottom_overlay.setFixedHeight(120)

        bottom_layout = QHBoxLayout(bottom_overlay)
        bottom_layout.setContentsMargins(50, 20, 50, 20)
        bottom_layout.setSpacing(40)

        btn_back = QPushButton("BACK")
        btn_back.setFixedSize(180, 60)
        btn_back.setStyleSheet("""
            background: rgba(80, 80, 100, 180); color: #cccccc;
            border: 2px solid #666688; border-radius: 12px;
            font-size: 18px; font-weight: bold;
        """)
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_back.clicked.connect(lambda: self.switch_to_page(4))
        btn_back.clicked.connect(self.play_click)

        # REDO CALIBRATION button
        self.btn_redo_calib = QPushButton("REDO")
        self.btn_redo_calib.setFixedSize(200, 60)
        self.btn_redo_calib.setStyleSheet("""
            QPushButton {
                background: rgba(255, 140, 0, 180); color: white;
                border: 2px solid #ff8800; border-radius: 12px;
                font-size: 18px; font-weight: bold;
            }
            QPushButton:hover { background: rgba(255, 165, 0, 220); }
            QPushButton:pressed { background: rgba(255, 120, 0, 200); }
        """)
        self.btn_redo_calib.setCursor(Qt.PointingHandCursor)
        self.btn_redo_calib.clicked.connect(self.redo_calibration)
        self.btn_redo_calib.clicked.connect(self.play_click)

        # CONFIRM button (starts disabled)
        self.btn_confirm_calib = QPushButton("CONFIRM")
        self.btn_confirm_calib.setFixedSize(240, 80)
        self.btn_confirm_calib.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #00d4ff, stop:1 #0095ff);
                color: white; font-size: 24px; font-weight: bold;
                border: none; border-radius: 40px;
            }
            QPushButton:hover { background: #40e0ff; }
            QPushButton:pressed { background: #0077cc; }
            QPushButton:disabled {
                background: rgba(80, 80, 80, 150);
                color: #666666;
            }
        """)
        self.btn_confirm_calib.setCursor(Qt.PointingHandCursor)
        self.btn_confirm_calib.setEnabled(False)  # Disabled initially
        self.btn_confirm_calib.clicked.connect(self.finish_calibration)
        self.btn_confirm_calib.clicked.connect(self.play_click)

        bottom_layout.addWidget(btn_back)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_redo_calib)
        bottom_layout.addWidget(self.btn_confirm_calib)

        def update_overlays():
            w = video_container.width()
            h = video_container.height()
            self.video_label_calib.setGeometry(0, 0, w, h)
            top_overlay.setGeometry(0, 0, w, top_overlay.height())
            bottom_overlay.setGeometry(0, h - bottom_overlay.height(), w, bottom_overlay.height())

        video_container.resizeEvent = lambda event: update_overlays()
        update_overlays()

        page.setLayout(main_layout)
        self.stack.addWidget(page)

    def init_game_screen(self):
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Video container fills the entire page
        video_container = QWidget()
        # video_container.setStyleSheet("background-color: black;")
        main_layout.addWidget(video_container, stretch=1)

        # Full-screen video label inside the container
        self.video_label_game = QLabel(video_container)
        self.video_label_game.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label_game.setScaledContents(True)
        self.video_label_game.setStyleSheet("border: none;")

        # HUD overlaid at the top of the video container
        hud_frame = QFrame(video_container)
        hud_frame.setObjectName("HUD")
        hud_frame.setFixedHeight(75)
        hud_layout = QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(20, 5, 20, 5)

        self.lbl_score = QLabel("SCORE: 000")
        self.lbl_score.setObjectName("ScoreLabel")
        self.add_shadow(self.lbl_score)
        self.lbl_level = QLabel("LEVEL: 1")
        self.lbl_level.setObjectName("LevelLabel")
        self.add_shadow(self.lbl_level)
        self.lbl_timer = QLabel("TIME: 02:00")
        self.lbl_timer.setObjectName("TimerLabel")
        self.add_shadow(self.lbl_timer)
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: white; font-weight: bold; font-size: 18px;")

        btn_menu = QPushButton("END GAME")
        btn_menu.setFixedSize(200, 55)
        btn_menu.setStyleSheet("font-size: 20px; font-weight: bold; padding: 5px;")
        btn_menu.clicked.connect(self.end_game_and_show_results)

        hud_layout.addWidget(self.lbl_score)
        hud_layout.addSpacing(30)
        hud_layout.addWidget(self.lbl_level)
        hud_layout.addSpacing(30)
        hud_layout.addWidget(self.lbl_timer)
        hud_layout.addStretch()
        hud_layout.addWidget(self.lbl_status)
        hud_layout.addSpacing(20)
        hud_layout.addWidget(btn_menu)

        # Resize handler: video fills container, HUD floats on top
        def update_game_overlays():
            w = video_container.width()
            h = video_container.height()
            self.video_label_game.setGeometry(0, 0, w, h)
            hud_frame.setGeometry(0, 0, w, hud_frame.height())

        video_container.resizeEvent = lambda event: update_game_overlays()
        update_game_overlays()

        page.setLayout(main_layout)
        self.stack.addWidget(page)

    def init_result_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        container = QFrame()
        container.setObjectName("Panel")
        container.setFixedWidth(600)
        con_layout = QVBoxLayout(container)
        con_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        con_layout.setSpacing(20)
        con_layout.setContentsMargins(40, 40, 40, 40)
        
        lbl_title = QLabel("TIME'S UP!")
        lbl_title.setObjectName("GameOverTitle")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(lbl_title)
        
        self.lbl_final_score = QLabel("FINAL SCORE: 0")
        self.lbl_final_score.setObjectName("FinalScore")
        self.lbl_final_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.add_shadow(self.lbl_final_score)
        
        self.lbl_final_stats = QLabel("Total Airtime: 0s")
        self.lbl_final_stats.setObjectName("FinalStats")
        self.lbl_final_stats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(20)
        
        btn_restart = QPushButton("MAIN MENU")
        btn_restart.setMinimumWidth(200)
        btn_restart.clicked.connect(self.return_to_menu)
        
        btn_quit = QPushButton("QUIT GAME")
        btn_quit.setMinimumWidth(200)
        btn_quit.setStyleSheet("background-color: #4a0e0e; border-color: #ff0000;")
        btn_quit.clicked.connect(self.close)
        
        btn_row.addWidget(btn_restart)
        btn_row.addWidget(btn_quit)
        
        con_layout.addWidget(lbl_title)
        con_layout.addWidget(self.lbl_final_score)
        con_layout.addWidget(self.lbl_final_stats)
        con_layout.addSpacing(20)
        con_layout.addLayout(btn_row)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    def switch_to_page(self, index):
        self.play_click()
        self.stack.setCurrentIndex(index)

    def start_calibration_setup(self, level):
        self.play_click()
        self.selected_level = level
        mode = "Sitting" if self.rb_sitting.isChecked() else "Standing"
        self.game_logic.calibration_mode = mode
        self.game_logic.active = False
        self.game_logic.calibration_active = True
        self.game_logic.reset_calibration()  # NEW: Reset calibration state
        self.btn_confirm_calib.setEnabled(False)  # Disable confirm button
        self.switch_to_page(4)

    def redo_calibration(self):
        """Reset calibration and start over"""
        self.play_click()
        self.game_logic.reset_calibration()
        self.btn_confirm_calib.setEnabled(False)
        self.lbl_calib_instr.setText("Stretch BOTH arms out and hold steady for 3 seconds")
        self.lbl_calib_instr.setStyleSheet("""
            color: #e0f7ff; font-size: 30px; line-height: 1.5;
            background: transparent;
        """)

    def update_calibration_ui(self, calib_data):
        """Update UI based on calibration progress"""
        if self.stack.currentIndex() != 5:  # Only update on calibration screen
            return
        
        if calib_data["locked"]:
            # Calibration is locked!
            self.btn_confirm_calib.setEnabled(True)
            self.lbl_calib_instr.setText("✓ LOCKED! Click CONFIRM to start game")
            self.lbl_calib_instr.setStyleSheet("""
                color: #00ff00; font-size: 32px; font-weight: bold;
                background: transparent; text-shadow: 0 0 10px #00ff00;
            """)
        elif calib_data["progress"] > 0:
            # In progress
            remaining = 3.0 * (1.0 - calib_data["progress"])
            self.lbl_calib_instr.setText(f"Hold steady... {remaining:.1f}s")
            self.lbl_calib_instr.setStyleSheet("""
                color: #ffff00; font-size: 30px; font-weight: bold;
                background: transparent;
            """)
            self.btn_confirm_calib.setEnabled(False)
        else:
            # Waiting for hands
            self.lbl_calib_instr.setText("Stretch BOTH arms out and hold steady for 3 seconds")
            self.lbl_calib_instr.setStyleSheet("""
                color: #e0f7ff; font-size: 30px; line-height: 1.5;
                background: transparent;
            """)
            self.btn_confirm_calib.setEnabled(False)

    def finish_calibration(self):
        self.play_click()
        if self.game_logic.confirm_calibration():
            self.game_logic.calibration_active = False
            self.start_game()
        else:
            self.lbl_calib_instr.setText("❌ Calibration not locked yet!")
            self.lbl_calib_instr.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 24px;")

    def start_game(self):
        self.game_logic.reset_game(self.selected_level)
        self.last_known_score = 0
        self.game_logic.active = True
        self.switch_to_page(6)
        self.start_music()

    def return_to_menu(self):
        self.play_click()
        self.stop_music()
        self.game_logic.active = False
        self.game_logic.calibration_active = False
        self.switch_to_page(0)

    def end_game_and_show_results(self):
        self.play_click()
        self.stop_music()
        self.game_logic.active = False
        self.game_logic.game_over = True
        self.game_logic.paused = False
        stats = {
            "score": self.game_logic.score,
            "level": self.game_logic.current_level,
            "game_over": True,
            "time": self.game_logic.elapsed_time,
            "airtime": self.game_logic.total_airtime
        }
        self.show_results(stats)

    def update_image(self, qt_img):
        idx = self.stack.currentIndex()
        if idx == 5: 
            self.video_label_calib.setPixmap(QPixmap.fromImage(qt_img))
        elif idx == 6: 
            self.video_label_game.setPixmap(QPixmap.fromImage(qt_img))
            self.video_label_game.setScaledContents(True)

    def update_stats(self, stats):
        if self.stack.currentIndex() == 6:
            current_score = stats['score']
            self.lbl_score.setText(f"SCORE: {current_score:03d}")
            self.lbl_level.setText(f"LEVEL: {stats['level']}")
            
            if current_score > self.last_known_score:
                self.play_hit()
                self.last_known_score = current_score
            
            rem_sec = stats['time']
            mins = rem_sec // 60
            secs = rem_sec % 60
            self.lbl_timer.setText(f"TIME: {mins:02d}:{secs:02d}")
            # if rem_sec < 10:
            remaining = self.game_logic.selected_duration - rem_sec
            if remaining < 10:
                 self.lbl_timer.setStyleSheet("color: red; font-size: 24px; font-weight: bold;")
            else:
                 self.lbl_timer.setStyleSheet("color: #00fff5; font-size: 24px; font-weight: bold;")
            
            if stats['game_over']:
                self.game_logic.active = False
                self.show_results(stats)

    # def show_results(self, stats):
    #     self.stop_music()
    #     self.lbl_final_score.setText(f"FINAL SCORE: {stats['score']}")
    #     self.lbl_final_stats.setText(f"Level: {stats['level']}  |  Total Airtime: {stats['airtime']:.1f}s")
    #     self.switch_to_page(7)
    def show_results(self, stats):
        self.stop_music()
        self.lbl_final_score.setText(f"FINAL SCORE: {stats['score']}")
        
        # Show time used nicely
        elapsed = stats['time']
        mins = elapsed // 60
        secs = elapsed % 60
        time_text = f"Time Used: {mins:02d}:{secs:02d} / {self.game_logic.selected_duration // 60:02d}:{self.game_logic.selected_duration % 60:02d}"
        
        self.lbl_final_stats.setText(
            f"Level: {stats['level']}\n"
            f"{time_text}\n"
            f"Total Airtime: {stats['airtime']:.1f}s"
        )
        
        self.switch_to_page(7)

    # def closeEvent(self, event):
    #     self.thread.stop()
    #     event.accept()
    def closeEvent(self, event):
        print("Close requested...")  # ← temporary debug print

        # Immediately tell game logic to stop heavy processing
        self.game_logic.active = False
        self.game_logic.calibration_active = False

        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.running = False          # signal loop to exit
            self.thread.wait(1200)               # wait max ~1.2 seconds

            if self.thread.isRunning():
                print("Thread didn't exit in 1.2s – giving up wait")

        # Stop sounds / music early
        self.stop_music()
        if hasattr(self, 'hit_player'):
            try:
                self.hit_player.stop()
            except:
                pass

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JugglingWindow()
    window.show()
    sys.exit(app.exec())