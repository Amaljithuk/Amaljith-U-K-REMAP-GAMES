import sys
import os
import ctypes
from pathlib import Path
import pygame
# IMPORTANT: run DLL setup before importing native-extension packages (mediapipe/cv2)
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio init failed: {e}")

# =====================================================
# EXE PATH HELPER (KEEP THIS!)
# =====================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
import cv2
import numpy as np
import random
import math
import time

import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, QFrame, QSizePolicy, QGraphicsOpacityEffect, QSplashScreen)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect, QEasingCurve, QPointF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QLinearGradient, QGradient, QPainter, QPen, QRadialGradient, QConicalGradient

# Initialize Audio (don't crash if audio device is unavailable)
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio init failed: {e}")

# =====================================================
# INLINE SPLASH SCREEN (from splash_screen.py)
# =====================================================
class DynamicSplashScreen(QSplashScreen):
    def __init__(self, app):
        # Start with reasonable size - will be resized to full screen
        pixmap = QPixmap(1280, 720)
        pixmap.fill(Qt.transparent)
        super().__init__(pixmap)

        self.app = app
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Background
        bg_path = resource_path("assets/bg.jpg")
        self.bg_image = QPixmap(bg_path) if os.path.exists(bg_path) else None

        # Title image or fallback
        title_path = resource_path("assets/download.png")
        self.title_pixmap = None
        if os.path.exists(title_path):
            self.title_pixmap = QPixmap(title_path)

        self.current_progress = 0

        # Simple timer for progress
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(80)  # smooth ~12-15% per second

        self.center_and_resize()

    def center_and_resize(self):
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

    def update_progress(self):
        if self.current_progress < 100:
            self.current_progress += random.randint(1, 4)
            self.current_progress = min(100, self.current_progress)
        else:
            self.progress_timer.stop()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        w = self.width()
        h = self.height()

        # 1. Jungle background
        if self.bg_image:
            scaled_bg = self.bg_image.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            painter.drawPixmap(0, 0, scaled_bg)

        # 2. Subtle dark overlay
        painter.setBrush(QColor(10, 20, 30, 140))
        painter.setPen(Qt.NoPen)
        painter.drawRect(self.rect())

        # 3. Main Title - big and centered
        if self.title_pixmap:
            title = self.title_pixmap.scaled(int(w * 0.65), int(h * 0.28),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
            tx = (w - title.width()) // 2
            ty = int(h * 0.18)
            painter.drawPixmap(tx, ty, title)
        else:
            font = QFont("Verdana", int(h * 0.12), QFont.Bold)
            painter.setFont(font)
            painter.setPen(QColor(80, 255, 140))  # vibrant green
            painter.drawText(QRect(0, int(h * 0.18), w, int(h * 0.20)),
                            Qt.AlignCenter, "SNAKE AR")

        # 4. Subtitle
        sub_font = QFont("Verdana", int(h * 0.05), QFont.Bold)
        painter.setFont(sub_font)
        painter.setPen(QColor(255, 220, 80))  # warm gold

        # 5. Progress Bar - clean, centered, modern
        bar_w = int(w * 0.50)
        bar_h = int(h * 0.05)
        bar_x = (w - bar_w) // 2
        bar_y = int(h * 0.55)

        # Background track
        painter.setBrush(QColor(40, 40, 50, 220))
        painter.setPen(QPen(QColor(80, 80, 90), 3))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 25, 25)

        # Filled progress
        fill_w = int(bar_w * (self.current_progress / 100))
        if fill_w > 0:
            grad = QLinearGradient(bar_x, bar_y, bar_x + fill_w, bar_y + bar_h)
            grad.setColorAt(0, QColor(100, 255, 140))
            grad.setColorAt(1, QColor(60, 220, 100))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 25, 25)

        # Percentage text on right
        perc_font = QFont("Verdana", int(h * 0.045), QFont.Bold)
        painter.setFont(perc_font)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(bar_x + bar_w + 20, bar_y + bar_h // 2 + 15,
                        f"{self.current_progress}%")

        # 6. Small status text below bar
        status_font = QFont("Verdana", int(h * 0.032))
        painter.setFont(status_font)
        painter.setPen(QColor(220, 255, 220, 220))
        painter.drawText(QRect(0, int(h * 0.68), w, int(h * 0.08)),
                        Qt.AlignCenter, "Loading Jungle Adventure...")

        # 7. Tiny version at very bottom
        ver_font = QFont("Arial", int(h * 0.018))
        painter.setFont(ver_font)
        painter.setPen(QColor(180, 180, 200, 180))
        painter.drawText(QRect(0, h - int(h * 0.05), w, int(h * 0.05)),
                        Qt.AlignCenter, "Version 1.0 | © 2024 Snake AR")

def show_splash():
    """Create and show the splash screen."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    splash = DynamicSplashScreen(app)
    splash.show()
    splash.raise_()
    splash.activateWindow()
    app.processEvents()
    return splash, app

# =====================================================
# EXE PATH HELPER
# =====================================================
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# =====================================================
# ASSET MANAGER
# =====================================================
class Assets:
    def __init__(self):
        self.images = {}
        self.sounds = {}
        self.load_images()
        self.load_sounds()

    def load_images(self):
        def load_img(path, size=None):
            full_path = resource_path(path)
            if os.path.exists(full_path):
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    if size:
                        img = cv2.resize(img, size)
                    return img
            return None

        self.images['head'] = load_img("assets/snake_head.png", (80, 80))
        self.images['food'] = load_img("assets/apple.png", (65, 65))
        # Changed crosshair to larger size and will be colored red in code
        self.images['crosshair'] = load_img("assets/crosshair.png", (60, 60))
        self.images['heart_full'] = load_img("assets/heart_full.png", (40, 40))
        self.images['heart_empty'] = load_img("assets/heart_empty.png", (40, 40))
        self.images['ball'] = load_img("assets/ball.png", (50, 50))
        self.images['border'] = load_img("assets/brick_frame.png")

    def load_sounds(self):
        def load_snd(filename):
            full_path = resource_path(filename)
            if os.path.exists(full_path):
                return pygame.mixer.Sound(full_path)
            return None
         
        self.sounds['eat'] = load_snd("assets/eat.mp3")
        self.sounds['hit'] = load_snd("assets/hit.mp3")
        self.sounds['game_over'] = load_snd("assets/game_over.mp3")
        self.sounds['pause'] = load_snd("assets/pause.mp3")
        if self.sounds['pause'] is None:
            self.sounds['pause'] = load_snd("assets/hit.mp3")
        self.sounds['click'] = load_snd("assets/click.mp3")
        if self.sounds['click'] is None:
            self.sounds['click'] = load_snd("assets/hit.mp3")
        bg_music = resource_path("assets/hits.mp3")
        if os.path.exists(bg_music):
            pygame.mixer.music.load(bg_music)
            pygame.mixer.music.set_volume(0.2)

assets = Assets()

# =====================================================
# CUSTOM LOADING OVERLAY [EXPANDED & DYNAMIC]
# =====================================================
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_NoSystemBackground)
         
        # Load the snake head image
        self.head_source = None
        path = resource_path("assets/snake_head.png")
        if os.path.exists(path):
            self.head_source = QPixmap(path)

        # Animation Variables
        self.angle = 0
        self.dot_counter = 0
        self.loading_text_dots = ""
         
        # Timer for 60 FPS animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
         
        # Ensure it starts hidden until triggered
        self.hide()
         
    def animate(self):
        self.angle = (self.angle + 4) % 360
         
        # Update text dots every 20 frames
        if self.angle % 20 == 0:
            self.dot_counter = (self.dot_counter + 1) % 4
            self.loading_text_dots = "." * self.dot_counter
             
        self.update()

    def showEvent(self, event):
        self.timer.start(16)
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
         
        # 1. Dark Background
        painter.fillRect(self.rect(), QColor(20, 20, 30, 230))
         
        w = self.width()
        h = self.height()
         
        min_dim = min(w, h)
        orbit_radius = min_dim * 0.35
        base_size = min_dim * 0.08
         
        center_x = w // 2
        center_y = h // 2
         
        painter.translate(center_x, center_y)
         
        # 2. Draw Snake Body Trail
        trail_length = 12
        for i in range(trail_length):
            lag = i * 12
            trail_angle = math.radians(self.angle - lag)
            tx = orbit_radius * math.cos(trail_angle)
            ty = orbit_radius * math.sin(trail_angle)
             
            opacity = 255 - (i * (255 // trail_length))
            current_dot_size = base_size * (1 - (i / trail_length) * 0.6)
             
            painter.setBrush(QColor(0, 255, 0, int(opacity)))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(tx, ty), current_dot_size/2, current_dot_size/2)

        # 3. Draw Snake Head
        head_angle_rad = math.radians(self.angle)
        hx = orbit_radius * math.cos(head_angle_rad)
        hy = orbit_radius * math.sin(head_angle_rad)
         
        painter.save()
        painter.translate(hx, hy)
        painter.rotate(self.angle + 90)
         
        if self.head_source:
            scaled_head = self.head_source.scaled(int(base_size), int(base_size),
                                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            offset = int(base_size / 2)
            painter.drawPixmap(-offset, -offset, scaled_head)
        else:
            painter.setBrush(QColor(255, 215, 0))
            painter.drawEllipse(QPointF(0, 0), base_size/2, base_size/2)
        painter.restore()

        # 4. Draw Central Text
        font_size = int(h * 0.05)
        font = QFont("Verdana", font_size, QFont.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 215, 0))
         
        text_str = f"GAME LOADING{self.loading_text_dots}"
         
        text_rect = QRect(-w//2, -h//2, w, h)
        painter.drawText(text_rect, Qt.AlignCenter, text_str)

# =====================================================
# VISUAL HELPER FUNCTIONS
# =====================================================
def overlay_image(bg, overlay, x, y):
    if overlay is None: return bg
    h, w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    if x2 <= x1 or y2 <= y1: return bg
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    bg_crop = bg[y1:y2, x1:x2]
    overlay_crop = overlay[oy1:oy2, ox1:ox2]
    if overlay.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        for c in range(3):
            bg_crop[:, :, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * bg_crop[:, :, c])
    else:
        bg_crop[:] = overlay_crop
    bg[y1:y2, x1:x2] = bg_crop
    return bg

def rotate_image(image, angle):
    if image is None: return None
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def draw_shadow_text(img, text, pos, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x+3, y+3), cv2.FONT_HERSHEY_TRIPLEX, scale, (0, 0, 0), thickness+2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_TRIPLEX, scale, color, thickness)

def spawn_food(diff, trail, w, h, bounds=None, calib_mode="standing"):
    """Spawn food with adjusted ranges based on difficulty and mode"""
    if bounds:
        min_x, max_x, min_y, max_y = bounds
        min_x += 40; max_x -= 40; min_y += 40; max_y -= 40
    else:
        min_x, max_x = 60, w - 60
        min_y, max_y = 140, h - 60
     
    cx, cy = (min_x + max_x) // 2, (min_y + max_y) // 2
    width_span = (max_x - min_x) // 2
    height_span = (max_y - min_y) // 2

    # Adjusted spawn ranges for hard mode in standing position
    if diff == "easy":
        spawn_range_x = int(width_span * 0.3)
        spawn_range_y = int(height_span * 0.3)
    elif diff == "medium":
        spawn_range_x = int(width_span * 0.6)
        spawn_range_y = int(height_span * 0.6)
    else:  # hard
        # Make hard mode more reachable from standing position
        if calib_mode == "standing":
            spawn_range_x = int(width_span * 0.65)
            spawn_range_y = int(height_span * 0.65)
        else:
            spawn_range_x = int(width_span * 0.8)
            spawn_range_y = int(height_span * 0.8)

    for _ in range(50):
        dx = random.randint(-spawn_range_x, spawn_range_x)
        dy = random.randint(-spawn_range_y, spawn_range_y)
        x = max(min_x, min(max_x, cx + dx))
        y = max(min_y, min(max_y, cy + dy))

        collision = False
        for sx, sy in trail:
            if math.hypot(x - sx, y - sy) < 60:
                collision = True
                break
        if not collision: return (x, y)
    return (cx, cy)

# =====================================================
# SNAKE LOGIC
# =====================================================
class SnakeLogic:
    def __init__(self, width=1280, height=720):
        self.running = True
        self.mode = 'idle'
        self.width = width
        self.height = height
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.calibration_done = False
        self.game_over_flag = False
        self.calib_mode = None
        self.selected_hand = None
        self.difficulty = None
        self.duration = None
        self.bounds = None
        self.calib_min_x = width
        self.calib_max_x = 0
        self.calib_min_y = height
        self.calib_max_y = 0
        self.hold_time = 3.0
        self.hold_timer = 0
        self.score = 0
        self.snake_trail = []
        self.smooth_points = []
        self.max_trail = 2
        self.snake_pos = (640, 360)
        self.lives = 3
        self.invincible = False
        self.inv_time = 0
        self.angle = 0
        self.last_angle = 0
        self.food = None
        self.speed_factor = 0.2
        self.remaining = 0
        self.start_time = 0
        self.last_frame_time = time.time()
        self.cached_border = None
        # Detection box for center area
        self.detection_box = None
        self.show_bounding_box = False
        self.calibration_bounds_preview = None
        
        # NEW: Body detection and game pause functionality
        self.body_box = None  # Stores the body bounding box (min_x, max_x, min_y, max_y)
        self.body_box_preview = None  # For calibration preview
        self.body_hold_timer = 0
        self.body_hold_time = 3.0
        self.body_calibration_done = False
        self.game_paused = False
        self.pause_start_time = 0
        self.pause_timer = 0
        self.pause_hold_duration = 3.0  # Time needed to resume after stepping back in box
        self.show_pause_message = False
        self.out_of_bounds_time = 0
        self.body_detection_active = False  # Whether to actively check body position
        self.first_body_detection = True  # For initial calibration
        self.calibration_phase = "body"  # "body" or "arms"
        self.arms_calibration_done = False
        # Flag to track if we've already played pause sound
        self.pause_sound_played = False
        # Track if resume timer is active
        self.resume_timer_active = False

    def start_calibration(self, calib_mode):
        self.mode = 'calibration'
        self.calib_mode = calib_mode
        self.calibration_done = False
        self.arms_calibration_done = False
        self.body_calibration_done = False
        self.calibration_phase = "body"
        self.hold_timer = 0
        self.body_hold_timer = 0
        self.calib_min_x = self.width
        self.calib_max_x = 0
        self.calib_min_y = self.height
        self.calib_max_y = 0
        self.show_bounding_box = False
        self.calibration_bounds_preview = None
        self.first_body_detection = True

        # ─── CREATE FIXED DEFAULT ZONE ────────────────────────────────
        center_x = self.width // 2
        center_y = self.height // 2

        # Reasonable standing zone size – adjust if needed
        zone_width  = 380          # pixels left↔right
        zone_height = 780          # pixels top↔bottom

        self.body_box_preview = (
            center_x - zone_width//2,
            center_x + zone_width//2,
            center_y - zone_height//2,
            center_y + zone_height//2
        )

        # We will set the final zone after hold timer
        self.body_box = None

    def reset_calibration(self):
        self.hold_timer = 0
        self.body_hold_timer = 0
        self.calib_min_x = self.width
        self.calib_max_x = 0
        self.calib_min_y = self.height
        self.calib_max_y = 0
        self.body_box = None
        self.show_bounding_box = False
        self.calibration_bounds_preview = None
        self.calibration_phase = "body"
        self.body_calibration_done = False
        self.arms_calibration_done = False
        
        # Recreate default body box
        center_x, center_y = self.width // 2, self.height // 2
        body_w, body_h = 350, 500
        self.body_box_preview = (center_x - body_w//2, center_x + body_w//2, 
                               center_y - body_h//2, center_y + body_h//2)

    def force_complete(self):
        if self.calibration_phase == "body" and self.body_box_preview:
            self.body_box = self.body_box_preview
            self.body_calibration_done = True
            self.calibration_phase = "arms"
            self.hold_timer = 0  # Reset arm calibration timer
            self.show_bounding_box = False
        elif self.calibration_phase == "arms" and self.calibration_bounds_preview:
            self.bounds = self.calibration_bounds_preview
            
            # Calculate detection box (center 60% of screen)
            box_w = int(self.width * 0.6)
            box_h = int(self.height * 0.6)
            box_x1 = (self.width - box_w) // 2
            box_y1 = (self.height - box_h) // 2
            box_x2 = box_x1 + box_w
            box_y2 = box_y1 + box_h
            self.detection_box = (box_x1, box_y1, box_x2, box_y2)
            
            self.calibration_done = True
            self.mode = 'idle'
        else:
            # Fallback if no preview available
            if self.calibration_phase == "body":
                # Create a default body box (center of screen)
                center_x, center_y = self.width // 2, self.height // 2
                body_w, body_h = 350, 500  # Reasonable body size
                self.body_box = (center_x - body_w//2, center_x + body_w//2, 
                               center_y - body_h//2, center_y + body_h//2)
                self.body_calibration_done = True
                self.calibration_phase = "arms"
                self.hold_timer = 0
            else:
                self.bounds = (max(0, self.calib_min_x - 20), min(self.width, self.calib_max_x + 20),
                           max(0, self.calib_min_y - 20), min(self.height, self.calib_max_y + 20))
                
                box_w = int(self.width * 0.6)
                box_h = int(self.height * 0.6)
                box_x1 = (self.width - box_w) // 2
                box_y1 = (self.height - box_h) // 2
                box_x2 = box_x1 + box_w
                box_y2 = box_y1 + box_h
                self.detection_box = (box_x1, box_y1, box_x2, box_y2)
                
                self.calibration_done = True
                self.mode = 'idle'

    def start_game(self, diff, dur, hand):
        self.mode = 'game'
        self.difficulty = diff
        self.duration = dur
        self.selected_hand = hand
        self.game_over_flag = False
        self.game_paused = False
        self.show_pause_message = False
        self.out_of_bounds_time = 0
        self.body_detection_active = True  # Enable body detection during gameplay
        self.pause_sound_played = False
        self.resume_timer_active = False
        self.pause_timer = 0
        self.snake_trail = []
        self.smooth_points = []
        self.max_trail = 2
        self.snake_pos = (640, 360)
        self.score = 0
        self.lives = 3
        self.invincible = False
        self.inv_time = 0
        self.angle = 0
        self.last_angle = 0
        self.food = None
        self.start_time = time.time()
        # self.remaining = dur
        self.elapsed = 0
        self.cached_border = None
        
        # Speed factors remain the same
        if diff == "easy": self.speed_factor = 0.10
        elif diff == "medium": self.speed_factor = 0.20
        else: self.speed_factor = 0.35

    
    def check_body_position(self, pose_landmarks, w, h):
        if not self.body_box or not self.body_detection_active:
            return True

        min_x, max_x, min_y, max_y = self.body_box

        key_points = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
        ]

        points_inside = 0
        total_valid = 0

        for pt in key_points:
            lm = pose_landmarks.landmark[pt]
            if lm.visibility > 0.45:
                px = int(lm.x * w)
                py = int(lm.y * h)
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    points_inside += 1
                total_valid += 1

        # Require majority of visible torso points to be inside
        if total_valid >= 2:
            return points_inside >= max(2, total_valid - 1)
        else:
            # Very poor detection → assume still in zone (forgiving)
            return True

    def process_frame(self, frame):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == 'idle':
            return frame
        if not self.running:                  # we'll add self.running = True in __init__
            return frame

       

        elif self.mode == 'calibration':
            results_pose = self.pose.process(img_rgb)
            
            if results_pose.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                
                if self.calibration_phase == "body":
                    results_pose = self.pose.process(img_rgb)
                    
                    # Always show the fixed zone
                    min_x, max_x, min_y, max_y = self.body_box_preview
                    
                    # Semi-transparent fill
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (80, 220, 120), -1)
                    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                    
                    # Border
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (120, 255, 180), 3)
                    
                    # Big instruction
                    draw_shadow_text(frame, "STAND INSIDE THE GREEN ZONE", 
                                    (w//2 - 280, min_y - 60), 1.1, (220, 255, 220), 3)
                    
                    inside = False
                    
                    if results_pose.pose_landmarks:
                        # Only check shoulders + hips — ignore head/nose
                        key_points = [
                            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                            self.mp_pose.PoseLandmark.LEFT_HIP,
                            self.mp_pose.PoseLandmark.RIGHT_HIP,
                        ]
                        
                        points_inside = 0
                        total_valid = 0
                        
                        for idx in key_points:
                            lm = results_pose.pose_landmarks.landmark[idx]
                            if lm.visibility > 0.4:  # lower threshold because person is close
                                px = int(lm.x * w)
                                py = int(lm.y * h)
                                if (min_x <= px <= max_x) and (min_y <= py <= max_y):
                                    points_inside += 1
                                total_valid += 1
                        
                        # Consider "inside" if at least 3 of 4 key points are in zone
                        inside = (total_valid >= 3) and (points_inside >= 3)
                        
                        # Optional: draw faint landmarks during calibration
                        # self.mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Hold timer logic
                    if inside:
                        self.body_hold_timer += dt
                        self.body_hold_timer = min(self.body_hold_timer, self.body_hold_time)
                    else:
                        self.body_hold_timer = max(0, self.body_hold_timer - dt * 2.5)
                    
                    # Visual feedback
                    if self.body_hold_timer >= self.body_hold_time - 0.1:
                        color = (0, 255, 80)
                        thick = 6
                        status = "ZONE LOCKED! → Now stretch your arms wide"
                    elif inside:
                        progress = self.body_hold_timer / self.body_hold_time
                        g = int(180 + 75 * progress)
                        color = (60, g, 120)
                        thick = 4
                        status = f"HOLD STILL ... {self.body_hold_time - self.body_hold_timer:.1f}s"
                    else:
                        color = (80, 180, 255)
                        thick = 3
                        status = "PLEASE MOVE INTO THE GREEN ZONE"
                    
                    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), color, thick)
                    
                    draw_shadow_text(frame, status, (w//2 - 260, h - 140), 0.95, color, 3)
                    
                    # Progress bar at bottom
                    pb_x, pb_y = w//2 - 220, h - 90
                    cv2.rectangle(frame, (pb_x, pb_y), (pb_x + 440, pb_y + 40), (40,40,50), -1)
                    fill_w = int(440 * (self.body_hold_timer / self.body_hold_time))
                    if fill_w > 0:
                        cv2.rectangle(frame, (pb_x, pb_y), (pb_x + fill_w, pb_y + 40), color, -1)
                    
                    # Auto-advance when hold complete
                    if self.body_hold_timer >= self.body_hold_time and not self.body_calibration_done:
                        self.body_box = self.body_box_preview       # fixed zone becomes active
                        self.body_calibration_done = True
                        self.calibration_phase = "arms"
                        self.hold_timer = 0
                
                elif self.calibration_phase == "arms":
                    # PHASE 2: Calibrate arm stretch for game boundary
                    lw = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                    lx, ly = int(lw.x * w), int(lw.y * h)
                    rx, ry = int(rw.x * w), int(rw.y * h)
                    
                    # Only update bounds if NOT yet locked
                    if self.hold_timer < self.hold_time:
                        self.calib_min_x = min(self.calib_min_x, lx, rx)
                        self.calib_max_x = max(self.calib_max_x, lx, rx)
                        self.calib_min_y = min(self.calib_min_y, ly, ry)
                        self.calib_max_y = max(self.calib_max_y, ly, ry)
                        
                        margin = 20
                        self.calibration_bounds_preview = (
                            max(0, self.calib_min_x - margin),
                            min(w, self.calib_max_x + margin),
                            max(0, self.calib_min_y - margin),
                            min(h, self.calib_max_y + margin)
                        )
                    
                    # Check if arms are stretched wide
                    arm_span = abs(lx - rx)
                    within = arm_span > w * 0.4
                    self.show_bounding_box = True
                    
                    # Handle hold timer
                    if within:
                        if self.hold_timer < self.hold_time:
                            self.hold_timer += dt
                        else:
                            self.hold_timer = self.hold_time
                    elif self.hold_timer < self.hold_time:
                        self.hold_timer = max(0, self.hold_timer - dt)
                    
                    # Draw body box reference (faded)
                    if self.body_box:
                        min_x, max_x, min_y, max_y = self.body_box
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (100, 255, 100), -1)
                        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (100, 255, 100), 2)
                        cv2.putText(frame, "YOUR ZONE", (min_x, min_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 1)
                    
                    # Draw arm stretch box preview
                    if self.calibration_bounds_preview:
                        min_x, max_x, min_y, max_y = self.calibration_bounds_preview
                        
                        if self.hold_timer >= self.hold_time:
                            box_color = (0, 255, 0)
                            thickness = 4
                        else:
                            progress = self.hold_timer / self.hold_time
                            green = int(255 * progress)
                            red = 255 - green
                            box_color = (0, green, red)
                            thickness = 3
                        
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), box_color, thickness)
                        
                        # Add corner markers
                        corner_length = 30
                        cv2.line(frame, (min_x, min_y), (min_x + corner_length, min_y), box_color, thickness)
                        cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_length), box_color, thickness)
                        cv2.line(frame, (max_x, min_y), (max_x - corner_length, min_y), box_color, thickness)
                        cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_length), box_color, thickness)
                        cv2.line(frame, (min_x, max_y), (min_x + corner_length, max_y), box_color, thickness)
                        cv2.line(frame, (min_x, max_y), (min_x, max_y - corner_length), box_color, thickness)
                        cv2.line(frame, (max_x, max_y), (max_x - corner_length, max_y), box_color, thickness)
                        cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_length), box_color, thickness)
                        
                        # Add label
                        label = "GAME AREA - STRETCH ARMS TO EXPAND"
                        cv2.putText(frame, label, (min_x, min_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
                    
                    # Progress bar
                    bx, by = w//2 - 200, h - 80
                    cv2.rectangle(frame, (bx, by), (bx + 400, by + 30), (50, 50, 50), -1)
                    prog = int(400 * (self.hold_timer / self.hold_time))
                    if prog > 0:
                        cv2.rectangle(frame, (bx, by), (bx + prog, by + 30), (0, 255, 0) if within else (0, 165, 255), -1)
                    
                    # Instructions
                    if within:
                        if self.hold_timer < self.hold_time:
                            text = f"HOLD ARMS STRETCHED: {self.hold_time - self.hold_timer:.1f}s"
                        else:
                            text = "GAME AREA LOCKED! Press PROCEED"
                    else:
                        text = "STRETCH YOUR ARMS WIDE!"
                    
                    draw_shadow_text(frame, text, (w//2 - 250, h - 120), 0.9, (0, 255, 255), 2)

        elif self.mode == 'game':
            bound_min_x, bound_max_x, bound_min_y, bound_max_y = self.bounds
            
            if self.food is None: 
                self.food = spawn_food(self.difficulty, self.snake_trail, w, h, self.bounds, self.calib_mode)
            
            elapsed = current_time - self.start_time
            self.elapsed = int(elapsed)
            
            if self.invincible and current_time - self.inv_time > 2.0: 
                self.invincible = False
            
            # if self.remaining <= 0 or self.lives <= 0:
            #     self.game_over_flag = True
            #     self.mode = 'idle'
            #     return frame
            lives_game_over = (self.difficulty == "hard" and self.lives <= 0)
            if self.elapsed >= self.duration or lives_game_over:  # ← Ends when elapsed >= duration
                self.game_over_flag = True
                self.mode = 'idle'
                return frame

            # Check body position for pause functionality
            results_pose = self.pose.process(img_rgb)
            body_in_position = True
            
            if results_pose.pose_landmarks and self.body_box:
                # Draw body zone reference
                min_x, max_x, min_y, max_y = self.body_box
                # Draw semi-transparent body zone
                # overlay = frame.copy()
                # cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), (0, 255, 0), -1)
                # cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
                # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                # cv2.putText(frame, "YOUR ZONE", (min_x, min_y - 10), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
                # Check if body is in position
                body_in_position = self.check_body_position(results_pose.pose_landmarks, w, h)
            
            # Handle game pause based on body position
            if not body_in_position and not self.game_paused:
                # Player stepped out - pause game
                self.game_paused = True
                self.pause_start_time = current_time
                self.pause_timer = 0
                self.resume_timer_active = False
                self.out_of_bounds_time = current_time
                self.show_pause_message = True
                self.pause_sound_played = False
                if assets.sounds['pause'] and not self.pause_sound_played:
                    assets.sounds['pause'].play()
                    self.pause_sound_played = True
            
            elif body_in_position and self.game_paused:
                # Player back in box - start resume countdown
                if not self.resume_timer_active:
                    self.pause_start_time = current_time
                    self.pause_timer = 0
                    self.resume_timer_active = True
                
                time_in_box = current_time - self.pause_start_time
                self.pause_timer = time_in_box
                
                if time_in_box >= self.pause_hold_duration:
                    # Resume game after holding for 3 seconds
                    self.game_paused = False
                    self.resume_timer_active = False
                    self.pause_timer = 0
                    self.show_pause_message = False
                    self.pause_sound_played = False
                    if assets.sounds['click']:
                        assets.sounds['click'].play()
            
            elif not body_in_position and self.game_paused:
                # Still out of bounds - reset timer
                self.resume_timer_active = False
                self.pause_timer = 0
            
            # Draw pause overlay if game is paused
            if self.game_paused:
                # Darken screen
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                # Draw pause message
                if body_in_position and self.resume_timer_active:
                    # Countdown to resume
                    remaining = max(0, self.pause_hold_duration - self.pause_timer)
                    msg = f"STAY IN YOUR ZONE TO RESUME: {remaining:.1f}s"
                    color = (0, 255, 0)
                    
                    # Draw progress circle
                    center = (w//2, h//2 + 70)
                    radius = 50
                    # Draw background circle
                    cv2.circle(frame, center, radius, (100, 100, 100), 5)
                    
                    # Draw progress arc
                    angle = int(360 * (self.pause_timer / self.pause_hold_duration))
                    for i in range(0, angle, 2):  # Draw every 2 degrees for smoother circle
                        rad = math.radians(i - 90)  # Start from top
                        x = int(center[0] + radius * math.cos(rad))
                        y = int(center[1] + radius * math.sin(rad))
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                    
                    # Draw percentage text
                    percentage = int((self.pause_timer / self.pause_hold_duration) * 100)
                    cv2.putText(frame, f"{percentage}%", (center[0] - 30, center[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                elif not body_in_position:
                    msg = "MOVE BACK TO YOUR GREEN ZONE!"
                    color = (0, 0, 255)
                else:
                    msg = "WAITING..."
                    color = (255, 255, 0)
                
                # Draw main message
                draw_shadow_text(frame, "GAME PAUSED", (w//2 - 200, h//2 - 100), 1.3, (255, 255, 0), 4)
                draw_shadow_text(frame, msg, (w//2 - 250, h//2), 0.9, color, 3)
                
                # Draw body zone indicator (highlighted)
                if self.body_box:
                    min_x, max_x, min_y, max_y = self.body_box
                    if body_in_position:
                        # Green highlight when in zone
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 4)
                        cv2.putText(frame, "✓ YOU'RE IN ZONE", (min_x, min_y - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # Red highlight when out
                        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 4)
                        cv2.putText(frame, "✗ GET BACK HERE", (min_x, min_y - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                return frame  # Skip game updates while paused

            # Only process game logic if not paused
            results = self.hands.process(img_rgb)
            finger_detected = False
            
            # IMPROVED HAND DETECTION - Find hand closest to center within detection box
            target_hand_landmarks = None
            closest_dist = float('inf')
            center_x, center_y = w / 2, h / 2

            if results.multi_hand_landmarks and results.multi_handedness:
                box_x1, box_y1, box_x2, box_y2 = self.detection_box
                
                for idx, hand_info in enumerate(results.multi_handedness):
                    hand_label = hand_info.classification[0].label
                    
                    # Check if it matches the selected hand
                    if hand_label == self.selected_hand:
                        temp_landmarks = results.multi_hand_landmarks[idx]
                        
                        # Get Index Finger Tip position
                        lm8 = temp_landmarks.landmark[8]
                        cx, cy = int(lm8.x * w), int(lm8.y * h)
                        
                        # Check if hand is within detection box
                        if box_x1 <= cx <= box_x2 and box_y1 <= cy <= box_y2:
                            # Calculate distance to center
                            dist = math.hypot(cx - center_x, cy - center_y)
                            
                            # Keep the hand closest to center
                            if dist < closest_dist:
                                closest_dist = dist
                                target_hand_landmarks = temp_landmarks

            # Use the detected hand
            if target_hand_landmarks:
                lm = target_hand_landmarks.landmark[8]
                wx, wy = int(lm.x * w), int(lm.y * h)
                finger_detected = True
                self.smooth_points.append((wx, wy))
                if len(self.smooth_points) > 5: self.smooth_points.pop(0)
                
                # Draw red crosshair (more evident)
                if assets.images['crosshair'] is not None:
                    crosshair = assets.images['crosshair'].copy()
                    # Tint crosshair bright red
                    for i in range(crosshair.shape[0]):
                        for j in range(crosshair.shape[1]):
                            if crosshair[i, j, 3] > 0:  # If pixel is not transparent
                                crosshair[i, j, 0] = 0      # Blue = 0
                                crosshair[i, j, 1] = 0      # Green = 0
                                crosshair[i, j, 2] = 255    # Red = 255
                    frame = overlay_image(frame, crosshair, wx-30, wy-30)
                    # Add extra red glow
                    cv2.circle(frame, (wx, wy), 35, (0, 0, 255), 2)
                else:
                    # Fallback: draw large red circle with glow effect
                    cv2.circle(frame, (wx, wy), 35, (0, 0, 255), 3)
                    cv2.circle(frame, (wx, wy), 30, (0, 100, 255), 2)
                    cv2.circle(frame, (wx, wy), 8, (0, 0, 255), -1)

            if finger_detected and len(self.smooth_points) > 0:
                avg_x = int(sum(p[0] for p in self.smooth_points) / len(self.smooth_points))
                avg_y = int(sum(p[1] for p in self.smooth_points) / len(self.smooth_points))
                dx = avg_x - self.snake_pos[0]
                dy = avg_y - self.snake_pos[1]

                target_angle = math.degrees(math.atan2(dy, dx)) + 90
                angle_diff = abs(target_angle - self.last_angle)
                if angle_diff > 180: angle_diff = 360 - angle_diff

                if angle_diff < 150 or len(self.snake_trail) < 5:
                    self.angle = target_angle
                    self.last_angle = self.angle
                    self.snake_pos = (self.snake_pos[0] + dx * self.speed_factor, 
                                    self.snake_pos[1] + dy * self.speed_factor)
                else:
                    rad = math.radians(self.last_angle - 90)
                    move_step = 10 * self.speed_factor
                    self.snake_pos = (self.snake_pos[0] + math.cos(rad) * move_step, 
                                    self.snake_pos[1] + math.sin(rad) * move_step)

            x, y = int(self.snake_pos[0]), int(self.snake_pos[1])

            # MODIFIED COLLISION DETECTION
            hit = False
            correction = 90
            if self.calib_mode == "sitting":
                correction = 80
            
            # Wall collision only applies to HARD mode
            if self.difficulty == "hard":
                if (x < bound_min_x + correction or x > bound_max_x - correction or 
                    y < bound_min_y + correction or y > bound_max_y - correction):
                    hit = True

            # Self-collision check (hard mode only)
            if self.difficulty == "hard" and not hit and not self.invincible and len(self.snake_trail) > 20:
                head_pt = np.array([x, y])
                for i in range(0, len(self.snake_trail) - 15, 2):
                    if np.linalg.norm(head_pt - np.array(self.snake_trail[i])) < 20:
                        hit = True
                        break

            if hit and not self.invincible:
                self.lives -= 1
                if assets.sounds['hit']: assets.sounds['hit'].play()
                self.snake_pos = ((bound_min_x + bound_max_x)//2, (bound_min_y + bound_max_y)//2)
                self.snake_trail.clear()
                self.smooth_points.clear()
                self.max_trail = 2
                self.invincible, self.inv_time = True, current_time
                self.last_angle = 0

            x = max(bound_min_x, min(bound_max_x, x))
            y = max(bound_min_y, min(bound_max_y, y))
            self.snake_pos = (x, y)
            self.snake_trail.append(self.snake_pos)
            if len(self.snake_trail) > self.max_trail: self.snake_trail.pop(0)

            if math.hypot(x - self.food[0], y - self.food[1]) < 55:
                if assets.sounds['eat']: assets.sounds['eat'].play()
                self.score += 1
                self.max_trail += 2
                self.food = spawn_food(self.difficulty, self.snake_trail, w, h, self.bounds, self.calib_mode)

            # DRAW GAME CONTENT
            box_w = bound_max_x - bound_min_x
            box_h = bound_max_y - bound_min_y

            if assets.images.get('border') is not None:
                if (self.cached_border is None or 
                    self.cached_border.shape[1] != box_w or 
                    self.cached_border.shape[0] != box_h):
                    self.cached_border = cv2.resize(assets.images['border'], (box_w, box_h))
                frame = overlay_image(frame, self.cached_border, bound_min_x, bound_min_y)
            else:
                cv2.rectangle(frame, (bound_min_x-5, bound_min_y-5), (bound_max_x+5, bound_max_y+5), (0, 165, 255), 2)
                cv2.rectangle(frame, (bound_min_x, bound_min_y), (bound_max_x, bound_max_y), (0, 255, 255), 4)

            if assets.images['food'] is not None:
                frame = overlay_image(frame, assets.images['food'], self.food[0]-32, self.food[1]-32)
            else:
                cv2.circle(frame, self.food, 25, (0, 0, 255), -1)

            if len(self.snake_trail) > 1:
                for i in range(0, len(self.snake_trail), 3):
                    pt = self.snake_trail[i]
                    bx, by = int(pt[0]), int(pt[1])
                    if assets.images['ball'] is not None:
                        frame = overlay_image(frame, assets.images['ball'], bx - 25, by - 25)
                    else:
                        cv2.circle(frame, (bx, by), 25, (0, 255, 0), -1)

            if assets.images['head'] is not None:
                rotated_head = rotate_image(assets.images['head'], -self.angle)
                frame = overlay_image(frame, rotated_head, x-40, y-40)
            else:
                cv2.circle(frame, (x, y), 35, (0, 215, 255), -1)

            # DRAW HUD
            draw_shadow_text(frame, f"SCORE: {self.score:03d}", (40, 70), 1.4, (255, 255, 100), 3)

            time_color = (0, 255, 255) if (self.duration - self.elapsed) > 10 else (255, 80, 80)
            time_str = f"TIME {self.elapsed // 60:02d}:{self.elapsed % 60:02d}"
            draw_shadow_text(frame, time_str, (w//2 - 150, 70), 1.4, time_color, 3)

            # Lives (hard mode only)
            if self.difficulty == "hard":
                heart_y = 40
                heart_start_x = w - 350
                for i in range(3):
                    h_img = assets.images['heart_full'] if i < self.lives else assets.images['heart_empty']
                    if h_img is not None:
                        frame = overlay_image(frame, h_img, heart_start_x + i*70, heart_y)
                    else:
                        color = (0, 255, 0) if i < self.lives else (100, 100, 100)
                        cv2.circle(frame, (heart_start_x + i*70 + 35, heart_y + 35), 30, color, -1)

        return frame

# =====================================================
# VIDEO THREAD
# =====================================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    calibration_complete = pyqtSignal(tuple)
    game_over = pyqtSignal()

    def __init__(self, logic):
        super().__init__()
        self.logic = logic
        self.running = True

    def run(self):
        source = 0
        test_cap = cv2.VideoCapture(1)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                source = 1
            test_cap.release()
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.015)
                continue
            if ret:
                processed_frame = self.logic.process_frame(frame)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.change_pixmap_signal.emit(qt_img)

                if self.logic.calibration_done:
                    self.calibration_complete.emit(self.logic.bounds)
                    self.logic.calibration_done = False

                if self.logic.game_over_flag:
                    self.game_over.emit()
                    self.logic.game_over_flag = False

            time.sleep(0.006)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# =====================================================
# UI MAIN WINDOW
# =====================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AR - Jungle Adventure")
        self.setGeometry(100, 100, 1280, 720)

        # UI Configuration (same as before)
        self.ui_config = {
            'start_title_img':  {'x': 710, 'y': 200,  'w': 500, 'h': 400, 'center_x': False}, 
            'start_play':       {'x': 800, 'y': 630, 'w': 300, 'h': 70,  'center_x': False, 'font': 22},
            'start_exit':       {'x': 800, 'y': 730, 'w': 300, 'h': 70,  'center_x': False, 'font': 22},

            'mode_title_img':   {'x': 730, 'y': 270,  'w': 500, 'h': 200, 'center_x': False},
            'mode_stand':       {'x': 800, 'y': 500, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'mode_sit':         {'x': 800, 'y': 600, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'mode_back':        {'x': 870, 'y': 720, 'w': 200, 'h': 60,  'center_x': False, 'font': 22},

            'hand_title_img':   {'x': 650, 'y': 240,  'w': 700, 'h': 300, 'center_x': False},
            'hand_left':        {'x': 800, 'y': 500, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'hand_right':       {'x': 800, 'y': 600, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'hand_back':        {'x': 870, 'y': 720, 'w': 200, 'h': 60,  'center_x': False, 'font': 22},

            # Difficulty - only shows for STANDING mode
            'diff_title_img':   {'x': 730, 'y': 270,  'w': 450, 'h': 200, 'center_x': False},
            'diff_easy':        {'x': 800, 'y': 500, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_med':         {'x': 800, 'y': 600, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_hard':        {'x': 800, 'y': 700, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_back':        {'x': 870, 'y': 820, 'w': 200, 'h': 60,  'center_x': False, 'font': 22},

            # Duration - changed to 20 min max
            'dur_title_img':    {'x': 750, 'y': 270,  'w': 500, 'h': 200, 'center_x': False},
            'dur_minus':        {'x': 785, 'y': 500, 'w': 80,  'h': 60,  'center_x': False, 'font': 22},
            'dur_label':        {'x': 875, 'y': 500, 'w': 200, 'h': 60,  'center_x': False, 'font': 40},
            'dur_plus':         {'x': 1085, 'y': 500, 'w': 80,  'h': 60,  'center_x': False, 'font': 22},
            'dur_next':         {'x': 800, 'y': 600, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'dur_back':         {'x': 870, 'y': 730, 'w': 200, 'h': 60,  'center_x': False, 'font': 22},

            'inst_background':  {'x': 0,   'y': 0,    'w': 1280, 'h': 720, 'center_x': False},
            'inst_back':        {'x': 860,  'y': 870, 'w': 220, 'h': 70,  'center_x': False, 'font': 22},
            'inst_go':          {'x': 800,  'y': 770, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},

            'inst_title_img':   {'x': 725, 'y': 150,  'w': 450, 'h': 250, 'center_x': False},
            'inst_card':        {'x': 580, 'y': 370, 'w': 800, 'h': 350, 'center_x': False},

            # Game quit button - made more responsive
            'game_quit':        {'x': 1680, 'y': 45,  'w': 180, 'h': 60,  'center_x': False, 'font': 20},

            'over_card':        {'x': 300, 'y': 270, 'w': 500, 'h': 500, 'center_x': True},
            
            # Calibration screen buttons
            'calib_back':       {'x': 20, 'y': 20, 'w': 120, 'h': 50, 'center_x': False, 'font': 18},
            'calib_recalib':    {'x': 160, 'y': 20, 'w': 140, 'h': 50, 'center_x': False, 'font': 18},
            'calib_proceed':    {'x': 1080, 'y': 650, 'w': 160, 'h': 50, 'center_x': False, 'font': 18},
        }

        bg_path = resource_path("assets/bg.jpg")
        self.bg_pixmap = None
        if os.path.exists(bg_path):
            self.bg_pixmap = QImage(bg_path)
            self.update_background()
        else:
            self.setStyleSheet("QMainWindow { background-color: #1a1a2e; }")

        btn_wood_path = resource_path("assets/btn_wood.png").replace("\\", "/")
        
        self.setStyleSheet(self.styleSheet() + f"""
            QPushButton {{
                background-image: url({btn_wood_path});
                background-repeat: no-repeat;
                background-position: center;
                background-size: cover;
                color: #654321;
                border: 3px solid #DAA520;
                border-radius: 10px;
                font-family: 'Verdana','Noto Color Emoji', 'Segoe UI Emoji';
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover {{
                background: #CD853F;
                border: 3px solid #FFD700;
                color: white;
            }}
            QPushButton:pressed {{
                background: #8B4513;
                border: 3px solid #FF8C00;
            }}
            QFrame#GameOverCard {{
                background-color: rgba(0, 0, 0, 180);
                border: 4px solid #FFD700;
                border-radius: 20px;
            }}
        """)

        self.difficulty = "medium"
        self.duration = 120  # Default 2 minutes
        self.calibration_mode = "standing"
        self.selected_hand = "Right"
        self.calibrated_bounds = (50, 1230, 100, 670)
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.loader = LoadingOverlay(self)
        self.loader.resize(1280, 720)
        self.loader.hide()

        # Initialize screens
        self.init_start_screen() 
        self.init_mode_screen() 
        self.init_hand_screen() 
        self.init_calibration_screen() 
        self.init_difficulty_screen() 
        self.init_duration_screen() 
        self.init_instruction_screen() 
        self.init_game_screen() 
        self.init_gameover_screen() 
        
        self.snake_logic = SnakeLogic()
        self.thread = VideoThread(self.snake_logic)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.calibration_complete.connect(self.on_calibration_done)
        self.thread.game_over.connect(self.game_over)
        self.thread.start()

    def apply_config(self, widget, key):
        cfg = self.ui_config.get(key, {})
        w = cfg.get('w', 200)
        h = cfg.get('h', 50)
        widget.resize(w, h)
        x = cfg.get('x', 0)
        y = cfg.get('y', 0)
        if cfg.get('center_x', False):
            x = (1280 - w) // 2 + x
        widget.move(x, y)
        if 'font' in cfg:
            current_style = widget.styleSheet()
            widget.setStyleSheet(current_style + f" font-size: {cfg['font']}px;")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            # Only allow ESC to close if not in calibration or game mode
            if self.central_widget.currentIndex() not in [3, 7]:
                self.close()

    def resizeEvent(self, event):
        self.update_background()
        self.loader.resize(self.size())
        super().resizeEvent(event)

    def update_background(self):
        if self.bg_pixmap is not None:
            sImage = self.bg_pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            palette = QPalette()
            palette.setBrush(QPalette.Window, QBrush(sImage))
            self.setPalette(palette)

    def get_title_widget(self, image_name, fallback_text):
        lbl = QLabel(parent=None) 
        path = resource_path(image_name)
        if os.path.exists(path):
            pix = QPixmap(path)
            lbl.setPixmap(pix)
            lbl.setScaledContents(True)
        else:
            lbl.setText(fallback_text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-family: 'Verdana'; font-weight: 900; color: #FFD700; font-size: 60px;")
        return lbl

    def play_sound_and_switch(self, index):
        if assets.sounds['click']:
            assets.sounds['click'].play()
        self.central_widget.setCurrentIndex(index)

    def select_hand(self, hand):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.selected_hand = hand
        self.central_widget.setCurrentIndex(4)  # Always go to difficulty screen

    # SCREENS
    def init_start_screen(self):
        page = QWidget()
        title = self.get_title_widget("assets/download.png", "SNAKE AR")
        title.setParent(page)
        self.apply_config(title, 'start_title_img')

        btn_start = QPushButton("PLAY GAME", page)
        self.apply_config(btn_start, 'start_play')
        btn_start.clicked.connect(lambda: self.play_sound_and_switch(1))

        btn_exit = QPushButton("EXIT", page)
        self.apply_config(btn_exit, 'start_exit')
        btn_exit.clicked.connect(self.close)

        self.central_widget.addWidget(page)

    def init_mode_screen(self):
        page = QWidget()
        title = self.get_title_widget("assets/MODE.png", "SELECT MODE")
        title.setParent(page)
        self.apply_config(title, 'mode_title_img')
        
        btn_stand = QPushButton("STANDING", page)
        self.apply_config(btn_stand, 'mode_stand')
        btn_stand.clicked.connect(lambda: self.set_mode_and_next("standing"))
        
        btn_sit = QPushButton("SITTING", page)
        self.apply_config(btn_sit, 'mode_sit')
        btn_sit.clicked.connect(lambda: self.set_mode_and_next("sitting"))
        
        btn_back = QPushButton("BACK", page)
        self.apply_config(btn_back, 'mode_back')
        btn_back.clicked.connect(lambda: self.play_sound_and_switch(0))
        
        self.central_widget.addWidget(page)

    def set_mode_and_next(self, mode):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.calibration_mode = mode
        self.central_widget.setCurrentIndex(2)

    def init_hand_screen(self):
        page = QWidget()
        title = self.get_title_widget("assets/hand (2).png", "SELECT HAND")
        title.setParent(page)
        self.apply_config(title, 'hand_title_img')
        
        btn_left = QPushButton("LEFT HAND", page)
        self.apply_config(btn_left, 'hand_left')
        btn_left.clicked.connect(lambda: self.select_hand("Left"))
        
        btn_right = QPushButton("RIGHT HAND", page)
        self.apply_config(btn_right, 'hand_right')
        btn_right.clicked.connect(lambda: self.select_hand("Right"))
        
        btn_back = QPushButton("BACK", page)
        self.apply_config(btn_back, 'hand_back')
        btn_back.clicked.connect(lambda: self.play_sound_and_switch(1))
        
        self.central_widget.addWidget(page)

    def init_calibration_screen(self):
        page = QWidget()
        
        # Main layout with video label
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.calib_label = QLabel()
        self.calib_label.setAlignment(Qt.AlignCenter)
        self.calib_label.setScaledContents(True)
        self.calib_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.calib_label, stretch=1)

        # Button overlay layout
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left side buttons
        left_button_layout = QHBoxLayout()
        btn_back = QPushButton("← BACK")
        btn_back.setStyleSheet("""
            QPushButton {
                background-color: rgba(139, 69, 19, 200);
                color: white;
                border: 2px solid #FFD700;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(160, 82, 45, 220);
            }
        """)
        btn_back.clicked.connect(self.calibration_back)
        left_button_layout.addWidget(btn_back)
        
        btn_recalibrate = QPushButton("↻ RESET")
        btn_recalibrate.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 140, 0, 200);
                color: white;
                border: 2px solid #FFD700;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 165, 0, 220);
            }
        """)
        btn_recalibrate.clicked.connect(self.recalibrate)
        left_button_layout.addWidget(btn_recalibrate)
        
        button_layout.addLayout(left_button_layout)
        button_layout.addStretch()
        
        # Right side - Proceed button
        btn_proceed = QPushButton("PROCEED →")
        btn_proceed.setStyleSheet("""
            QPushButton {
                background-color: rgba(34, 139, 34, 200);
                color: white;
                border: 3px solid #FFD700;
                font-weight: bold;
                font-size: 20px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: rgba(50, 205, 50, 220);
            }
            QPushButton:disabled {
                background-color: rgba(100, 100, 100, 150);
                border: 3px solid #AAAAAA;
                color: #CCCCCC;
            }
        """)
        btn_proceed.clicked.connect(self.force_proceed)
        btn_proceed.setEnabled(False)  # Initially disabled
        button_layout.addWidget(btn_proceed)
        
        main_layout.addLayout(button_layout)
        page.setLayout(main_layout)
        
        # Store references
        self.calib_back_btn = btn_back
        self.calib_recalib_btn = btn_recalibrate
        self.calib_proceed_btn = btn_proceed
        
        self.central_widget.addWidget(page)

    def calibration_back(self):
        """Go back from calibration screen"""
        if assets.sounds['click']: assets.sounds['click'].play()
        self.snake_logic.mode = 'idle'
        self.snake_logic.reset_calibration()
        self.calib_proceed_btn.setEnabled(False)
        self.central_widget.setCurrentIndex(6)  # Go back to instruction screen

    def recalibrate(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.snake_logic.reset_calibration()
        self.calib_proceed_btn.setEnabled(False)

    def force_proceed(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.snake_logic.force_complete()

    def init_difficulty_screen(self):
        page = QWidget()
        title = self.get_title_widget("assets/difficulty.png", "DIFFICULTY")
        title.setParent(page)
        self.apply_config(title, 'diff_title_img')

        btn_easy = QPushButton("EASY", page)
        self.apply_config(btn_easy, 'diff_easy')
        btn_easy.clicked.connect(lambda: self.select_difficulty("easy"))

        btn_med = QPushButton("MEDIUM", page)
        self.apply_config(btn_med, 'diff_med')
        btn_med.clicked.connect(lambda: self.select_difficulty("medium"))

        btn_hard = QPushButton("HARD", page)
        self.apply_config(btn_hard, 'diff_hard')
        btn_hard.clicked.connect(lambda: self.select_difficulty("hard"))
            
        btn_back = QPushButton("BACK", page)
        self.apply_config(btn_back, 'diff_back')
        btn_back.clicked.connect(lambda: self.play_sound_and_switch(2))
        
        self.central_widget.addWidget(page)

    def init_duration_screen(self):
        page = QWidget()
        title = self.get_title_widget("assets/duration.png", "DURATION")
        title.setParent(page)
        self.apply_config(title, 'dur_title_img')
        
        b_minus = QPushButton("➖", page)
        self.apply_config(b_minus, 'dur_minus')
        b_minus.clicked.connect(lambda: self.change_time(-60))
        
        self.time_label = QLabel("2 MIN", page)
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: #654321; font-weight: bold;")
        self.apply_config(self.time_label, 'dur_label')
        
        b_plus = QPushButton("➕", page)
        self.apply_config(b_plus, 'dur_plus')
        b_plus.clicked.connect(lambda: self.change_time(60))
        
        btn_next = QPushButton("NEXT", page)
        self.apply_config(btn_next, 'dur_next')
        btn_next.clicked.connect(lambda: self.play_sound_and_switch(6))
        
        btn_back = QPushButton("BACK", page)
        self.apply_config(btn_back, 'dur_back')
        btn_back.clicked.connect(self.duration_back)
        
        self.central_widget.addWidget(page)
    
    def duration_back(self):
        """Go back from duration to difficulty screen"""
        if assets.sounds['click']: assets.sounds['click'].play()
        self.central_widget.setCurrentIndex(4)  # Difficulty (both modes)

    def update_instruction_background_size(self, event=None):
        current_idx = self.central_widget.currentIndex()
        if current_idx == 6:
            widget = self.central_widget.currentWidget()
            if hasattr(self, 'inst_bg_label'):
                self.inst_bg_label.setGeometry(0, 0, widget.width(), widget.height())

    def init_instruction_screen(self):
        page = QWidget()

        self.inst_bg_label = QLabel(page)
        self.inst_bg_label.setAlignment(Qt.AlignCenter)
        self.inst_bg_label.setScaledContents(True)

        inst_path = resource_path("assets/instruct.png")
        if os.path.exists(inst_path):
            pix = QPixmap(inst_path)
            self.inst_bg_label.setPixmap(pix)
        else:
            self.inst_bg_label.setStyleSheet("background-color: #1e3a2f; color: white;")
            self.inst_bg_label.setText("Instructions Image Missing")
            self.inst_bg_label.setFont(QFont("Verdana", 40, QFont.Bold))

        self.inst_bg_label.setGeometry(0, 0, page.width(), page.height())

        btn_back = QPushButton("BACK", page)
        self.apply_config(btn_back, 'inst_back')
        btn_back.clicked.connect(lambda: self.play_sound_and_switch(5))
        btn_back.raise_()

        btn_go = QPushButton("GO!", page)
        self.apply_config(btn_go, 'inst_go')
        btn_go.clicked.connect(self.trigger_game_load)
        btn_go.raise_()

        btn_back.setStyleSheet(btn_back.styleSheet() + " background-color: rgba(139,69,19,220); color: white; border: 3px solid #8B4513;")
        btn_go.setStyleSheet(btn_go.styleSheet() + " background-color: rgba(34,139,34,220); color: white; font-weight: bold; border: 3px solid #228B22;")

        self.inst_bg_label.lower()
        self.central_widget.addWidget(page)
        page.resizeEvent = self.update_instruction_background_size

    def init_game_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.game_label = QLabel()
        self.game_label.setScaledContents(True)
        self.game_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.game_label)
        page.setLayout(layout)

        # Improved quit button - more responsive
        self.quit_btn = QPushButton("✕ QUIT", page)
        self.apply_config(self.quit_btn, 'game_quit')
        self.quit_btn.clicked.connect(self.quit_game)
        self.quit_btn.raise_()
        self.quit_btn.setStyleSheet(self.quit_btn.styleSheet() + """
            background-color: rgba(255, 0, 0, 220); 
            color: white; 
            font-weight: bold; 
            border: 3px solid #FFD700;
            font-size: 22px;
        """)

        self.central_widget.addWidget(page)

    def quit_game(self):
        if assets.sounds['click']:
            assets.sounds['click'].play()
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        self.snake_logic.mode = 'idle'
        self.final_score.setText(str(self.snake_logic.score))
        QTimer.singleShot(100, lambda: self.central_widget.setCurrentIndex(8))

    def init_gameover_screen(self):
        page = QWidget()
        card = QFrame(page)
        card.setObjectName("GameOverCard")
        self.apply_config(card, 'over_card')
        
        card_layout = QVBoxLayout(card)
        card_layout.setAlignment(Qt.AlignCenter)
        card_layout.setSpacing(20)
        
        lbl_over = QLabel("GAME OVER")
        lbl_over.setStyleSheet("color: white; font-weight: bold; font-size: 32px;")
        
        lbl_text = QLabel("FINAL SCORE")
        lbl_text.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 24px; letter-spacing: 2px;")
        
        self.final_score = QLabel("0")
        self.final_score.setStyleSheet("color: #FFFFFF; font-weight: 900; font-size: 90px;")
        
        btn_restart = QPushButton("PLAY AGAIN")
        btn_restart.setFixedSize(250, 60)
        btn_restart.clicked.connect(self.restart)
        
        btn_quit = QPushButton("QUIT")
        btn_quit.setFixedSize(250, 60)
        btn_quit.clicked.connect(self.close)

        lbl_text2 = QLabel(" ")
        lbl_text2.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 24px; letter-spacing: 2px;")

        card_layout.addWidget(lbl_over, 0, Qt.AlignCenter)
        card_layout.addWidget(lbl_text, 0, Qt.AlignCenter)
        card_layout.addWidget(self.final_score, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_restart, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_quit, 0, Qt.AlignCenter)
        card_layout.addWidget(lbl_text2, 0, Qt.AlignCenter)
        
        self.central_widget.addWidget(page)

    def update_image(self, qt_img):
        idx = self.central_widget.currentIndex()
        if idx == 3:
            self.calib_label.setPixmap(QPixmap.fromImage(qt_img))
            # Check if we can enable proceed button - both phases complete
            if self.snake_logic.body_calibration_done and self.snake_logic.hold_timer >= self.snake_logic.hold_time:
                self.calib_proceed_btn.setEnabled(True)
                self.calib_proceed_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(34, 139, 34, 200);
                        color: white;
                        border: 3px solid #FFD700;
                        font-weight: bold;
                        font-size: 20px;
                        padding: 10px 20px;
                    }
                    QPushButton:hover {
                        background-color: rgba(50, 205, 50, 220);
                    }
                """)
            else:
                self.calib_proceed_btn.setEnabled(False)
                self.calib_proceed_btn.setStyleSheet("""
                    QPushButton {
                        background-color: rgba(100, 100, 100, 150);
                        color: #CCCCCC;
                        border: 3px solid #AAAAAA;
                        font-weight: bold;
                        font-size: 20px;
                        padding: 10px 20px;
                    }
                """)
        elif idx == 7:
            self.game_label.setPixmap(QPixmap.fromImage(qt_img))

    def select_difficulty(self, diff):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.difficulty = diff
        self.central_widget.setCurrentIndex(5)

    def change_time(self, delta):
        if assets.sounds['click']: assets.sounds['click'].play()
        # Changed max time to 20 minutes (1200 seconds)
        self.duration = max(60, min(1200, self.duration + delta))
        mins = self.duration // 60
        secs = self.duration % 60
        self.time_label.setText(f"{mins:02d}:{secs:02d}")

    def trigger_game_load(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.snake_logic.start_calibration(self.calibration_mode)
        self.calib_proceed_btn.setEnabled(False)
        self.central_widget.setCurrentIndex(3)

    def on_calibration_done(self, bounds):
        self.calibrated_bounds = bounds
        self.snake_logic.mode = 'idle'
        
        self.loader.show()
        QTimer.singleShot(2200, lambda: self._delayed_game_start(bounds))

    def _delayed_game_start(self, bounds):
        self.loader.hide()
        self.snake_logic.start_game(self.difficulty, self.duration, self.selected_hand)
        
        self.snake_logic.invincible = True
        self.snake_logic.inv_time = time.time()
        
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
        bg_music = resource_path("assets/hits.mp3")
        if os.path.exists(bg_music): pygame.mixer.music.play(-1)
        
        self.central_widget.setCurrentIndex(7)

    def game_over(self):
        if assets.sounds['game_over']: assets.sounds['game_over'].play()
        if pygame.mixer.music.get_busy(): pygame.mixer.music.stop()
        self.final_score.setText(str(self.snake_logic.score))
        QTimer.singleShot(100, lambda: self.central_widget.setCurrentIndex(8))

    def restart(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.central_widget.setCurrentIndex(0)

    # def closeEvent(self, event):
    #     self.thread.stop()
    #     pygame.mixer.quit()
    #     event.accept()
    def closeEvent(self, event):
    # Add this line as the very first thing
        self.snake_logic.mode = 'idle'                 # ← prevents MediaPipe from running more processing

        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.running = False                 # ← use the existing flag
            # Optional: give it a tiny bit more time to notice
            self.thread.wait(800)                       # 800 ms instead of default (which can be longer)
        
        # Stop sounds early
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except:
            pass

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash, _ = show_splash()
    app.setStyle('Fusion')
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    app.processEvents()
    
    import time
    time.sleep(1.5)
    window = MainWindow()
    
    for i in range(10):
        time.sleep(0.1)
        app.processEvents()
        if splash:
            splash.update()
    
    def show_main():
        if splash:
            splash.close()
        window.showFullScreen()
        window.update_instruction_background_size()
    
    QTimer.singleShot(500, show_main)
    sys.exit(app.exec_())