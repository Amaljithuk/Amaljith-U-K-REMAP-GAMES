import sys
import os
import ctypes
from pathlib import Path
import pygame
import numpy as np
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd

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
import random
import math
import time

import mediapipe as mp
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QFrame, QSizePolicy, QGraphicsOpacityEffect, 
                             QSplashScreen, QScrollArea, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect, QEasingCurve, QPointF, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QLinearGradient, QGradient, QPainter, QPen, QRadialGradient, QConicalGradient

# Initialize Audio (don't crash if audio device is unavailable)
try:
    pygame.mixer.init()
except Exception as e:
    print(f"Audio init failed: {e}")

# =====================================================
# MOVEMENT ANALYTICS CLASS (INTEGRATED)
# =====================================================
class MovementAnalytics:
    """Tracks and analyzes body movements during gameplay"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all analytics data"""
        self.start_time = None
        self.end_time = None
        self.hand_positions = []
        self.wrist_positions = {'left': [], 'right': []}
        self.shoulder_positions = {'left': [], 'right': []}
        self.elbow_positions = {'left': [], 'right': []}
        self.body_center_positions = []
        self.movement_velocities = []
        self.total_distance = 0
        self.max_reach_left = 0
        self.max_reach_right = 0
        self.max_reach_up = 0
        self.max_reach_down = 0
        self.frame_count = 0
        self.collision_events = []
        self.food_collection_events = []
        self.active_hand = None
        self.calibration_mode = None
        self.difficulty = None
        self.duration_target = 0
        self.final_score = 0
        self.calories_burned = 0
        
    def start_session(self, mode, difficulty, duration, hand):
        """Start a new analytics session"""
        self.reset()
        self.start_time = datetime.now()
        self.calibration_mode = mode
        self.difficulty = difficulty
        self.duration_target = duration
        self.active_hand = hand
    
    def record_frame(self, hand_pos, pose_landmarks, frame_width, frame_height):
        """Record movement data for a single frame"""
        self.frame_count += 1
        
        if hand_pos:
            self.hand_positions.append({
                'x': hand_pos[0],
                'y': hand_pos[1],
                'timestamp': time.time()
            })
            
            # Calculate velocity if we have previous position
            if len(self.hand_positions) > 1:
                prev = self.hand_positions[-2]
                curr = self.hand_positions[-1]
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                dt = curr['timestamp'] - prev['timestamp']
                if dt > 0:
                    velocity = math.sqrt(dx*dx + dy*dy) / dt
                    self.movement_velocities.append(velocity)
                    self.total_distance += math.sqrt(dx*dx + dy*dy)
        
        # Record pose data
        if pose_landmarks:
            mp_pose = mp.solutions.pose
            
            # Left wrist
            lw = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            self.wrist_positions['left'].append({
                'x': int(lw.x * frame_width),
                'y': int(lw.y * frame_height),
                'timestamp': time.time()
            })
            
            # Right wrist
            rw = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            self.wrist_positions['right'].append({
                'x': int(rw.x * frame_width),
                'y': int(rw.y * frame_height),
                'timestamp': time.time()
            })
            
            # Shoulders
            ls = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            self.shoulder_positions['left'].append({
                'x': int(ls.x * frame_width),
                'y': int(ls.y * frame_height)
            })
            self.shoulder_positions['right'].append({
                'x': int(rs.x * frame_width),
                'y': int(rs.y * frame_height)
            })
            
            # Elbows
            le = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
            re = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            self.elbow_positions['left'].append({
                'x': int(le.x * frame_width),
                'y': int(le.y * frame_height)
            })
            self.elbow_positions['right'].append({
                'x': int(re.x * frame_width),
                'y': int(re.y * frame_height)
            })
            
            # Body center (midpoint between shoulders)
            center_x = int((ls.x + rs.x) * frame_width / 2)
            center_y = int((ls.y + rs.y) * frame_height / 2)
            self.body_center_positions.append({
                'x': center_x,
                'y': center_y,
                'timestamp': time.time()
            })
            
            # Update reach extremes
            if hand_pos:
                if hand_pos[0] < frame_width / 2:
                    self.max_reach_left = max(self.max_reach_left, 
                                             abs(hand_pos[0] - center_x))
                else:
                    self.max_reach_right = max(self.max_reach_right, 
                                              abs(hand_pos[0] - center_x))
                
                if hand_pos[1] < frame_height / 2:
                    self.max_reach_up = max(self.max_reach_up, 
                                           abs(hand_pos[1] - center_y))
                else:
                    self.max_reach_down = max(self.max_reach_down, 
                                             abs(hand_pos[1] - center_y))
    
    def record_collision(self, collision_type, position):
        """Record a collision event"""
        self.collision_events.append({
            'type': collision_type,  # 'wall' or 'self'
            'position': position,
            'timestamp': time.time()
        })
    
    def record_food_collection(self, position):
        """Record a food collection event"""
        self.food_collection_events.append({
            'position': position,
            'timestamp': time.time()
        })
    
    def end_session(self, final_score):
        """End the analytics session"""
        self.end_time = datetime.now()
        self.final_score = final_score
        self._calculate_calories()
    
    def _calculate_calories(self):
        """Estimate calories burned based on movement"""
        if not self.start_time or not self.end_time:
            return
        
        duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        avg_velocity = np.mean(self.movement_velocities) if self.movement_velocities else 0
        
        # Simple calorie estimation (very rough approximation)
        base_calories_per_minute = 3  # Light activity
        
        if avg_velocity > 200:
            intensity_multiplier = 2.5  # High intensity
        elif avg_velocity > 100:
            intensity_multiplier = 1.8  # Moderate intensity
        else:
            intensity_multiplier = 1.2  # Low intensity
        
        self.calories_burned = duration_minutes * base_calories_per_minute * intensity_multiplier
    
    def generate_report(self):
        """Generate a comprehensive movement analysis report"""
        if not self.start_time or not self.end_time:
            return None
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate statistics
        avg_velocity = np.mean(self.movement_velocities) if self.movement_velocities else 0
        max_velocity = max(self.movement_velocities) if self.movement_velocities else 0
        
        # Calculate arm extension metrics
        arm_extensions = []
        for left_pos, right_pos in zip(self.wrist_positions['left'], 
                                       self.wrist_positions['right']):
            extension = abs(left_pos['x'] - right_pos['x'])
            arm_extensions.append(extension)
        
        avg_arm_extension = np.mean(arm_extensions) if arm_extensions else 0
        max_arm_extension = max(arm_extensions) if arm_extensions else 0
        
        # Calculate active range of motion
        total_reach = self.max_reach_left + self.max_reach_right
        vertical_reach = self.max_reach_up + self.max_reach_down
        
        # Body stability (lower is more stable)
        body_movements = []
        for i in range(1, len(self.body_center_positions)):
            prev = self.body_center_positions[i-1]
            curr = self.body_center_positions[i]
            movement = math.sqrt((curr['x'] - prev['x'])**2 + 
                               (curr['y'] - prev['y'])**2)
            body_movements.append(movement)
        
        body_stability_score = 100 - min(100, np.mean(body_movements) if body_movements else 0)
        
        # Reaction time (time between food spawns)
        reaction_times = []
        for i in range(1, len(self.food_collection_events)):
            prev_time = self.food_collection_events[i-1]['timestamp']
            curr_time = self.food_collection_events[i]['timestamp']
            reaction_times.append(curr_time - prev_time)
        
        avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
        
        # Movement efficiency (score per distance)
        movement_efficiency = (self.final_score / self.total_distance * 100) if self.total_distance > 0 else 0
        
        # Calculate movement consistency
        velocity_std = np.std(self.movement_velocities) if self.movement_velocities else 0
        movement_consistency = max(0, 100 - (velocity_std / (avg_velocity + 1) * 100))
        
        report = {
            'session_info': {
                'date': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_played': f"{int(duration // 60)}:{int(duration % 60):02d}",
                'duration_target': f"{int(self.duration_target // 60)}:{int(self.duration_target % 60):02d}",
                'mode': self.calibration_mode,
                'difficulty': self.difficulty,
                'active_hand': self.active_hand,
                'final_score': self.final_score,
                'calories_burned': f"{self.calories_burned:.1f} kcal"
            },
            'movement_metrics': {
                'total_distance': f"{self.total_distance:.1f} pixels",
                'avg_velocity': f"{avg_velocity:.1f} px/s",
                'max_velocity': f"{max_velocity:.1f} px/s",
                'movement_efficiency': f"{movement_efficiency:.2f}",
                'movement_consistency': f"{movement_consistency:.1f}%"
            },
            'range_of_motion': {
                'horizontal_reach': f"{total_reach:.1f} pixels",
                'max_left_reach': f"{self.max_reach_left:.1f} pixels",
                'max_right_reach': f"{self.max_reach_right:.1f} pixels",
                'vertical_reach': f"{vertical_reach:.1f} pixels",
                'max_up_reach': f"{self.max_reach_up:.1f} pixels",
                'max_down_reach': f"{self.max_reach_down:.1f} pixels"
            },
            'arm_metrics': {
                'avg_arm_extension': f"{avg_arm_extension:.1f} pixels",
                'max_arm_extension': f"{max_arm_extension:.1f} pixels"
            },
            'performance_metrics': {
                'body_stability_score': f"{body_stability_score:.1f}/100",
                'avg_reaction_time': f"{avg_reaction_time:.2f} seconds",
                'foods_collected': len(self.food_collection_events),
                'total_collisions': len(self.collision_events),
                'wall_collisions': sum(1 for e in self.collision_events if e['type'] == 'wall'),
                'self_collisions': sum(1 for e in self.collision_events if e['type'] == 'self'),
                'collision_rate': f"{(len(self.collision_events) / (duration / 60)):.1f} per minute" if duration > 0 else "0.0 per minute"
            },
            'health_insights': self._generate_health_insights(
                duration, avg_velocity, body_stability_score, 
                total_reach, vertical_reach, movement_consistency
            )
        }
        
        return report
    
    def _generate_health_insights(self, duration, avg_velocity, stability, 
                                  h_reach, v_reach, consistency):
        """Generate health and fitness insights"""
        insights = []
        
        # Activity level
        if duration >= 600:  # 10 minutes
            insights.append("🌟 Excellent session duration! You maintained physical activity for 10+ minutes.")
        elif duration >= 300:  # 5 minutes
            insights.append("✓ Great session duration! You maintained physical activity for over 5 minutes.")
        elif duration >= 180:  # 3 minutes
            insights.append("✓ Good session length. Try to aim for 5+ minutes for better cardio benefits.")
        else:
            insights.append("→ Short session. Aim for 5+ minutes for better fitness benefits.")
        
        # Movement intensity
        if avg_velocity > 200:
            insights.append("🔥 High movement intensity! Excellent for cardiovascular health.")
        elif avg_velocity > 100:
            insights.append("✓ Moderate movement intensity. Good for maintaining fitness.")
        else:
            insights.append("→ Consider increasing movement speed for better cardio workout.")
        
        # Range of motion
        if h_reach > 400 and v_reach > 300:
            insights.append("🎯 Excellent range of motion! Great for flexibility and shoulder mobility.")
        elif h_reach > 250 and v_reach > 200:
            insights.append("✓ Good range of motion. Keep stretching to improve flexibility.")
        else:
            insights.append("→ Try to extend your reach further to improve flexibility.")
        
        # Stability
        if stability > 85:
            insights.append("💪 Excellent core stability! Your posture control is very strong.")
        elif stability > 70:
            insights.append("✓ Good core stability. Your balance is solid.")
        elif stability > 50:
            insights.append("→ Work on core stability exercises to improve balance.")
        else:
            insights.append("→ Focus on core strengthening and balance exercises.")
        
        # Consistency
        if consistency > 80:
            insights.append("⭐ Very consistent movements! This shows good motor control.")
        elif consistency > 60:
            insights.append("✓ Fairly consistent movement patterns.")
        else:
            insights.append("→ Practice smoother, more controlled movements.")
        
        # Calorie burn insight
        if self.calories_burned > 30:
            insights.append(f"🔥 Burned approximately {self.calories_burned:.1f} calories!")
        elif self.calories_burned > 15:
            insights.append(f"✓ Burned approximately {self.calories_burned:.1f} calories.")
        
        return insights
    
    def save_to_file(self, filename=None):
        """Save analytics data to JSON file"""
        if filename is None:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S') if self.start_time else 'unknown'
            filename = f"snake_ar_report_{timestamp}.json"
        
        report = self.generate_report()
        if report:
            try:
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                return filename
            except Exception as e:
                print(f"Error saving report: {e}")
                return None
        return None

    def save_to_excel(self, filename="snake_analytics.xlsx"):
        """Append all metrics for this session as a single row to the cumulative Excel file."""
        report = self.generate_report()
        if not report:
            return None
        try:
            row = {}
            row.update(report['session_info'])
            row.update(report['movement_metrics'])
            row.update(report['range_of_motion'])
            row.update(report['arm_metrics'])
            row.update(report['performance_metrics'])
            row['health_insights'] = ' | '.join(report['health_insights'])
            new_df = pd.DataFrame([row])
            if os.path.exists(filename):
                existing_df = pd.read_excel(filename)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            combined_df.to_excel(filename, index=False)
            print(f"✓ Saved Excel report: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving Excel report: {e}")
            return None
    
    def get_summary_text(self):
        """Get a formatted text summary of the report"""
        report = self.generate_report()
        if not report:
            return "No data available"
        
        text = f"""
╔══════════════════════════════════════════════════════════════╗
║           SNAKE AR - MOVEMENT ANALYSIS REPORT                ║
╚══════════════════════════════════════════════════════════════╝

📅 SESSION INFORMATION
─────────────────────────────────────────────────────────────
Date & Time: {report['session_info']['date']}
Duration: {report['session_info']['duration_played']} / {report['session_info']['duration_target']}
Mode: {report['session_info']['mode'].upper()}
Difficulty: {report['session_info']['difficulty'].upper()}
Active Hand: {report['session_info']['active_hand']}
Final Score: {report['session_info']['final_score']}
Calories Burned: {report['session_info']['calories_burned']}

🏃 MOVEMENT METRICS
─────────────────────────────────────────────────────────────
Total Distance Moved: {report['movement_metrics']['total_distance']}
Average Velocity: {report['movement_metrics']['avg_velocity']}
Maximum Velocity: {report['movement_metrics']['max_velocity']}
Movement Efficiency: {report['movement_metrics']['movement_efficiency']}
Movement Consistency: {report['movement_metrics']['movement_consistency']}

📏 RANGE OF MOTION
─────────────────────────────────────────────────────────────
Horizontal Reach: {report['range_of_motion']['horizontal_reach']}
  ← Left: {report['range_of_motion']['max_left_reach']}
  → Right: {report['range_of_motion']['max_right_reach']}

Vertical Reach: {report['range_of_motion']['vertical_reach']}
  ↑ Up: {report['range_of_motion']['max_up_reach']}
  ↓ Down: {report['range_of_motion']['max_down_reach']}

💪 ARM EXTENSION
─────────────────────────────────────────────────────────────
Average Extension: {report['arm_metrics']['avg_arm_extension']}
Maximum Extension: {report['arm_metrics']['max_arm_extension']}

📊 PERFORMANCE METRICS
─────────────────────────────────────────────────────────────
Body Stability Score: {report['performance_metrics']['body_stability_score']}
Average Reaction Time: {report['performance_metrics']['avg_reaction_time']}
Foods Collected: {report['performance_metrics']['foods_collected']}
Total Collisions: {report['performance_metrics']['total_collisions']}
  • Wall Collisions: {report['performance_metrics']['wall_collisions']}
  • Self Collisions: {report['performance_metrics']['self_collisions']}
Collision Rate: {report['performance_metrics']['collision_rate']}

💡 HEALTH & FITNESS INSIGHTS
─────────────────────────────────────────────────────────────
"""
        for insight in report['health_insights']:
            text += f"{insight}\n"
        
        text += "\n" + "═" * 62 + "\n"
        
        return text

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
            spawn_range_y = int(width_span * 0.8)

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
# SNAKE LOGIC (WITH ANALYTICS INTEGRATED)
# =====================================================
class SnakeLogic:
    def __init__(self, width=1280, height=720):
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
        
        # ANALYTICS INTEGRATION
        self.analytics = MovementAnalytics()
        self.current_pose_landmarks = None

    def start_calibration(self, calib_mode):
        self.mode = 'calibration'
        self.calib_mode = calib_mode
        self.calibration_done = False
        self.hold_timer = 0
        self.calib_min_x = self.width
        self.calib_max_x = 0
        self.calib_min_y = self.height
        self.calib_max_y = 0
        self.show_bounding_box = False
        self.calibration_bounds_preview = None

    def reset_calibration(self):
        self.hold_timer = 0
        self.calib_min_x = self.width
        self.calib_max_x = 0
        self.calib_min_y = self.height
        self.calib_max_y = 0
        self.show_bounding_box = False
        self.calibration_bounds_preview = None

    def force_complete(self):
        if self.calibration_bounds_preview:
            self.bounds = self.calibration_bounds_preview
        else:
            self.bounds = (max(0, self.calib_min_x - 20), min(self.width, self.calib_max_x + 20),
                       max(0, self.calib_min_y - 20), min(self.height, self.calib_max_y + 20))
        
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

    def start_game(self, diff, dur, hand):
        self.mode = 'game'
        self.difficulty = diff
        self.duration = dur
        self.selected_hand = hand
        self.game_over_flag = False
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
        self.remaining = dur
        self.cached_border = None
        
        # ANALYTICS: Start tracking session
        self.analytics.start_session(
            mode=self.calib_mode,
            difficulty=diff,
            duration=dur,
            hand=hand
        )
        
        # Speed factors remain the same
        if diff == "easy": self.speed_factor = 0.10
        elif diff == "medium": self.speed_factor = 0.20
        else: self.speed_factor = 0.35

    def process_frame(self, frame):
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.mode == 'idle':
            return frame

        elif self.mode == 'calibration':
            within = False
            results_pose = self.pose.process(img_rgb)
            if results_pose.pose_landmarks:
                self.mp_draw.draw_landmarks(frame, results_pose.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                lw = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
                rw = results_pose.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
                lx, ly = int(lw.x * w), int(lw.y * h)
                rx, ry = int(rw.x * w), int(rw.y * h)
                
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

                arm_span = abs(lx - rx)
                if arm_span > w * 0.4:
                    within = True
                    self.show_bounding_box = True
                else:
                    within = False
                    if self.hold_timer < self.hold_time:
                        self.hold_timer = max(0, self.hold_timer - dt * 2)

            if within:
                if self.hold_timer < self.hold_time:
                    self.hold_timer += dt
                else:
                    self.hold_timer = self.hold_time
            elif self.hold_timer < self.hold_time:
                self.hold_timer = max(0, self.hold_timer - dt)

            if self.show_bounding_box and self.calibration_bounds_preview:
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
                
                corner_length = 30
                cv2.line(frame, (min_x, min_y), (min_x + corner_length, min_y), box_color, thickness)
                cv2.line(frame, (min_x, min_y), (min_x, min_y + corner_length), box_color, thickness)
                cv2.line(frame, (max_x, min_y), (max_x - corner_length, min_y), box_color, thickness)
                cv2.line(frame, (max_x, min_y), (max_x, min_y + corner_length), box_color, thickness)
                cv2.line(frame, (min_x, max_y), (min_x + corner_length, max_y), box_color, thickness)
                cv2.line(frame, (min_x, max_y), (min_x, max_y - corner_length), box_color, thickness)
                cv2.line(frame, (max_x, max_y), (max_x - corner_length, max_y), box_color, thickness)
                cv2.line(frame, (max_x, max_y), (max_x, max_y - corner_length), box_color, thickness)

            bx, by = w//2 - 200, h - 100
            cv2.rectangle(frame, (bx, by), (bx + 400, by + 30), (50, 50, 50), -1)
            prog = int(400 * (self.hold_timer / self.hold_time))
            
            if within:
                color = (0, 255, 0)
                status_text = "ARMS STRETCHED! Hold position..."
            else:
                color = (0, 165, 255)
                status_text = "STRETCH YOUR ARMS WIDE!"
            
            if prog > 0:
                cv2.rectangle(frame, (bx, by), (bx + prog, by + 30), color, -1)
            
            progress_percent = int((self.hold_timer / self.hold_time) * 100)
            cv2.putText(frame, f"{progress_percent}%", (bx + 410, by + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            draw_shadow_text(frame, status_text, (w//2 - 250, h - 140), 1.0, color, 2)
            
            if self.show_bounding_box:
                if self.hold_timer < self.hold_time:
                    hold_text = f"HOLD FOR {self.hold_time - self.hold_timer:.1f}s TO LOCK"
                    text_color = (0, 255, 255)
                else:
                    hold_text = "BOX LOCKED! Press PROCEED to continue"
                    text_color = (0, 255, 0)
                draw_shadow_text(frame, hold_text, (w//2 - 200, h - 180), 0.8, text_color, 2)

        elif self.mode == 'game':
            bound_min_x, bound_max_x, bound_min_y, bound_max_y = self.bounds
            
            if self.food is None: 
                self.food = spawn_food(self.difficulty, self.snake_trail, w, h, self.bounds, self.calib_mode)
            
            elapsed = current_time - self.start_time
            self.remaining = max(0, self.duration - int(elapsed))
            
            if self.invincible and current_time - self.inv_time > 2.0: 
                self.invincible = False
            
            if self.remaining <= 0 or self.lives <= 0:
                # ANALYTICS: End session
                self.analytics.end_session(self.score)
                self.analytics.save_to_file()
                self.analytics.save_to_excel()
                
                self.game_over_flag = True
                self.mode = 'idle'
                return frame

            # ANALYTICS: Track pose for analytics
            results_pose = self.pose.process(img_rgb)
            if results_pose.pose_landmarks:
                self.current_pose_landmarks = results_pose.pose_landmarks

            results = self.hands.process(img_rgb)
            finger_detected = False
            
            target_hand_landmarks = None
            closest_dist = float('inf')
            center_x, center_y = w / 2, h / 2

            if results.multi_hand_landmarks and results.multi_handedness:
                box_x1, box_y1, box_x2, box_y2 = self.detection_box
                
                for idx, hand_info in enumerate(results.multi_handedness):
                    hand_label = hand_info.classification[0].label
                    
                    if hand_label == self.selected_hand:
                        temp_landmarks = results.multi_hand_landmarks[idx]
                        
                        lm8 = temp_landmarks.landmark[8]
                        cx, cy = int(lm8.x * w), int(lm8.y * h)
                        
                        if box_x1 <= cx <= box_x2 and box_y1 <= cy <= box_y2:
                            dist = math.hypot(cx - center_x, cy - center_y)
                            
                            if dist < closest_dist:
                                closest_dist = dist
                                target_hand_landmarks = temp_landmarks

            if target_hand_landmarks:
                lm = target_hand_landmarks.landmark[8]
                wx, wy = int(lm.x * w), int(lm.y * h)
                finger_detected = True
                self.smooth_points.append((wx, wy))
                if len(self.smooth_points) > 5: self.smooth_points.pop(0)
                
                if assets.images['crosshair'] is not None:
                    crosshair = assets.images['crosshair'].copy()
                    for i in range(crosshair.shape[0]):
                        for j in range(crosshair.shape[1]):
                            if crosshair[i, j, 3] > 0:
                                crosshair[i, j, 0] = 0
                                crosshair[i, j, 1] = 0
                                crosshair[i, j, 2] = 255
                    frame = overlay_image(frame, crosshair, wx-30, wy-30)
                    cv2.circle(frame, (wx, wy), 35, (0, 0, 255), 2)
                else:
                    cv2.circle(frame, (wx, wy), 35, (0, 0, 255), 3)
                    cv2.circle(frame, (wx, wy), 30, (0, 100, 255), 2)
                    cv2.circle(frame, (wx, wy), 8, (0, 0, 255), -1)

            if finger_detected and len(self.smooth_points) > 0:
                avg_x = int(sum(p[0] for p in self.smooth_points) / len(self.smooth_points))
                avg_y = int(sum(p[1] for p in self.smooth_points) / len(self.smooth_points))
                
                # ANALYTICS: Record this frame's movement
                self.analytics.record_frame(
                    hand_pos=(avg_x, avg_y),
                    pose_landmarks=self.current_pose_landmarks,
                    frame_width=w,
                    frame_height=h
                )
                
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

            hit = False
            collision_type = None
            correction = 90
            if self.calib_mode == "sitting":
                correction = 80
            
            if self.difficulty == "hard":
                if (x < bound_min_x + correction or x > bound_max_x - correction or 
                    y < bound_min_y + correction or y > bound_max_y - correction):
                    hit = True
                    collision_type = 'wall'

            if not hit and not self.invincible and len(self.snake_trail) > 20:
                head_pt = np.array([x, y])
                for i in range(0, len(self.snake_trail) - 15, 2):
                    if np.linalg.norm(head_pt - np.array(self.snake_trail[i])) < 20:
                        hit = True
                        collision_type = 'self'
                        break

            if hit and not self.invincible:
                # ANALYTICS: Record collision
                self.analytics.record_collision(collision_type, (x, y))
                
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
                # ANALYTICS: Record food collection
                self.analytics.record_food_collection(self.food)
                
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

            time_color = (0, 255, 255) if self.remaining > 10 else (255, 80, 80)
            time_str = f"TIME  {self.remaining // 60:02d}:{self.remaining % 60:02d}"
            draw_shadow_text(frame, time_str, (w//2 - 150, 70), 1.4, time_color, 3)

            # Lives
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

            time.sleep(0.01)

        cap.release()

    def stop(self):
        self.running = False
        self.wait()

# =====================================================
# UI MAIN WINDOW (WITH REPORT SCREEN ADDED)
# =====================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake AR - Jungle Adventure")
        self.setGeometry(100, 100, 1280, 720)

        # UI Configuration
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

            'diff_title_img':   {'x': 730, 'y': 270,  'w': 450, 'h': 200, 'center_x': False},
            'diff_easy':        {'x': 800, 'y': 500, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_med':         {'x': 800, 'y': 600, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_hard':        {'x': 800, 'y': 700, 'w': 350, 'h': 70,  'center_x': False, 'font': 22},
            'diff_back':        {'x': 870, 'y': 820, 'w': 200, 'h': 60,  'center_x': False, 'font': 22},

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

            'game_quit':        {'x': 1680, 'y': 45,  'w': 180, 'h': 60,  'center_x': False, 'font': 20},

            'over_card':        {'x': 300, 'y': 270, 'w': 500, 'h': 500, 'center_x': True},
            
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
        self.duration = 120
        self.calibration_mode = "standing"
        self.selected_hand = "Right"
        self.calibrated_bounds = (50, 1230, 100, 670)
        
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)

        self.loader = LoadingOverlay(self)
        self.loader.resize(1280, 720)
        self.loader.hide()

        # Initialize screens (added report screen at index 9)
        self.init_start_screen()      # 0
        self.init_mode_screen()       # 1
        self.init_hand_screen()       # 2
        self.init_calibration_screen()# 3
        self.init_difficulty_screen() # 4
        self.init_duration_screen()   # 5
        self.init_instruction_screen()# 6
        self.init_game_screen()       # 7
        self.init_gameover_screen()   # 8
        self.init_report_screen()     # 9 - NEW REPORT SCREEN
        
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
        
        if self.calibration_mode == "sitting":
            self.difficulty = "medium"
            self.central_widget.setCurrentIndex(5)
        else:
            self.central_widget.setCurrentIndex(4)

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
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.calib_label = QLabel()
        self.calib_label.setAlignment(Qt.AlignCenter)
        self.calib_label.setScaledContents(True)
        self.calib_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.calib_label, stretch=1)

        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(10, 10, 10, 10)
        
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
        btn_proceed.setEnabled(False)
        button_layout.addWidget(btn_proceed)
        
        main_layout.addLayout(button_layout)
        page.setLayout(main_layout)
        
        self.calib_back_btn = btn_back
        self.calib_recalib_btn = btn_recalibrate
        self.calib_proceed_btn = btn_proceed
        
        self.central_widget.addWidget(page)

    def calibration_back(self):
        if assets.sounds['click']: assets.sounds['click'].play()
        self.snake_logic.mode = 'idle'
        self.snake_logic.reset_calibration()
        self.calib_proceed_btn.setEnabled(False)
        self.central_widget.setCurrentIndex(6)

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
        if assets.sounds['click']: assets.sounds['click'].play()
        if self.calibration_mode == "sitting":
            self.central_widget.setCurrentIndex(2)
        else:
            self.central_widget.setCurrentIndex(4)

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
        
        # CHANGED: View Report button instead of Play Again
        btn_report = QPushButton("📊 VIEW REPORT")
        btn_report.setFixedSize(250, 60)
        btn_report.clicked.connect(self.show_report)
        
        btn_quit = QPushButton("QUIT")
        btn_quit.setFixedSize(250, 60)
        btn_quit.clicked.connect(self.close)

        lbl_text2 = QLabel(" ")
        lbl_text2.setStyleSheet("color: #FFD700; font-weight: bold; font-size: 24px; letter-spacing: 2px;")

        card_layout.addWidget(lbl_over, 0, Qt.AlignCenter)
        card_layout.addWidget(lbl_text, 0, Qt.AlignCenter)
        card_layout.addWidget(self.final_score, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_report, 0, Qt.AlignCenter)
        card_layout.addWidget(btn_quit, 0, Qt.AlignCenter)
        card_layout.addWidget(lbl_text2, 0, Qt.AlignCenter)
        
        self.central_widget.addWidget(page)
    
    # NEW REPORT SCREEN
    def init_report_screen(self):
        """Initialize the analytics report screen"""
        page = QWidget()
        page.setStyleSheet("background-color: rgba(20, 20, 30, 250);")
        
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 30, 50, 30)
        
        # Title
        title = QLabel("📊 MOVEMENT ANALYSIS REPORT")
        title.setStyleSheet("""
            color: #FFD700;
            font-size: 36px;
            font-weight: bold;
            padding: 20px;
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Scrollable report area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: 3px solid #FFD700;
                border-radius: 10px;
                background-color: rgba(30, 30, 40, 200);
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 60, 200);
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #FFD700;
                border-radius: 7px;
            }
        """)
        
        # Report content widget
        self.report_content = QLabel()
        self.report_content.setWordWrap(True)
        self.report_content.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.report_content.setStyleSheet("""
            color: white;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            padding: 20px;
            line-height: 1.6;
        """)
        self.report_content.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        scroll.setWidget(self.report_content)
        layout.addWidget(scroll)
        
        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(20)
        
        # Export button
        btn_export = QPushButton("💾 EXPORT TO FILE")
        btn_export.setFixedSize(250, 60)
        btn_export.clicked.connect(self.export_report)
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: rgba(34, 139, 34, 200);
                color: white;
                border: 3px solid #228B22;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(50, 205, 50, 220);
            }
        """)
        
        # Play again button
        btn_again = QPushButton("🔄 PLAY AGAIN")
        btn_again.setFixedSize(250, 60)
        btn_again.clicked.connect(self.restart)
        btn_again.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 140, 0, 200);
                color: white;
                border: 3px solid #FF8C00;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(255, 165, 0, 220);
            }
        """)
        
        # Main menu button
        btn_menu = QPushButton("🏠 MAIN MENU")
        btn_menu.setFixedSize(250, 60)
        btn_menu.clicked.connect(lambda: self.play_sound_and_switch(0))
        btn_menu.setStyleSheet("""
            QPushButton {
                background-color: rgba(139, 69, 19, 200);
                color: white;
                border: 3px solid #8B4513;
                border-radius: 10px;
                font-weight: bold;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: rgba(160, 82, 45, 220);
            }
        """)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_export)
        btn_layout.addWidget(btn_again)
        btn_layout.addWidget(btn_menu)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        self.central_widget.addWidget(page)
    
    def show_report(self):
        """Display the analytics report"""
        if assets.sounds['click']:
            assets.sounds['click'].play()
        
        # Generate and display report
        report_text = self.snake_logic.analytics.get_summary_text()
        self.report_content.setText(report_text)
        
        # Switch to report screen (index 9)
        self.central_widget.setCurrentIndex(9)
    
    def export_report(self):
        """Export report to JSON and Excel files"""
        if assets.sounds['click']:
            assets.sounds['click'].play()
        
        json_file = self.snake_logic.analytics.save_to_file()
        excel_file = self.snake_logic.analytics.save_to_excel()
        if json_file or excel_file:
            saved_files = []
            if json_file: saved_files.append(f"JSON: {json_file}")
            if excel_file: saved_files.append(f"Excel: {excel_file}")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Reports Saved")
            msg.setText("Reports saved successfully!\n\n" + "\n".join(saved_files))
            msg.setStyleSheet("""
                QMessageBox {
                    background-color: #2a2a3e;
                    color: white;
                }
                QPushButton {
                    background-color: #4a4a5e;
                    color: white;
                    border: 2px solid #FFD700;
                    padding: 5px 15px;
                    min-width: 80px;
                }
            """)
            msg.exec_()

    def update_image(self, qt_img):
        idx = self.central_widget.currentIndex()
        if idx == 3:
            self.calib_label.setPixmap(QPixmap.fromImage(qt_img))
            if self.snake_logic.hold_timer >= self.snake_logic.hold_time:
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

    def closeEvent(self, event):
        self.thread.stop()
        pygame.mixer.quit()
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