import sys
import cv2
import math
import mediapipe as mp
import numpy as np
import time
import random
import os
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QStackedLayout,
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, QToolButton, QSizePolicy, 
                             QFrame, QButtonGroup, QRadioButton, QGraphicsDropShadowEffect,
                             QScrollArea, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRect, QSize, QUrl
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QBrush, QColor, QIcon
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist

# =====================================================
# MOVEMENT ANALYTICS CLASS (INTEGRATED)
# =====================================================
class JugglingAnalytics:
    """Tracks and analyzes body movements during juggling gameplay"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all analytics data"""
        self.start_time = None
        self.end_time = None
        self.hand_positions = {'left': [], 'right': []}
        self.hand_velocities = {'left': [], 'right': []}
        self.wrist_positions = {'left': [], 'right': []}
        self.shoulder_positions = {'left': [], 'right': []}
        self.elbow_positions = {'left': [], 'right': []}
        self.body_center_positions = []
        self.total_hand_distance = {'left': 0, 'right': 0}
        self.max_hand_speed = {'left': 0, 'right': 0}
        self.max_reach_left = 0
        self.max_reach_right = 0
        self.max_reach_up = 0
        self.max_reach_down = 0
        self.frame_count = 0
        self.hit_events = []
        self.miss_events = []
        self.hand_coordination_scores = []
        self.stance_mode = None
        self.difficulty_level = None
        self.duration_target = 0
        self.final_score = 0
        self.total_airtime = 0
        self.calories_burned = 0
        self.avg_reaction_time = 0
        
    def start_session(self, stance, level, duration):
        """Start a new analytics session"""
        self.reset()
        self.start_time = datetime.now()
        self.stance_mode = stance
        self.difficulty_level = level
        self.duration_target = duration
    
    def record_frame(self, hand_landmarks_list, frame_width, frame_height):
        """Record movement data for a single frame"""
        self.frame_count += 1
        
        if not hand_landmarks_list:
            return
            
        # Process each detected hand
        for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
            # Determine which hand (left/right)
            hand_label = 'left' if hand_idx == 0 else 'right'
            
            # Get hand center (average of key points)
            indices = [0, 5, 9, 13, 17]
            cx = int(np.mean([hand_landmarks.landmark[i].x * frame_width for i in indices]))
            cy = int(np.mean([hand_landmarks.landmark[i].y * frame_height for i in indices]))
            
            current_pos = {'x': cx, 'y': cy, 'timestamp': time.time()}
            self.hand_positions[hand_label].append(current_pos)
            
            # Calculate velocity if we have previous position
            if len(self.hand_positions[hand_label]) > 1:
                prev = self.hand_positions[hand_label][-2]
                dx = current_pos['x'] - prev['x']
                dy = current_pos['y'] - prev['y']
                dt = current_pos['timestamp'] - prev['timestamp']
                
                if dt > 0:
                    speed = math.sqrt(dx*dx + dy*dy) / dt
                    self.hand_velocities[hand_label].append(speed)
                    self.total_hand_distance[hand_label] += math.sqrt(dx*dx + dy*dy)
                    self.max_hand_speed[hand_label] = max(self.max_hand_speed[hand_label], speed)
            
            # Track wrist position
            wrist = hand_landmarks.landmark[0]
            self.wrist_positions[hand_label].append({
                'x': int(wrist.x * frame_width),
                'y': int(wrist.y * frame_height),
                'timestamp': time.time()
            })
            
            # Track reach extremes
            center_x, center_y = frame_width / 2, frame_height / 2
            if cx < center_x:
                self.max_reach_left = max(self.max_reach_left, abs(cx - center_x))
            else:
                self.max_reach_right = max(self.max_reach_right, abs(cx - center_x))
            
            if cy < center_y:
                self.max_reach_up = max(self.max_reach_up, abs(cy - center_y))
            else:
                self.max_reach_down = max(self.max_reach_down, abs(cy - center_y))
        
        # Calculate hand coordination if both hands detected
        if len(hand_landmarks_list) >= 2:
            left_hand = hand_landmarks_list[0]
            right_hand = hand_landmarks_list[1]
            
            # Calculate symmetry score (0-100)
            left_y = np.mean([left_hand.landmark[i].y for i in [0, 5, 9, 13, 17]])
            right_y = np.mean([right_hand.landmark[i].y for i in [0, 5, 9, 13, 17]])
            
            symmetry = 100 - min(100, abs(left_y - right_y) * 200)
            self.hand_coordination_scores.append(symmetry)
    
    def record_hit(self, position):
        """Record a successful hit event"""
        self.hit_events.append({
            'position': position,
            'timestamp': time.time()
        })
    
    def record_miss(self, position):
        """Record a miss event (ball fell)"""
        self.miss_events.append({
            'position': position,
            'timestamp': time.time()
        })
    
    def end_session(self, final_score, total_airtime):
        """End the analytics session"""
        self.end_time = datetime.now()
        self.final_score = final_score
        self.total_airtime = total_airtime
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate derived metrics"""
        if not self.start_time or not self.end_time:
            return
        
        duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        
        # Calculate average hand speed
        left_avg_speed = np.mean(self.hand_velocities['left']) if self.hand_velocities['left'] else 0
        right_avg_speed = np.mean(self.hand_velocities['right']) if self.hand_velocities['right'] else 0
        avg_speed = (left_avg_speed + right_avg_speed) / 2
        
        # Calorie estimation based on activity intensity
        base_calories_per_minute = 4  # Juggling is moderate-high intensity
        
        if avg_speed > 300:
            intensity_multiplier = 2.8  # Very high intensity
        elif avg_speed > 200:
            intensity_multiplier = 2.2  # High intensity
        elif avg_speed > 100:
            intensity_multiplier = 1.6  # Moderate intensity
        else:
            intensity_multiplier = 1.2  # Light intensity
        
        self.calories_burned = duration_minutes * base_calories_per_minute * intensity_multiplier
        
        # Calculate reaction time
        if len(self.hit_events) > 1:
            reaction_times = []
            for i in range(1, len(self.hit_events)):
                reaction_times.append(self.hit_events[i]['timestamp'] - self.hit_events[i-1]['timestamp'])
            self.avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
    
    def generate_report(self):
        """Generate a comprehensive juggling analysis report"""
        if not self.start_time or not self.end_time:
            return None
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Hand movement statistics
        left_avg_speed = np.mean(self.hand_velocities['left']) if self.hand_velocities['left'] else 0
        right_avg_speed = np.mean(self.hand_velocities['right']) if self.hand_velocities['right'] else 0
        
        # Coordination metrics
        avg_coordination = np.mean(self.hand_coordination_scores) if self.hand_coordination_scores else 0
        
        # Performance metrics
        total_hits = len(self.hit_events)
        total_misses = len(self.miss_events)
        hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
        
        # Range of motion
        total_reach = self.max_reach_left + self.max_reach_right
        vertical_reach = self.max_reach_up + self.max_reach_down
        
        # Hand balance (how evenly both hands were used)
        total_left = self.total_hand_distance['left']
        total_right = self.total_hand_distance['right']
        hand_balance = 100 - abs(total_left - total_right) / max(total_left + total_right, 1) * 100
        
        report = {
            'session_info': {
                'date': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_played': f"{int(duration // 60)}:{int(duration % 60):02d}",
                'duration_target': f"{int(self.duration_target // 60)}:{int(self.duration_target % 60):02d}",
                'stance': self.stance_mode,
                'difficulty': f"Level {self.difficulty_level}",
                'final_score': self.final_score,
                'total_airtime': f"{self.total_airtime:.1f}s",
                'calories_burned': f"{self.calories_burned:.1f} kcal"
            },
            'hand_metrics': {
                'left_hand_distance': f"{total_left:.1f} pixels",
                'right_hand_distance': f"{total_right:.1f} pixels",
                'left_avg_speed': f"{left_avg_speed:.1f} px/s",
                'right_avg_speed': f"{right_avg_speed:.1f} px/s",
                'left_max_speed': f"{self.max_hand_speed['left']:.1f} px/s",
                'right_max_speed': f"{self.max_hand_speed['right']:.1f} px/s",
                'hand_balance': f"{hand_balance:.1f}%"
            },
            'coordination_metrics': {
                'avg_hand_coordination': f"{avg_coordination:.1f}/100",
                'avg_reaction_time': f"{self.avg_reaction_time:.2f} seconds",
                'frames_tracked': self.frame_count
            },
            'range_of_motion': {
                'horizontal_reach': f"{total_reach:.1f} pixels",
                'max_left_reach': f"{self.max_reach_left:.1f} pixels",
                'max_right_reach': f"{self.max_reach_right:.1f} pixels",
                'vertical_reach': f"{vertical_reach:.1f} pixels",
                'max_up_reach': f"{self.max_reach_up:.1f} pixels",
                'max_down_reach': f"{self.max_reach_down:.1f} pixels"
            },
            'performance_metrics': {
                'total_hits': total_hits,
                'total_misses': total_misses,
                'hit_rate': f"{hit_rate:.1f}%",
                'hits_per_minute': f"{(total_hits / (duration / 60)):.1f}" if duration > 0 else "0.0"
            },
            'health_insights': self._generate_health_insights(
                duration, left_avg_speed, right_avg_speed, avg_coordination, 
                total_reach, vertical_reach, hand_balance
            )
        }
        
        return report
    
    def _generate_health_insights(self, duration, left_speed, right_speed, coordination, 
                                  h_reach, v_reach, hand_balance):
        """Generate health and fitness insights"""
        insights = []
        
        # Activity duration
        if duration >= 600:
            insights.append("🌟 Excellent session! 10+ minutes of juggling provides great cardio and coordination benefits.")
        elif duration >= 300:
            insights.append("✓ Great session duration! You maintained active juggling for 5+ minutes.")
        elif duration >= 180:
            insights.append("✓ Good session. Aim for 5+ minutes to maximize cardio benefits.")
        else:
            insights.append("→ Try longer sessions (5+ minutes) for better fitness results.")
        
        # Hand speed and intensity
        avg_speed = (left_speed + right_speed) / 2
        if avg_speed > 250:
            insights.append("🔥 Exceptional hand speed! This level of intensity is excellent for reflexes and agility.")
        elif avg_speed > 150:
            insights.append("✓ Great hand movement intensity! Good for improving hand-eye coordination.")
        else:
            insights.append("→ Try increasing movement speed for better reflex training.")
        
        # Hand coordination
        if coordination > 85:
            insights.append("🎯 Outstanding hand coordination! Your bilateral coordination is excellent.")
        elif coordination > 70:
            insights.append("✓ Good hand coordination. Both hands are working well together.")
        else:
            insights.append("→ Practice synchronized movements to improve bilateral coordination.")
        
        # Hand balance
        if hand_balance > 85:
            insights.append("⚖️ Excellent hand balance! You're using both hands equally.")
        elif hand_balance > 70:
            insights.append("✓ Good hand balance between left and right.")
        else:
            insights.append("→ Try to use both hands more evenly for balanced development.")
        
        # Range of motion
        if h_reach > 400 and v_reach > 300:
            insights.append("💪 Excellent range of motion! Great for shoulder mobility and flexibility.")
        elif h_reach > 250 and v_reach > 200:
            insights.append("✓ Good range of motion. Keep extending your reach.")
        else:
            insights.append("→ Extend your reach further to improve flexibility.")
        
        # Calorie burn
        if self.calories_burned > 40:
            insights.append(f"🔥 Burned approximately {self.calories_burned:.1f} calories! Great workout!")
        elif self.calories_burned > 20:
            insights.append(f"✓ Burned approximately {self.calories_burned:.1f} calories.")
        
        return insights
    
    def save_to_file(self, filename=None):
        """Save analytics data to JSON file"""
        if filename is None:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S') if self.start_time else 'unknown'
            filename = f"juggling_report_{timestamp}.json"
        
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

    def save_to_excel(self, filename="juggling_analytics.xlsx"):
        """Append all metrics for this session as a single row to the cumulative Excel file."""
        report = self.generate_report()
        if not report:
            return None
        try:
            row = {}
            row.update(report['session_info'])
            row.update(report['hand_metrics'])
            row.update(report['coordination_metrics'])
            row.update(report['range_of_motion'])
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
║        VIRTUAL JUGGLING PRO - MOVEMENT ANALYSIS              ║
╚══════════════════════════════════════════════════════════════╝

📅 SESSION INFORMATION
─────────────────────────────────────────────────────────────
Date & Time: {report['session_info']['date']}
Duration: {report['session_info']['duration_played']} / {report['session_info']['duration_target']}
Stance: {report['session_info']['stance']}
Difficulty: {report['session_info']['difficulty']}
Final Score: {report['session_info']['final_score']}
Total Airtime: {report['session_info']['total_airtime']}
Calories Burned: {report['session_info']['calories_burned']}

🤲 HAND MOVEMENT METRICS
─────────────────────────────────────────────────────────────
Left Hand Distance: {report['hand_metrics']['left_hand_distance']}
Right Hand Distance: {report['hand_metrics']['right_hand_distance']}
Left Avg Speed: {report['hand_metrics']['left_avg_speed']}
Right Avg Speed: {report['hand_metrics']['right_avg_speed']}
Left Max Speed: {report['hand_metrics']['left_max_speed']}
Right Max Speed: {report['hand_metrics']['right_max_speed']}
Hand Balance: {report['hand_metrics']['hand_balance']}

🎯 COORDINATION METRICS
─────────────────────────────────────────────────────────────
Hand Coordination Score: {report['coordination_metrics']['avg_hand_coordination']}
Average Reaction Time: {report['coordination_metrics']['avg_reaction_time']}
Frames Tracked: {report['coordination_metrics']['frames_tracked']}

📏 RANGE OF MOTION
─────────────────────────────────────────────────────────────
Horizontal Reach: {report['range_of_motion']['horizontal_reach']}
  ← Left: {report['range_of_motion']['max_left_reach']}
  → Right: {report['range_of_motion']['max_right_reach']}

Vertical Reach: {report['range_of_motion']['vertical_reach']}
  ↑ Up: {report['range_of_motion']['max_up_reach']}
  ↓ Down: {report['range_of_motion']['max_down_reach']}

📊 PERFORMANCE METRICS
─────────────────────────────────────────────────────────────
Total Hits: {report['performance_metrics']['total_hits']}
Total Misses: {report['performance_metrics']['total_misses']}
Hit Rate: {report['performance_metrics']['hit_rate']}
Hits Per Minute: {report['performance_metrics']['hits_per_minute']}

💡 HEALTH & FITNESS INSIGHTS
─────────────────────────────────────────────────────────────
"""
        for insight in report['health_insights']:
            text += f"{insight}\n"
        
        text += "\n" + "═" * 62 + "\n"
        
        return text

# --- EXE PATH HELPER ---
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- GAME CONFIGURATION ---
LEVEL_CONFIG = {
    1: { "balls": 1, "gravity": 0.2, "base_speed": 4.0, "desc": "Level 1: Warm Up", "details": "1 Ball  •  Slow" },
    2: { "balls": 1, "gravity": 0.3, "base_speed": 6.0, "desc": "Level 2: Standard", "details": "1 Ball  •  Normal" },
    3: { "balls": 2, "gravity": 0.4, "base_speed": 7.0, "desc": "Level 3: Double Trouble", "details": "2 Balls  •  Fast" },
    4: { "balls": 2, "gravity": 0.55, "base_speed": 9.0, "desc": "Level 4: Expert Mode", "details": "2 Balls  •  Extreme" }
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

    QFrame#HUD { background-color: rgba(0, 0, 0, 150); border-bottom: 2px solid #00fff5; border-radius: 0px; }
    QLabel#ScoreLabel { color: #00ff00; font-size: 24px; font-weight: bold; }
    QLabel#LevelLabel { color: #ffcc00; font-size: 24px; font-weight: bold; }
    QLabel#TimerLabel { color: #00fff5; font-size: 24px; font-weight: bold; }
    
    QFrame#Panel { background-color: rgba(0,0,0,180); border-radius: 20px; border: 1px solid #444; }
"""

class GameLogic:
    def __init__(self, width=1280, height=720):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.width = width
        self.height = height
        self.active = False
        self.paused = False
        self.calibration_active = False
        self.calibration_mode = "Sitting"
        self.game_over = False
        self.calibration_y = height - 100 
        self.play_area_min_x = 0
        self.play_area_max_x = width
        self.calibrated = False
        self.current_level = 1
        self.score = 0
        self.total_airtime = 0
        self.selected_duration = 120
        self.time_remaining = 120
        self.last_time_check = 0
        self.balls = []
        self.prev_hand_positions = {}
        self.last_spawn_time = 0
        self.hand_radius = 50
        self.push_force = 15
        self.max_push_angle = 35
        
        # ANALYTICS INTEGRATION
        self.analytics = JugglingAnalytics()
        self.current_hand_landmarks = None

    def reset_game(self, level):
        self.current_level = level
        self.score = 0
        self.balls = []
        self.total_airtime = 0
        self.game_over = False
        self.paused = False
        self.last_spawn_time = time.time()
        self.prev_hand_positions = {}
        self.time_remaining = self.selected_duration
        self.last_time_check = time.time()
        
        # ANALYTICS: Start tracking session
        self.analytics.start_session(
            stance=self.calibration_mode,
            level=level,
            duration=self.selected_duration
        )

    def calibrate(self, hand_landmarks_list):
        if len(hand_landmarks_list) < 2: return False
        h, w = self.height, self.width
        wrist_1_x = hand_landmarks_list[0].landmark[0].x * w
        wrist_1_y = hand_landmarks_list[0].landmark[0].y * h
        wrist_2_x = hand_landmarks_list[1].landmark[0].x * w
        wrist_2_y = hand_landmarks_list[1].landmark[0].y * h
        avg_y = (wrist_1_y + wrist_2_y) / 2
        self.calibration_y = int(avg_y) - 50 
        min_x = min(wrist_1_x, wrist_2_x)
        max_x = max(wrist_1_x, wrist_2_x)
        if self.calibration_mode == "Sitting":
            self.play_area_min_x = int(min_x)
            self.play_area_max_x = int(max_x)
        else:
            span = max_x - min_x
            center = (min_x + max_x) / 2
            self.play_area_min_x = int(max(0, center - (span * 0.75)))
            self.play_area_max_x = int(min(self.width, center + (span * 0.75)))
        self.calibrated = True
        return True

    def spawn_ball(self):
        config = LEVEL_CONFIG[self.current_level]
        if len(self.balls) < config["balls"]:
            safe_min = self.play_area_min_x + 30
            safe_max = self.play_area_max_x - 30
            if safe_max <= safe_min: safe_min, safe_max = 50, self.width - 50
            x = random.randint(safe_min, safe_max)
            ball = {
                'pos': np.array([x, 50], dtype=float), 
                'vel': np.array([0, config["base_speed"]], dtype=float), 
                'radius': 25, 
                'color': (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)), 
                'airtime': 0, 
                'last_hit_time': 0
            }
            self.balls.append(ball)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        h, w, c = frame.shape
        hand_centers = []
        
        # Store hand landmarks for analytics
        if results.multi_hand_landmarks:
            self.current_hand_landmarks = results.multi_hand_landmarks
        else:
            self.current_hand_landmarks = None
        
        if self.calibrated or self.calibration_active:
            cv2.line(frame, (0, self.calibration_y), (w, self.calibration_y), (0, 255, 255), 2)
            cv2.line(frame, (self.play_area_min_x, 0), (self.play_area_min_x, h), (255, 0, 0), 2)
            cv2.line(frame, (self.play_area_max_x, 0), (self.play_area_max_x, h), (255, 0, 0), 2)
            cv2.putText(frame, f"Mode: {self.calibration_mode}", (self.play_area_min_x + 10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if results.multi_hand_landmarks:
            # ANALYTICS: Record frame data during active gameplay
            if self.active and not self.paused:
                self.analytics.record_frame(results.multi_hand_landmarks, w, h)
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self.active and not self.paused:
                    indices = [0, 5, 9, 13, 17]
                    cx = int(np.mean([hand_landmarks.landmark[i].x * w for i in indices]))
                    cy = int(np.mean([hand_landmarks.landmark[i].y * h for i in indices]))
                    center = (cx, cy)
                    hand_centers.append(center)
                    if hand_idx in self.prev_hand_positions:
                        prev = self.prev_hand_positions[hand_idx]
                        velocity = np.array(center) - np.array(prev)
                    else: 
                        velocity = np.array([0,0])
                    self.prev_hand_positions[hand_idx] = center
                    self.check_collisions(center, velocity)
                    cv2.circle(frame, center, self.hand_radius, (0, 255, 0), 2)
            if self.calibration_active: 
                self.calibrate(results.multi_hand_landmarks)
        
        if self.active and not self.paused and not self.game_over:
            current_time = time.time()
            if current_time - self.last_time_check >= 1.0:
                self.time_remaining -= 1
                self.last_time_check = current_time
                if self.time_remaining <= 0:
                    # ANALYTICS: End session
                    self.analytics.end_session(self.score, self.total_airtime)
                    self.analytics.save_to_file()
                    self.analytics.save_to_excel()
                    self.game_over = True
                    self.time_remaining = 0
            self.update_physics()
            self.spawn_ball()
            self.draw_balls(frame)
        return frame

    def check_collisions(self, hand_center, hand_vel):
        current_time = time.time()
        for ball in self.balls:
            dist = np.linalg.norm(ball['pos'] - np.array(hand_center))
            if dist < (ball['radius'] + self.hand_radius) and (current_time - ball['last_hit_time']) > 0.3:
                # ANALYTICS: Record hit
                self.analytics.record_hit(tuple(ball['pos'].astype(int)))
                
                direction = ball['pos'] - np.array(hand_center)
                norm = np.linalg.norm(direction)
                if norm > 0: direction /= norm
                angle = np.degrees(np.arctan2(direction[0], -direction[1]))
                angle = np.clip(angle, -self.max_push_angle, self.max_push_angle)
                rad = np.radians(angle)
                direction = np.array([np.sin(rad), -np.cos(rad)])
                speed = np.linalg.norm(hand_vel)
                force = self.push_force + min(speed * 0.5, 15)
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
                # ANALYTICS: Record miss
                self.analytics.record_miss(tuple(ball['pos'].astype(int)))
                
                balls_to_remove.append(i)
                self.total_airtime += ball['airtime']
            else: 
                ball['airtime'] += 0.03
        for i in reversed(balls_to_remove): 
            self.balls.pop(i)

    def draw_balls(self, frame):
        for ball in self.balls:
            pos = tuple(ball['pos'].astype(int))
            cv2.circle(frame, pos, ball['radius'], ball['color'], -1)
            cv2.circle(frame, pos, ball['radius'], (255,255,255), 2)

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    stats_signal = pyqtSignal(dict)
    
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
                stats = {
                    "score": self.game.score, 
                    "level": self.game.current_level, 
                    "game_over": self.game.game_over, 
                    "calibrated": self.game.calibrated,
                    "time": self.game.time_remaining,
                    "airtime": self.game.total_airtime
                }
                self.stats_signal.emit(stats)
                rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                p = convert_to_Qt_format.scaled(1050, 590, Qt.AspectRatioMode.KeepAspectRatio)
                self.change_pixmap_signal.emit(p)
            time.sleep(0.01)
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
        
        # --- STACK ORDER (ADDED REPORT AT INDEX 8) ---
        # 0: Main Menu
        # 1: Setup 1 - Stance
        # 2: Setup 2 - Duration
        # 3: Setup 3 - Level
        # 4: Instructions
        # 5: Calibration
        # 6: Game
        # 7: Results
        # 8: Report (NEW)
        
        self.init_main_menu()           # 0
        self.init_stance_screen()       # 1
        self.init_duration_screen()     # 2
        self.init_level_screen()        # 3
        self.init_instruction_screen()  # 4
        self.init_calibration_screen()  # 5
        self.init_game_screen()         # 6
        self.init_result_screen()       # 7
        self.init_report_screen()       # 8 - NEW REPORT SCREEN
        
        self.thread = VideoThread(self.game_logic)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.stats_signal.connect(self.update_stats)
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
            if self.click_player.state() == QMediaPlayer.PlayingState: 
                self.click_player.stop()
            self.click_player.play()

    def play_hit(self):
        if self.hit_player.mediaStatus() != QMediaPlayer.NoMedia:
            if self.hit_player.state() == QMediaPlayer.PlayingState: 
                self.hit_player.stop()
            self.hit_player.play()

    def start_music(self):
        if self.music_player.playlist() is not None: 
            self.music_player.play()

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
        if event.key() == Qt.Key_Escape:
            self.close()
        elif self.stack.currentIndex() == 2:
            if event.key() == Qt.Key_Right or event.key() == Qt.Key_Up:
                self.change_duration(60)
            elif event.key() == Qt.Key_Left or event.key() == Qt.Key_Down:
                self.change_duration(-60)
        super().keyPressEvent(event)

    def change_duration(self, amount):
        self.play_click()
        new_time = self.game_logic.selected_duration + amount
        if 120 <= new_time <= 1500:
            self.game_logic.selected_duration = new_time
            self.update_time_label()

    def update_time_label(self):
        mins = self.game_logic.selected_duration // 60
        secs = self.game_logic.selected_duration % 60
        self.lbl_time_display.setText(f"{mins:02d}:{secs:02d}")

    # --- 0. MAIN MENU ---
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

    # --- 1. SETUP PAGE 1: STANCE ---
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
        
        self.rb_sitting = QRadioButton(" Sitting Mode")
        self.rb_standing = QRadioButton(" Standing Mode")
        self.rb_sitting.setChecked(True)
        self.rb_sitting.clicked.connect(self.play_click)
        self.rb_standing.clicked.connect(self.play_click)
        
        radio_layout = QVBoxLayout()
        radio_layout.addWidget(self.rb_sitting)
        radio_layout.addWidget(self.rb_standing)
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

    # --- 2. SETUP PAGE 2: DURATION ---
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
        
        lbl_sub = QLabel("(2 min - 25 min)")
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

    # --- 3. SETUP PAGE 3: LEVEL ---
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

    # --- 4. INSTRUCTION SCREEN ---
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

    # --- 5. CALIBRATION SCREEN ---
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
            "Step 1: Stretch BOTH arms out (like holding a big box)\n"
            "Step 4: Click CONFIRM when ready"
        )
        self.lbl_calib_instr.setStyleSheet("""
            color: #e0f7ff; font-size: 30px; line-height: 1.5;
            background: transparent;
        """)
        self.lbl_calib_instr.setWordWrap(True)
        self.lbl_calib_instr.setAlignment(Qt.AlignCenter)

        top_layout.addWidget(lbl_title)
        top_layout.addWidget(self.lbl_calib_instr)

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

        btn_confirm = QPushButton("CONFIRM CALIBRATION")
        btn_confirm.setFixedSize(340, 80)
        btn_confirm.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #00d4ff, stop:1 #0095ff);
                color: white; font-size: 24px; font-weight: bold;
                border: none; border-radius: 40px;
            }
            QPushButton:hover { background: #40e0ff; }
            QPushButton:pressed { background: #0077cc; }
        """)
        btn_confirm.setCursor(Qt.PointingHandCursor)
        btn_confirm.clicked.connect(self.finish_calibration)
        btn_confirm.clicked.connect(self.play_click)

        bottom_layout.addWidget(btn_back)
        bottom_layout.addStretch()
        bottom_layout.addWidget(btn_confirm)

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

    # --- 6. GAME SCREEN ---
    def init_game_screen(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        hud_frame = QFrame()
        hud_frame.setObjectName("HUD")
        hud_layout = QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(20, 10, 20, 10)
        
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
        btn_menu.setFixedSize(150, 40)
        btn_menu.setStyleSheet("font-size: 14px; padding: 5px;")
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
        
        self.video_label_game = QLabel()
        self.video_label_game.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label_game.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout.addWidget(hud_frame)
        layout.addWidget(self.video_label_game)
        
        page.setLayout(layout)
        self.stack.addWidget(page)

    # --- 7. RESULT SCREEN (MODIFIED TO SHOW REPORT BUTTON) ---
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
        
        # CHANGED: View Report button instead of Main Menu
        btn_report = QPushButton("📊 VIEW REPORT")
        btn_report.setMinimumWidth(200)
        btn_report.clicked.connect(self.show_report)
        
        btn_quit = QPushButton("MAIN MENU")
        btn_quit.setMinimumWidth(200)
        btn_quit.clicked.connect(self.return_to_menu)
        
        btn_row.addWidget(btn_report)
        btn_row.addWidget(btn_quit)
        
        con_layout.addWidget(lbl_title)
        con_layout.addWidget(self.lbl_final_score)
        con_layout.addWidget(self.lbl_final_stats)
        con_layout.addSpacing(20)
        con_layout.addLayout(btn_row)
        
        layout.addWidget(container)
        page.setLayout(layout)
        self.stack.addWidget(page)

    # --- 8. REPORT SCREEN (NEW) ---
    def init_report_screen(self):
        """Initialize the analytics report screen"""
        page = QWidget()
        page.setStyleSheet("background-color: rgba(20, 20, 30, 250);")
        
        layout = QVBoxLayout(page)
        layout.setContentsMargins(50, 30, 50, 30)
        
        # Title
        title = QLabel("📊 JUGGLING PERFORMANCE REPORT")
        title.setStyleSheet("""
            color: #00fff5;
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
                border: 3px solid #00fff5;
                border-radius: 10px;
                background-color: rgba(30, 30, 40, 200);
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 60, 200);
                width: 15px;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical {
                background: #00fff5;
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
        btn_again.clicked.connect(self.return_to_menu)
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
        btn_menu.clicked.connect(lambda: self.switch_to_page(0))
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
        
        self.stack.addWidget(page)
    
    def show_report(self):
        """Display the analytics report"""
        self.play_click()
        
        # Generate and display report
        report_text = self.game_logic.analytics.get_summary_text()
        self.report_content.setText(report_text)
        
        # Switch to report screen (index 8)
        self.stack.setCurrentIndex(8)
    
    def export_report(self):
        """Export report to JSON and Excel files"""
        self.play_click()
        
        json_file = self.game_logic.analytics.save_to_file()
        excel_file = self.game_logic.analytics.save_to_excel()
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
                    border: 2px solid #00fff5;
                    padding: 5px 15px;
                    min-width: 80px;
                }
            """)
            msg.exec_()

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
        self.game_logic.calibrated = False
        self.switch_to_page(4)

    def finish_calibration(self):
        self.play_click()
        if self.game_logic.calibrated:
            self.game_logic.calibration_active = False
            self.start_game()
        else:
            self.lbl_calib_instr.setText("❌ HANDS NOT DETECTED! Please stretch arms fully.")
            self.lbl_calib_instr.setStyleSheet("color: #ff4444; font-weight: bold; font-size: 18px;")

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
        
        # End analytics session
        self.game_logic.analytics.end_session(self.game_logic.score, self.game_logic.total_airtime)
        self.game_logic.analytics.save_to_file()
        self.game_logic.analytics.save_to_excel()
        
        # Force game over state
        self.game_logic.active = False
        self.game_logic.game_over = True
        self.game_logic.paused = False
        
        stats = {
            "score": self.game_logic.score,
            "level": self.game_logic.current_level,
            "game_over": True,
            "time": self.game_logic.time_remaining,
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
            if rem_sec < 10:
                 self.lbl_timer.setStyleSheet("color: red; font-size: 24px; font-weight: bold;")
            else:
                 self.lbl_timer.setStyleSheet("color: #00fff5; font-size: 24px; font-weight: bold;")
            
            if stats['game_over']:
                self.game_logic.active = False
                self.show_results(stats)

    def show_results(self, stats):
        self.stop_music()
        self.lbl_final_score.setText(f"FINAL SCORE: {stats['score']}")
        self.lbl_final_stats.setText(f"Level: {stats['level']}  |  Total Airtime: {stats['airtime']:.1f}s")
        self.switch_to_page(7)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = JugglingWindow()
    window.show()
    sys.exit(app.exec())