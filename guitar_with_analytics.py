"""
Rhythm Hero AR - MODERN UI VERSION
Enhanced with: Glassmorphism, Gradients, Neon Glows, Shadows, Animations
Professional, Attractive UI Design
"""
import pygame
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import os
import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional
import sys
import json
import pandas as pd
from datetime import datetime

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Update your Config class to use this:

    # ... rest of your config
# ============================================================================
# MOVEMENT ANALYTICS CLASS
# ============================================================================
class GuitarAnalytics:
    """Tracks and analyzes body movements during guitar gameplay"""
    
    def __init__(self):
        self.start_time: datetime = datetime.now()
        self.end_time: datetime = datetime.now()
        self.hand_positions: list = []
        self.hand_velocities: list = []
        self.wrist_positions: list = []
        self.total_hand_distance: float = 0.0
        self.max_hand_speed: float = 0.0
        self.max_reach_left: float = 0.0
        self.max_reach_right: float = 0.0
        self.max_reach_up: float = 0.0
        self.max_reach_down: float = 0.0
        self.frame_count: int = 0
        self.hit_events: list = []
        self.miss_events: list = []
        self.difficulty_level: str = "Medium"
        self.duration_target: int = 120
        self.final_score: int = 0
        self.calories_burned: float = 0.0
        self.avg_reaction_time: float = 0.0
        self.reset()
    
    def reset(self):
        """Reset all analytics data"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.hand_positions = []
        self.hand_velocities = []
        self.wrist_positions = []
        self.total_hand_distance = 0.0
        self.max_hand_speed = 0.0
        self.max_reach_left = 0.0
        self.max_reach_right = 0.0
        self.max_reach_up = 0.0
        self.max_reach_down = 0.0
        self.frame_count = 0
        self.hit_events = []
        self.miss_events = []
        self.difficulty_level = "Medium"
        self.duration_target = 120
        self.final_score = 0
        self.calories_burned = 0.0
        self.avg_reaction_time = 0.0
        
    def start_session(self, level, duration):
        """Start a new analytics session"""
        self.reset()
        self.start_time = datetime.now()
        self.difficulty_level = level
        self.duration_target = duration
    
    def record_frame(self, hand_landmarks, frame_width, frame_height):
        """Record movement data for a single frame"""
        self.frame_count += 1
        if not hand_landmarks: return
            
        # Get hand center (average of key points)
        indices = [0, 5, 9, 13, 17]
        cx = int(np.mean([hand_landmarks.landmark[i].x * frame_width for i in indices]))
        cy = int(np.mean([hand_landmarks.landmark[i].y * frame_height for i in indices]))
        
        current_pos = {'x': cx, 'y': cy, 'timestamp': time.time()}
        self.hand_positions.append(current_pos)
        
        # Calculate velocity if we have previous position
        if len(self.hand_positions) > 1:
            prev = self.hand_positions[-2]
            dx = current_pos['x'] - prev['x']
            dy = current_pos['y'] - prev['y']
            dt = current_pos['timestamp'] - prev['timestamp']
            
            if dt > 0:
                dist = math.sqrt(dx*dx + dy*dy)
                speed = dist / dt
                self.hand_velocities.append(speed)
                self.total_hand_distance += dist
                self.max_hand_speed = max(self.max_hand_speed, speed)
        
        # Track wrist position
        wrist = hand_landmarks.landmark[0]
        self.wrist_positions.append({
            'x': int(wrist.x * frame_width),
            'y': int(wrist.y * frame_height),
            'timestamp': time.time()
        })
        
        # Track reach extremes
        center_x, center_y = frame_width / 2, frame_height / 2
        if cx < center_x: self.max_reach_left = max(self.max_reach_left, abs(cx - center_x))
        else: self.max_reach_right = max(self.max_reach_right, abs(cx - center_x))
        
        if cy < center_y: self.max_reach_up = max(self.max_reach_up, abs(cy - center_y))
        else: self.max_reach_down = max(self.max_reach_down, abs(cy - center_y))
    
    def record_hit(self, position):
        """Record a successful hit event"""
        self.hit_events.append({'position': position, 'timestamp': time.time()})
    
    def record_miss(self, position):
        """Record a miss event"""
        self.miss_events.append({'position': position, 'timestamp': time.time()})
    
    def end_session(self, final_score):
        """End the analytics session"""
        self.end_time = datetime.now()
        self.final_score = final_score
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate derived metrics"""
        if not self.start_time or not self.end_time: return
        duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        avg_speed = np.mean(self.hand_velocities) if self.hand_velocities else 0
        base_calories_per_minute = 3.5  
        intensity_multiplier = 2.5 if avg_speed > 300 else (2.0 if avg_speed > 200 else (1.5 if avg_speed > 100 else 1.1))
        self.calories_burned = duration_minutes * base_calories_per_minute * intensity_multiplier
        if len(self.hit_events) > 1:
            reaction_times = [self.hit_events[i]['timestamp'] - self.hit_events[i-1]['timestamp'] for i in range(1, len(self.hit_events))]
            self.avg_reaction_time = np.mean(reaction_times) if reaction_times else 0
    
    def generate_report(self):
        """Generate a comprehensive guitar analysis report"""
        if not self.start_time or not self.end_time: return None
        duration = (self.end_time - self.start_time).total_seconds()
        avg_speed = np.mean(self.hand_velocities) if self.hand_velocities else 0
        total_hits, total_misses = len(self.hit_events), len(self.miss_events)
        hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
        
        return {
            'session_info': {
                'date': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_played': f"{int(duration // 60)}:{int(duration % 60):02d}",
                'duration_target': f"{int(self.duration_target // 60)}:{int(self.duration_target % 60):02d}",
                'difficulty': f"{self.difficulty_level}",
                'final_score': self.final_score,
                'calories_burned': f"{self.calories_burned:.1f} kcal"
            },
            'movement_metrics': {
                'total_distance': f"{self.total_hand_distance:.1f} pixels",
                'avg_speed': f"{avg_speed:.1f} px/s",
                'max_speed': f"{self.max_hand_speed:.1f} px/s",
                'frames_tracked': self.frame_count
            },
            'performance_metrics': {
                'total_hits': total_hits, 'total_misses': total_misses, 'hit_rate': f"{hit_rate:.1f}%",
                'avg_reaction_time': f"{self.avg_reaction_time:.2f}s"
            },
            'range_of_motion': {
                'horizontal_reach': f"{self.max_reach_left + self.max_reach_right:.1f} pixels",
                'vertical_reach': f"{self.max_reach_up + self.max_reach_down:.1f} pixels"
            },
            'health_insights': self._generate_health_insights(duration, avg_speed, hit_rate)
        }

    def _generate_health_insights(self, duration, avg_speed, hit_rate):
        insights = []
        if duration >= 600: insights.append("🌟 Excellent stamina! 10+ minutes of continuous play is great for cardio.")
        elif duration >= 300: insights.append("✓ Great session! You maintained focus for 5+ minutes.")
        if avg_speed > 200: insights.append("🔥 High intensity! Your movements are fast and energetic.")
        if hit_rate > 90: insights.append("🎯 Incredible precision! Your hand-eye coordination is top-tier.")
        insights.append(f"🔥 Estimated {self.calories_burned:.1f} calories burned during this session.")
        return insights

    def save_to_json(self, filename=None):
        if filename is None:
            timestamp = self.start_time.strftime('%Y%m%d_%H%M%S') if self.start_time else 'unknown'
            filename = f"guitar_report_{timestamp}.json"
        report = self.generate_report()
        if report:
            try:
                with open(filename, 'w') as f: json.dump(report, f, indent=2)
                return filename
            except Exception as e: print(f"Error saving JSON report: {e}")
        return None

    def save_to_excel(self, filename="guitar_analytics.xlsx"):
        """Append all metrics for this session as a single row to the cumulative Excel file."""
        report = self.generate_report()
        if not report:
            return None
        try:
            # Flatten all sections into one dict
            row = {}
            row.update(report['session_info'])
            row.update(report['movement_metrics'])
            row.update(report['performance_metrics'])
            row.update(report['range_of_motion'])
            # Join health insights into a single readable string
            row['health_insights'] = " | ".join(report['health_insights'])

            new_df = pd.DataFrame([row])

            if os.path.exists(filename):
                # Load existing data and append
                existing_df = pd.read_excel(filename)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            combined_df.to_excel(filename, index=False)
            return filename
        except Exception as e:
            print(f"Error saving Excel report: {e}")
        return None

    def get_summary_text(self):
        report = self.generate_report()
        if not report: return "No data available"
        lines = [
            "G U I T A R   P E R F O R M A N C E   A N A L Y S I S",
            "─"*50,
            f"Date: {report['session_info']['date']}",
            f"Difficulty: {report['session_info']['difficulty']}",
            f"Final Score: {report['session_info']['final_score']}",
            f"Duration: {report['session_info']['duration_played']}",
            "─"*50,
            f"Total Hits: {report['performance_metrics']['total_hits']}",
            f"Hit Rate: {report['performance_metrics']['hit_rate']}",
            f"Avg Speed: {report['movement_metrics']['avg_speed']}",
            f"Calories: {report['session_info']['calories_burned']}",
            "─"*50,
            "Health Insights:"
        ]
        for insight in report['health_insights']: lines.append(f"• {insight}")
        return "\n".join(lines)

# ============================================================================
# ENUMERATIONS
# ============================================================================
class GameState(Enum):
    LOADING = 0
    HOME = 1
    SETTINGS = 2
    INSTRUCTIONS = 3
    CALIBRATION = 4
    PLAYING = 5
    PAUSED = 6
    GAME_OVER = 7
    REPORT = 8

class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    EXPERT = 4

class NoteType(Enum):
    STANDARD = 1

# ============================================================================
# MODERN UI UTILITIES
# ============================================================================
class ModernUI:
    """Modern UI rendering utilities with juggling game aesthetics"""
    
    @staticmethod
    def draw_gradient_background(surface, color1=(10, 20, 40), color2=(30, 60, 90)):
        """Draw a gradient background like juggling game"""
        width, height = surface.get_size()
        for y in range(height):
            ratio = y / height
            r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
            g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
            b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
            pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
    
    @staticmethod
    def draw_glass_panel(surface, rect, base_color=(0, 0, 0, 180), border_color=(0, 255, 245), border_width=2, radius=20):
        """Draw a glass panel similar to juggling game"""
        x, y, w, h = rect
        panel_surf = pygame.Surface((w, h), pygame.SRCALPHA)
        
        # Main background
        pygame.draw.rect(panel_surf, base_color, (0, 0, w, h), border_radius=radius)
        
        # Top highlight
        highlight = pygame.Surface((w, h//3), pygame.SRCALPHA)
        highlight.fill((255, 255, 255, 30))
        panel_surf.blit(highlight, (0, 0))
        
        # Border
        pygame.draw.rect(panel_surf, border_color, (0, 0, w, h), border_radius=radius, width=border_width)
        
        surface.blit(panel_surf, (x, y))
    
    @staticmethod
    def draw_neon_text(surface, text, pos, font, color=(0, 255, 245), glow=True, shadow=True):
        """Draw text with neon glow effect like juggling game"""
        x, y = pos
        
        if glow:
            # Glow effect
            for i in range(3, 0, -1):
                glow_color = (color[0], color[1], color[2], 50)
                glow_surf = font.render(text, True, glow_color)
                surface.blit(glow_surf, (x - i, y - i))
                surface.blit(glow_surf, (x + i, y - i))
                surface.blit(glow_surf, (x - i, y + i))
                surface.blit(glow_surf, (x + i, y + i))
        
        if shadow:
            # Shadow
            shadow_surf = font.render(text, True, (0, 0, 0))
            surface.blit(shadow_surf, (x + 2, y + 2))
        
        # Main text
        main_text = font.render(text, True, color)
        surface.blit(main_text, (x, y))
        
        return main_text.get_rect(topleft=(x, y))
    
    @staticmethod
    def create_gradient_surface(width, height, color1, color2, vertical=True):
        """Create a gradient surface for buttons"""
        surf = pygame.Surface((width, height), pygame.SRCALPHA)
        if vertical:
            for y in range(height):
                ratio = y / height
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surf, (r, g, b), (0, y), (width, y))
        else:
            for x in range(width):
                ratio = x / width
                r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
                g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
                b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
                pygame.draw.line(surf, (r, g, b), (x, 0), (x, height))
        return surf
    
    @staticmethod
    def draw_animated_background(surface, time_offset=0):
        """Draw an animated gradient background with moving elements"""
        width, height = surface.get_size()
        
        # Base gradient
        for y in range(height):
            ratio = y / height
            # Deep space colors
            r = int(15 + 25 * ratio + 10 * math.sin(time_offset * 0.5 + ratio * 3))
            g = int(20 + 35 * ratio + 15 * math.sin(time_offset * 0.7 + ratio * 2))
            b = int(45 + 60 * ratio + 20 * math.sin(time_offset * 0.3 + ratio * 4))
            pygame.draw.line(surface, (r, g, b), (0, y), (width, y))
        
        # Animated stars/particles
        for i in range(50):
            star_x = (i * 137 + time_offset * 20) % width
            star_y = (i * 213) % height
            star_size = 1 + (i % 3)
            alpha = int(150 + 105 * math.sin(time_offset * 2 + i))
            star_surf = pygame.Surface((star_size*2, star_size*2), pygame.SRCALPHA)
            pygame.draw.circle(star_surf, (255, 255, 255, alpha), (star_size, star_size), star_size)
            surface.blit(star_surf, (star_x, star_y))

# ============================================================================
# CONFIGURATION & SETTINGS
# ============================================================================
class Config:
    ASSETS_DIR = "asset"
    BACKGROUND_IMAGE = resource_path(f"{ASSETS_DIR}/background.png")
    PLAY_BUTTON_IMAGE = resource_path(f"{ASSETS_DIR}/play_button.png")
    EXIT_BUTTON_IMAGE = resource_path(f"{ASSETS_DIR}/exit_button.png")
    BLUE_BALL_IMAGE = resource_path(f"{ASSETS_DIR}/blue_ball.png")
    BG_PNG = resource_path(f"{ASSETS_DIR}/bg.png")
    LOGO_PNG = resource_path(f"{ASSETS_DIR}/logo.png")
    BLUE_HIT_SOUND = resource_path(f"{ASSETS_DIR}/blue_hit.mp3")
    HOME_IMAGE = resource_path(f"{ASSETS_DIR}/HOME.png")
    SETTINGS_BG = resource_path(f"{ASSETS_DIR}/background.jpg")
    CLICK_SOUND = resource_path(f"{ASSETS_DIR}/click.mp3")
    SCREEN_WIDTH = 1280
    SCREEN_HEIGHT = 720
    FPS = 60

    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_INDEX = 0

    NUM_LANES = 4
    INITIAL_SPEED = 3.0
    HIT_BOX_TOLERANCE = 60
    PINCH_THRESHOLD = 0.05
    CALIBRATION_DURATION = 3.0

    DIFFICULTY_SETTINGS = {
        Difficulty.EASY:    {'speed_multiplier': 0.7,  'spawn_multiplier': 1.3, 'acceleration': 0.005, 'name': 'Easy',   'color': (100, 255, 100)},
        Difficulty.MEDIUM:  {'speed_multiplier': 1.0,  'spawn_multiplier': 1.0, 'acceleration': 0.01,  'name': 'Medium', 'color': (255, 255, 100)},
        Difficulty.HARD:    {'speed_multiplier': 1.3,  'spawn_multiplier': 0.8, 'acceleration': 0.015, 'name': 'Hard',   'color': (255, 165, 0)},
        Difficulty.EXPERT:  {'speed_multiplier': 1.6,  'spawn_multiplier': 0.6, 'acceleration': 0.02,  'name': 'Expert', 'color': (255, 50, 50)}
    }

    TIME_OPTIONS = [120, 180, 240, 300,360,420, 480,540, 600,660, 720,780,840, 900,960,1020,1080,1140, 1200]
    STANDARD_NOTE_COLOR = (40, 140, 255)
    BG_COLOR = (20, 20, 35)
    LANE_COLOR_BASE = (60, 60, 80)
    HIT_ZONE_COLOR = (255, 215, 0)
    STAR_COLOR = (255, 255, 100)

    THEME_COLORS = {
        1: (100, 150, 200),
        2: (100, 150, 255),
        3: (150, 100, 255),
        4: (255, 215, 0)
    }

    MULTIPLIER_THRESHOLDS = {1: 0, 2: 11, 3: 26, 4: 50}
    PARTICLES_PER_HIT = 18
    PARTICLE_LIFETIME = 0.5
    PARTICLE_GRAVITY = 500.0
    VANISHING_POINT_Y = 100
    HORIZON_Y = 150
    BASE_NOTE_SIZE = 38

    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    HAND_CONFIDENCE = 0.7

    SPAWN_INTERVAL_MIN = 0.3
    SPAWN_INTERVAL_MAX = 1.2
    LANE_LINE_WIDTH = 12

# ============================================================================
# ASSET MANAGER
# ============================================================================
class AssetManager:
    def __init__(self):
        self.images = {}
        self.sounds = {}
        print("\n" + "="*60)
        print("ASSET LOADING")
        print("="*60)
        self.load_assets()
        print("="*60 + "\n")

    def load_assets(self):
        try:
            pygame.mixer.init()
            print("✓ Sound system initialized")
        except Exception as e:
            print(f"⚠ Sound init failed: {e}")

        for name, path in [
            ('background', Config.BACKGROUND_IMAGE),
            ('play_button', Config.PLAY_BUTTON_IMAGE),
            ('exit_button', Config.EXIT_BUTTON_IMAGE),
            ('blue_ball', Config.BLUE_BALL_IMAGE),
            ('bg_png', Config.BG_PNG),
            ('logo', Config.LOGO_PNG),
            ('home_img', Config.HOME_IMAGE),
            ('settings_bg', Config.SETTINGS_BG),
            ('instruct', resource_path(f"{Config.ASSETS_DIR}/instruct.png")),
        ]:
            try:
                img = pygame.image.load(path).convert_alpha()
                if name == 'background':
                    img = pygame.transform.scale(img, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
                elif name == 'home_img':
                    img = pygame.transform.scale(img, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
                elif name == 'settings_bg':
                    img = pygame.transform.scale(img, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
                elif name == 'play_button':
                    img = pygame.transform.scale(img, (320, 100))
                elif name == 'exit_button':
                    img = pygame.transform.scale(img, (320, 100))
                elif name == 'instruct':
                    img = pygame.transform.scale(img, (Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
                self.images[name] = img
                print(f"✓ Loaded {name}")
            except:
                print(f"⚠ {name} not found")
                self.images[name] = None

        try:
            sound = pygame.mixer.Sound(Config.BLUE_HIT_SOUND)
            self.sounds['blue_hit'] = sound
            print("✓ Loaded blue hit sound")
        except:
            print("⚠ No blue hit sound")
            self.sounds['blue_hit'] = None

        try:
            sound = pygame.mixer.Sound(Config.CLICK_SOUND)
            self.sounds['click'] = sound
            print("✓ Loaded click sound")
        except:
            print("⚠ No click sound")
            self.sounds['click'] = None

    def get_note_image(self, size: int) -> pygame.Surface:
        size = max(12, int(size))
        if self.images.get('blue_ball'):
            return pygame.transform.smoothscale(self.images['blue_ball'], (size * 4, size * 4))

        surf = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
        for i in range(6, 0, -1):
            r = min(255, 40 + i*35)
            g = min(255, 100 + i*25)
            b = min(255, 220 + i*10)
            pygame.draw.circle(surf, (r, g, b, 50 + i*30), (size, size), size + i*2 - 2)
        pygame.draw.circle(surf, Config.STANDARD_NOTE_COLOR, (size, size), size)
        pygame.draw.circle(surf, (140, 200, 255), (size - size//3, size - size//3), size//2 + 2)
        pygame.draw.circle(surf, (220, 240, 255), (size, size), size, 2)
        return surf

    def play_sound(self):
        if self.sounds.get('blue_hit'):
            try:
                self.sounds['blue_hit'].play()
            except:
                pass

    def play_click(self):
        if self.sounds.get('click'):
            try:
                self.sounds['click'].play()
            except:
                pass

# ============================================================================
# MODERN UI COMPONENTS
# ============================================================================
class ModernButton:
    def __init__(self, x, y, w, h, text="", primary=True):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.primary = primary
        self.is_hovered = False
        self.click_animation = 0
        self.hover_animation = 0
        self.font = pygame.font.Font(None, 32 if w < 200 else 36)
        
        if primary:
            self.base_color = (0, 200, 180)
            self.hover_color = (0, 220, 200)
            self.border_color = (0, 255, 245)
            self.text_color = (255, 255, 255)
        else:
            self.base_color = (80, 80, 100)
            self.hover_color = (100, 100, 120)
            self.border_color = (136, 136, 136)
            self.text_color = (204, 204, 204)

    def update(self, mouse_pos, dt):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        if self.is_hovered:
            self.hover_animation = min(1.0, self.hover_animation + dt * 8)
        else:
            self.hover_animation = max(0.0, self.hover_animation - dt * 8)
        
        if self.click_animation > 0:
            self.click_animation = max(0, self.click_animation - dt * 5)

    def draw(self, surface):
        scale = 1.0 + self.hover_animation * 0.02 - self.click_animation * 0.02
        scaled_w = int(self.rect.w * scale)
        scaled_h = int(self.rect.h * scale)
        scaled_rect = pygame.Rect(0, 0, scaled_w, scaled_h)
        scaled_rect.center = self.rect.center
        
        btn_surf = pygame.Surface((scaled_w, scaled_h), pygame.SRCALPHA)
        
        # Color interpolation
        r = int(self.base_color[0] + (self.hover_color[0] - self.base_color[0]) * self.hover_animation)
        g = int(self.base_color[1] + (self.hover_color[1] - self.base_color[1]) * self.hover_animation)
        b = int(self.base_color[2] + (self.hover_color[2] - self.base_color[2]) * self.hover_animation)
        color = (r, g, b)
        
        # Background
        pygame.draw.rect(btn_surf, color, (0, 0, scaled_w, scaled_h), border_radius=15)
        
        # Border
        border_color = (min(255, r + 50), min(255, g + 50), min(255, b + 50))
        pygame.draw.rect(btn_surf, border_color, (0, 0, scaled_w, scaled_h), 
                        border_radius=15, width=2)
        
        surface.blit(btn_surf, scaled_rect)
        
        # Text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=(scaled_rect.centerx, scaled_rect.centery))
        surface.blit(text_surf, text_rect)

    def is_clicked(self, mouse_pos, pressed):
        if self.is_hovered and pressed:
            self.click_animation = 0.5
            return True
        return False

class ModernSlider:
    def __init__(self, x, y, width, options, default_index=0, label=""):
        self.x = x
        self.y = y
        self.width = width
        self.options = options
        self.current_index = default_index
        self.label = label
        self.font = pygame.font.Font(None, 36)
        self.label_font = pygame.font.Font(None, 28)
        
        bw = 50
        self.left = ModernButton(x-60, y-5, bw, 45, "<", primary=False)
        self.right = ModernButton(x+width+10, y-5, bw, 45, ">", primary=False)
        self.value_animation = 0

    def update(self, mouse_pos, dt):
        self.left.update(mouse_pos, dt)
        self.right.update(mouse_pos, dt)
        
        if self.value_animation > 0:
            self.value_animation = max(0, self.value_animation - dt * 5)

    def handle_click(self, mouse_pos, pressed):
        if self.left.is_clicked(mouse_pos, pressed):
            if self.current_index > 0:
                self.current_index -= 1
                self.value_animation = 1.0
                return True
        if self.right.is_clicked(mouse_pos, pressed):
            if self.current_index < len(self.options)-1:
                self.current_index += 1
                self.value_animation = 1.0
                return True
        return False

    def draw(self, surface):
        # Label — large dark green, matching frog game style
        if self.label:
            label_font = pygame.font.Font(None, 58)
            label_surf = label_font.render(self.label, True, (255, 255, 255))
            surface.blit(label_surf, (self.x + self.width//2 - label_surf.get_width()//2, self.y - 50))
        
        # Panel background — narrower box centered within label
        panel_w = 260
        panel_rect = (self.x + (self.width - panel_w)//2, self.y, panel_w, 35)
        ModernUI.draw_glass_panel(surface, panel_rect, (0, 0, 0, 180), (100, 100, 120), 1, 10)
        
        # Value text
        if isinstance(self.options[self.current_index], Difficulty):
            val_text = self.options[self.current_index].name
            colors = {Difficulty.EASY: (100, 255, 100), Difficulty.MEDIUM: (255, 255, 100), 
                      Difficulty.HARD: (255, 165, 0), Difficulty.EXPERT: (255, 50, 50)}
            val_color = colors[self.options[self.current_index]]
        else:
            val_text = str(self.options[self.current_index])
            val_color = (0, 255, 245)
        
        scale = 1.0 + self.value_animation * 0.1
        scaled_font = pygame.font.Font(None, int(30 * scale))
        
        text_surf = scaled_font.render(val_text, True, val_color)
        text_x = self.x + self.width//2 - text_surf.get_width()//2
        text_y = self.y + 18 - text_surf.get_height()//2
        surface.blit(text_surf, (text_x, text_y))
        
        self.left.draw(surface)
        self.right.draw(surface)

    def get_value(self):
        return self.options[self.current_index]

# ============================================================================
# DATA CLASSES
# ============================================================================
@dataclass
class Note:
    lane: int
    x: float = 0.0
    y: float = 0.0
    note_type: NoteType = NoteType.STANDARD
    size: float = Config.BASE_NOTE_SIZE
    speed: float = 3.0
    active: bool = True

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    size: int
    color: Tuple[int, int, int]

# ============================================================================
# CALIBRATION
# ============================================================================
class Calibration:
    def __init__(self):
        self.is_calibrated = False
        self.start_time = None
        self.duration = Config.CALIBRATION_DURATION
        self.min_hand_size_norm = 0.15
        self.consecutive_good_frames = 0
        self.required_frames = int(Config.FPS * self.duration)

    def start(self):
        self.is_calibrated = False
        self.start_time = time.time()
        self.consecutive_good_frames = 0

    def get_hand_bounding_box(self, landmarks, frame_w: int, frame_h: int):
        if not landmarks:
            return None, 0.0

        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        norm_size = max(x_max - x_min, y_max - y_min)

        px_min_x = int(x_min * frame_w)
        px_min_y = int(y_min * frame_h)
        px_w = int((x_max - x_min) * frame_w)
        px_h = int((y_max - y_min) * frame_h)

        return (px_min_x, px_min_y, px_w, px_h), norm_size

    def update(self, hand_detected: bool, landmarks=None, frame_w=640, frame_h=480):
        if not hand_detected or not landmarks:
            self.consecutive_good_frames = 0
            return 0.0, False, "Show your hand to the camera", None

        bbox, norm_size = self.get_hand_bounding_box(landmarks, frame_w, frame_h)
        is_good_size = norm_size >= self.min_hand_size_norm

        if is_good_size:
            self.consecutive_good_frames += 1
            if self.start_time is None:
                self.start_time = time.time()
        else:
            self.consecutive_good_frames = 0
            self.start_time = None

        progress = 0.0
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            progress = min(elapsed / self.duration, 1.0)
            if progress >= 1.0:
                self.is_calibrated = True

        message = "Move closer • Hand too small" if not is_good_size else f"Hold steady... {int(progress*100)}%"
        return progress, is_good_size, message, bbox

    def draw_hand_box(self, frame: np.ndarray, bbox, is_good: bool):
        if bbox is None:
            return frame
        x, y, w, h = bbox
        color = (0, 220, 0) if is_good else (0, 0, 220)
        thick = 5 if is_good else 3
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thick)
        return frame

# ============================================================================
# HAND TRACKER
# ============================================================================
class HandTracker:
    FINGER_COLORS = [
        (100, 200, 255), # Index - Light Blue (RGB)
        (0, 255, 100),   # Middle
        (255, 100, 255), # Ring
        (255, 0, 0)      # Pinky - Red (RGB)
    ]

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=Config.HAND_CONFIDENCE,
            min_tracking_confidence=Config.HAND_CONFIDENCE
        )
        self.detected = False
        self.pinch_lane = None
        self.distances = [0.0]*4
        self.landmarks = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if res.multi_hand_landmarks:
            self.landmarks = res.multi_hand_landmarks[0]
            lm = self.landmarks.landmark
            thumb = lm[Config.THUMB_TIP]
            tips = [lm[i] for i in [Config.INDEX_TIP, Config.MIDDLE_TIP, Config.RING_TIP, Config.PINKY_TIP]]
            self.distances = [math.hypot(t.x-thumb.x, t.y-thumb.y, t.z-thumb.z) for t in tips]
            self.pinch_lane = next((i for i,d in enumerate(self.distances) if d < Config.PINCH_THRESHOLD), None)
            self.detected = True
            return True
        self.detected = False
        self.pinch_lane = None
        self.landmarks = None
        return False

    def draw(self, frame):
        if self.landmarks:
            self.mp_draw.draw_landmarks(
                frame, self.landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(180,180,180), thickness=1),
                self.mp_draw.DrawingSpec(color=(120,120,120), thickness=1)
            )
            h, w, _ = frame.shape
            thumb_pos = (int(self.landmarks.landmark[Config.THUMB_TIP].x * w),
                         int(self.landmarks.landmark[Config.THUMB_TIP].y * h))

            for i, idx in enumerate([Config.INDEX_TIP, Config.MIDDLE_TIP, Config.RING_TIP, Config.PINKY_TIP]):
                tip = self.landmarks.landmark[idx]
                pos = (int(tip.x * w), int(tip.y * h))
                # Convert RGB to BGR for OpenCV
                col_rgb = self.FINGER_COLORS[i]
                col_bgr = (col_rgb[2], col_rgb[1], col_rgb[0])
                
                cv2.circle(frame, pos, 9, col_bgr, -1)
                cv2.circle(frame, pos, 11, (255,255,255), 1)
                if i == self.pinch_lane:
                    cv2.line(frame, thumb_pos, pos, col_bgr, 5)
                    cv2.circle(frame, thumb_pos, 8, col_bgr, -1)
                else:
                    cv2.line(frame, thumb_pos, pos, (60,60,60), 1)
        return frame

    def get_pinch_lane(self):
        return self.pinch_lane

    def is_pinching(self):
        return self.pinch_lane is not None

    def cleanup(self):
        self.hands.close()

# ============================================================================
# NOTE MANAGER
# ============================================================================
class NoteManager:
    def __init__(self, difficulty: Difficulty, assets: AssetManager):
        self.notes: List[Note] = []
        self.assets = assets
        self.diff = difficulty
        self.sett = Config.DIFFICULTY_SETTINGS[difficulty]
        self.last_spawn = 0.0
        self.spawn_interval = Config.SPAWN_INTERVAL_MAX * self.sett['spawn_multiplier']
        self.speed = Config.INITIAL_SPEED * self.sett['speed_multiplier']
        self.accel = self.sett['acceleration']
        self.lane_x = [int(Config.SCREEN_WIDTH / (Config.NUM_LANES+1) * (i+1)) for i in range(Config.NUM_LANES)]

    def adjust_difficulty(self, mult: int):
        base = Config.SPAWN_INTERVAL_MAX * self.sett['spawn_multiplier']
        self.spawn_interval = max(Config.SPAWN_INTERVAL_MIN, base / (1 + 0.28 * mult))

    def get_x_at_y(self, lane: int, y: float) -> float:
        vp_x = Config.SCREEN_WIDTH // 2
        bottom_x = self.lane_x[lane]
        start_x = vp_x + (bottom_x - vp_x) * 0.3
        prog = (y - Config.VANISHING_POINT_Y) / (Config.SCREEN_HEIGHT - Config.VANISHING_POINT_Y)
        return start_x + (bottom_x - start_x) * prog

    def spawn(self):
        lane = random.randint(0, Config.NUM_LANES-1)
        y = Config.VANISHING_POINT_Y
        x = self.get_x_at_y(lane, y)
        note = Note(lane=lane, x=x, y=y, note_type=NoteType.STANDARD,
                    size=Config.BASE_NOTE_SIZE, speed=self.speed)
        self.notes.append(note)

    def update(self, dt: float, game_time: float):
        self.speed = Config.INITIAL_SPEED * self.sett['speed_multiplier'] + self.accel * game_time
        for n in self.notes:
            n.y += n.speed * dt * 60
            n.x = self.get_x_at_y(n.lane, n.y)
        self.notes = [n for n in self.notes if n.y < Config.SCREEN_HEIGHT + 120]

        now = time.time()
        if now - self.last_spawn > self.spawn_interval:
            self.spawn()
            self.last_spawn = now

    def check_hit(self, lane: Optional[int]) -> Optional[Note]:
        if lane is None: return None
        hit_y = Config.SCREEN_HEIGHT - 100
        for n in self.notes:
            if n.active and abs(n.y - hit_y) < Config.HIT_BOX_TOLERANCE:
                if n.lane == lane:
                    n.active = False
                    return n
        return None

    def render(self, surf: pygame.Surface):
        for n in self.notes:
            if not n.active: continue
            img = self.assets.get_note_image(n.size)
            r = img.get_rect(center=(int(n.x), int(n.y)))
            surf.blit(img, r)

    def clear(self):
        self.notes.clear()

# ============================================================================
# PARTICLE EMITTER
# ============================================================================
class ParticleEmitter:
    def __init__(self):
        self.particles: List[Particle] = []

    def emit(self, x, y, color):
        for _ in range(Config.PARTICLES_PER_HIT):
            a = random.uniform(0, math.tau)
            s = random.uniform(90, 280)
            self.particles.append(Particle(
                x=x, y=y,
                vx=math.cos(a)*s, vy=math.sin(a)*s - 120,
                life=Config.PARTICLE_LIFETIME,
                max_life=Config.PARTICLE_LIFETIME,
                size=random.randint(3,8),
                color=color
            ))

    def update(self, dt):
        for p in self.particles[:]:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += Config.PARTICLE_GRAVITY * dt
            p.life -= dt
            if p.life <= 0:
                self.particles.remove(p)

    def render(self, surf):
        for p in self.particles:
            a = int(255 * (p.life / p.max_life))
            s = pygame.Surface((p.size*2, p.size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*p.color, a), (p.size, p.size), p.size)
            surf.blit(s, (int(p.x - p.size), int(p.y - p.size)))

    def clear(self):
        self.particles.clear()

# ============================================================================
# DIFFICULTY / SCORE MANAGER
# ============================================================================
class DifficultyManager:
    def __init__(self):
        self.score = 0
        self.streak = 0
        self.max_streak = 0
        self.multiplier = 1
        self.hits = 0

    def hit(self) -> int:
        points = 100 * self.multiplier
        self.score += points
        self.streak += 1
        self.max_streak = max(self.max_streak, self.streak)
        self.hits += 1
        self._update_mult()
        return points

    def miss(self):
        self.streak = 0
        self.multiplier = 1

    def _update_mult(self):
        for m, thresh in sorted(Config.MULTIPLIER_THRESHOLDS.items(), reverse=True):
            if self.streak >= thresh:
                self.multiplier = m
                break

    def get_theme_color(self):
        return Config.THEME_COLORS.get(self.multiplier, (100,150,200))

    def reset(self):
        self.score = 0
        self.streak = 0
        self.max_streak = 0
        self.multiplier = 1
        self.hits = 0

# ============================================================================
# MODERN HUD
# ============================================================================
class ModernHUD:
    def __init__(self):
        pygame.font.init()
        self.big = pygame.font.Font(None, 82)
        self.med = pygame.font.Font(None, 56)
        self.small = pygame.font.Font(None, 42)
        self.feedback = []
        # Move QUIT to top-left below the Score/Time panel
        self.quit_button = ModernButton(25, 145, 130, 60, "QUIT", primary=False)

    def render(self, surf, diff: DifficultyManager, time_val: float, hand_ok: bool):
        # 1. Combined Score & Time Panel (Frog Style)
        panel_rect = (25, 25, 240, 110)
        ModernUI.draw_glass_panel(surf, panel_rect, (0, 0, 0, 180), (0, 255, 245), 1, 15)
        
        # Score - Green
        score_font = pygame.font.Font(None, 44)
        ModernUI.draw_neon_text(surf, f"SCORE: {diff.score}",
                               (40, 35), score_font, (0, 255, 100), False)
        
        # Time - Cyan
        time_font = pygame.font.Font(None, 44)
        tstr = f"{int(time_val//60):02d}:{int(time_val%60):02d}"
        ModernUI.draw_neon_text(surf, f"TIME:  {tstr}",
                               (40, 80), time_font, (0, 255, 245), False)

        # 2. Streak - Separate box below QUIT
        streak_text = f"Streak: {diff.streak}"
        st_w = self.small.size(streak_text)[0] + 30
        ModernUI.draw_glass_panel(surf, (25, 215, st_w, 50), (35, 35, 55, 200), (0, 255, 245), 1, 12)
        st_surf = self.small.render(streak_text, True, (210, 230, 255))
        surf.blit(st_surf, (40, 228))

        # 3. Quit button
        self.quit_button.draw(surf)

    def update(self, dt, mouse_pos):
        for f in self.feedback[:]:
            f['life'] -= dt
            f['y'] -= 60 * dt
            if f['life'] <= 0:
                self.feedback.remove(f)
        
        self.quit_button.update(mouse_pos, dt)

    def render_feedback(self, surf):
        for f in self.feedback:
            a = int(255 * (f['life']/f['max']))
            scale = 1.0 + (1.0 - f['life']/f['max']) * 0.3
            scaled_font = pygame.font.Font(None, int(56 * scale))
            ModernUI.draw_neon_text(surf, f['text'], 
                                   (f['x'] - scaled_font.size(f['text'])[0]//2, f['y']),
                                   scaled_font, (240, 240, 100), True)

    def add_feedback(self, text, x, y):
        self.feedback.append({'text':text, 'x':x, 'y':y, 'life':1.0, 'max':1.0})

    def check_quit_clicked(self, mouse_pos, pressed):
        return self.quit_button.is_clicked(mouse_pos, pressed)

# ============================================================================
# MAIN GAME
# ============================================================================
class RhythmHeroAR:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("Rhythm Hero AR - Modern UI")
        self.clock = pygame.time.Clock()

        self.state = GameState.LOADING
        self.assets = None
        self.loading_progress = 0.0
        self.loading_start = time.time()
        self.animation_time = 0.0

        self.selected_diff = Difficulty.MEDIUM
        self.selected_time = 120

        self.camera = None
        self.calib = None
        self.tracker = None
        self.notes = None
        self.particles = None
        self.score_mgr = None
        self.hud = None
        self.analytics = GuitarAnalytics()

        self.home_buttons = []
        self.settings_buttons = []
        self.instr_buttons = []
        self.gameover_buttons = []

        self.diff_slider = None
        self.time_slider = None

        self.running = True
        self.paused = False
        self.game_start = 0.0
        self.camera_surf = None
        self.mouse_clicked = False
        
        self.calib_progress = 0.0
        self.calib_is_good = False
        self.calib_message = ""
        
        # UI Elements
        self.play_btn = None
        self.exit_btn = None
        self.play_btn_rect = None
        self.exit_btn_rect = None
        self._play_hover = 0.0
        self._exit_hover = 0.0
        
        self.next_btn = None
        self.settings_back_btn = None
        
        self.start_btn = None
        self.instr_back_btn = None
        self.calib_back_btn = None
        
        self.play_again_btn: Optional[ModernButton] = None
        self.menu_exit_btn: Optional[ModernButton] = None
        self.view_report_btn: Optional[ModernButton] = None
        self.report_back_btn: Optional[ModernButton] = None

    def create_home_ui(self):
        cx = Config.SCREEN_WIDTH // 2
        cy = Config.SCREEN_HEIGHT // 2
        # Play button: slightly below center
        self.play_btn_rect = pygame.Rect(cx - 160, cy + 50, 320, 100)
        # Exit button: top-right corner
        self.exit_btn_rect = pygame.Rect(Config.SCREEN_WIDTH - 180, 20, 160, 60)
        # Hover animation states (0.0 to 1.0)
        self._play_hover = 0.0
        self._exit_hover = 0.0
        # Fallback text buttons
        self.play_btn = ModernButton(cx-180, cy+50, 360, 90, "START GAME", primary=True)
        self.exit_btn = ModernButton(Config.SCREEN_WIDTH - 180, 20, 160, 60, "EXIT", primary=False)
        self.home_buttons = [self.play_btn, self.exit_btn]

    def create_settings_ui(self):
        cx = Config.SCREEN_WIDTH // 2
        self.diff_slider = ModernSlider(cx-200, 270, 400, list(Difficulty), 1, "DIFFICULTY")
        times = [f"{t//60}:{t%60:02d}" for t in Config.TIME_OPTIONS]
        self.time_slider = ModernSlider(cx-200, 400, 400, times, 2, "TIME LIMIT")
        self.next_btn = ModernButton(cx-110, 530, 220, 70, "NEXT", primary=True)
        self.settings_back_btn = ModernButton(60, Config.SCREEN_HEIGHT-90, 160, 60, "BACK", primary=False)
        self.settings_buttons = [self.next_btn, self.settings_back_btn]

    def create_instructions_ui(self):
        self.start_btn = ModernButton(Config.SCREEN_WIDTH//2-130, Config.SCREEN_HEIGHT-130, 260, 80, "START", primary=True)
        self.instr_back_btn = ModernButton(60, Config.SCREEN_HEIGHT-90, 160, 60, "BACK", primary=False)
        self.instr_buttons = [self.start_btn, self.instr_back_btn]
        self.calib_back_btn = ModernButton(60, Config.SCREEN_HEIGHT-90, 160, 60, "BACK", primary=False)

    def create_gameover_ui(self):
        cx = Config.SCREEN_WIDTH // 2
        self.play_again_btn = ModernButton(cx-280, 500, 240, 80, "PLAY AGAIN", primary=True)
        self.menu_exit_btn = ModernButton(cx+40, 500, 240, 80, "MENU", primary=False)
        self.view_report_btn = ModernButton(Config.SCREEN_WIDTH-260, 20, 240, 60, "VIEW REPORT", primary=True)
        self.report_back_btn = ModernButton(Config.SCREEN_WIDTH//2-120, Config.SCREEN_HEIGHT-100, 240, 60, "BACK", primary=False)
        self.gameover_buttons = [self.play_again_btn, self.menu_exit_btn, self.view_report_btn]

    def _get_camera(self):
        """Attempts to find an external camera (index 1), falls back to system camera (index 0)."""
        # Try external camera first
        for idx in [1, 0]:
            print(f"--- Attempting to open camera index {idx} ---")
            cam = cv2.VideoCapture(idx)
            if cam.isOpened():
                ret, frame = cam.read()
                if ret:
                    print(f"✓ Successfully opened camera at index {idx}")
                    # Apply configurations
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                    return cam
                else:
                    print(f"⚠ Camera at index {idx} opened but could not read frame.")
                    cam.release()
            else:
                print(f"⚠ Could not open camera at index {idx}")
        
        print("✖ No cameras found!")
        return None

    def _preinit_camera(self):
        """Open the camera in a background thread during loading so it's ready at calibration."""
        try:
            self.camera = self._get_camera()
            if self.camera:
                # Warm up: read a few frames so the device is fully active
                for _ in range(5):
                    self.camera.read()
                print("✓ Camera pre-initialised in background")
        except Exception as e:
            print(f"⚠ Camera pre-init failed: {e}")

    def init_game(self):
        # Reuse pre-opened camera if already available from loading phase
        if self.camera is None:
            self.camera = self._get_camera()

        self.calib = Calibration()
        self.tracker = HandTracker()
        self.notes = NoteManager(self.selected_diff, self.assets)
        self.particles = ParticleEmitter()
        self.score_mgr = DifficultyManager()
        self.hud = ModernHUD()
        self.analytics.start_session(self.selected_diff.name, self.selected_time)
        self.final_time_display = 0
        self.calib.start()
        self.game_start = time.time()

    def handle_events(self):
        mx, my = pygame.mouse.get_pos()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    if self.state == GameState.PLAYING:
                        self.state = GameState.GAME_OVER
                    elif self.state in (GameState.SETTINGS, GameState.INSTRUCTIONS, GameState.GAME_OVER):
                        self.state = GameState.HOME
                    else:
                        self.running = False
                elif e.key == pygame.K_SPACE:
                    if self.state == GameState.PLAYING:
                        self.paused = not self.paused
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                if time.time() - getattr(self, 'last_click', 0) > 0.18:
                    self.mouse_clicked = True
                    self.last_click = time.time()
                    self.assets.play_click()
                    self._handle_clicks(mx, my)

    def _handle_clicks(self, mx, my):
        if self.state == GameState.HOME:
            play_img = self.assets.images.get('play_button') if self.assets else None
            exit_img = self.assets.images.get('exit_button') if self.assets else None
            if play_img and exit_img:
                if self.play_btn_rect.collidepoint(mx, my): self.state = GameState.SETTINGS
                if self.exit_btn_rect.collidepoint(mx, my): self.running = False
            else:
                if self.play_btn.is_clicked((mx,my), True): self.state = GameState.SETTINGS
                if self.exit_btn.is_clicked((mx,my), True): self.running = False
        elif self.state == GameState.SETTINGS:
            if self.diff_slider.handle_click((mx,my), True): pass
            if self.time_slider.handle_click((mx,my), True): pass
            if self.next_btn.is_clicked((mx,my), True):
                self.selected_diff = self.diff_slider.get_value()
                idx = self.time_slider.current_index
                self.selected_time = Config.TIME_OPTIONS[idx]
                self.state = GameState.INSTRUCTIONS
            if self.settings_back_btn.is_clicked((mx,my), True):
                self.state = GameState.HOME
        elif self.state == GameState.INSTRUCTIONS:
            if self.start_btn.is_clicked((mx,my), True):
                self.init_game()
                self.state = GameState.CALIBRATION
            if self.instr_back_btn.is_clicked((mx,my), True):
                self.state = GameState.SETTINGS
        elif self.state == GameState.CALIBRATION:
            if self.calib_back_btn.is_clicked((mx,my), True):
                # Release camera and tracker resources, go back to instructions
                if self.tracker:
                    self.tracker.cleanup()
                    self.tracker = None
                self.state = GameState.INSTRUCTIONS
        elif self.state == GameState.PLAYING:
            if self.hud and self.hud.check_quit_clicked((mx,my), True):
                if self.score_mgr:
                    self.analytics.end_session(self.score_mgr.score)
                self.analytics.save_to_json()
                self.analytics.save_to_excel()
                self.state = GameState.GAME_OVER
        elif self.state == GameState.GAME_OVER:
            if self.play_again_btn and self.play_again_btn.is_clicked((mx, my), True):
                # Reset score and restart from calibration
                if self.score_mgr:
                    self.score_mgr.reset()
                self.notes.clear() if self.notes else None
                if self.particles:
                    self.particles.clear()
                self.init_game()
                self.state = GameState.CALIBRATION
            elif self.menu_exit_btn and self.menu_exit_btn.is_clicked((mx, my), True):
                self.state = GameState.HOME
            elif self.view_report_btn and self.view_report_btn.is_clicked((mx, my), True):
                self.state = GameState.REPORT
        elif self.state == GameState.REPORT:
            if self.report_back_btn and self.report_back_btn.is_clicked((mx, my), True):
                self.state = GameState.GAME_OVER

    def update(self, dt):
        self.animation_time += dt
        
        if self.state == GameState.LOADING:
            self._update_loading()
            return

        mx, my = pygame.mouse.get_pos()
        
        if self.state in (GameState.HOME, GameState.SETTINGS, GameState.INSTRUCTIONS, GameState.CALIBRATION, GameState.GAME_OVER, GameState.REPORT):
            if self.state == GameState.HOME:
                for b in self.home_buttons: b.update((mx,my), dt)
            elif self.state == GameState.SETTINGS:
                self.diff_slider.update((mx,my), dt)
                self.time_slider.update((mx,my), dt)
                for b in self.settings_buttons: b.update((mx,my), dt)
            elif self.state == GameState.INSTRUCTIONS:
                for b in self.instr_buttons: b.update((mx,my), dt)
            elif self.state == GameState.CALIBRATION:
                self.calib_back_btn.update((mx,my), dt)
            elif self.state == GameState.GAME_OVER:
                for b in self.gameover_buttons: b.update((mx,my), dt)
            elif self.state == GameState.REPORT:
                if self.report_back_btn: self.report_back_btn.update((mx,my), dt)
            if self.state != GameState.CALIBRATION:
                self.mouse_clicked = False
                return

        if self.camera:
            ret, frame = self.camera.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1)
                self.tracker.process(frame)
                frame = self.tracker.draw(frame)
                if self.state == GameState.PLAYING:
                    self.analytics.record_frame(self.tracker.landmarks, Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT)

                if self.state == GameState.CALIBRATION:
                    progress, is_good, message, bbox = self.calib.update(
                        self.tracker.detected,
                        self.tracker.landmarks,
                        Config.CAMERA_WIDTH,
                        Config.CAMERA_HEIGHT
                    )
                    self.calib_progress = progress
                    self.calib_is_good = is_good
                    self.calib_message = message
                    frame = self.calib.draw_hand_box(frame, bbox, is_good)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                small = cv2.resize(rgb, (320, 240))
                self.camera_surf = pygame.surfarray.make_surface(np.rot90(small))

        if self.state == GameState.CALIBRATION:
            if self.calib.is_calibrated:
                self.state = GameState.PLAYING
                self.game_start = time.time()
        elif self.state == GameState.PLAYING and not self.paused:
            self._update_playing(dt, (mx, my))

    def _update_loading(self):
        elapsed = time.time() - self.loading_start
        self.loading_progress = min(elapsed / 2.8, 1.0)
        if self.loading_progress >= 0.45 and self.assets is None:
            self.assets = AssetManager()
            self.create_home_ui()
            self.create_settings_ui()
            self.create_instructions_ui()
            self.create_gameover_ui()
            # Kick off camera init in the background so it's ready at calibration
            t = threading.Thread(target=self._preinit_camera, daemon=True)
            t.start()
        if self.loading_progress >= 1.0:
            self.state = GameState.HOME

    def _update_playing(self, dt, mouse_pos):
        elapsed = time.time() - self.game_start
        remaining = max(0, self.selected_time - elapsed)
        if remaining <= 0:
            if self.score_mgr:
                self.analytics.end_session(self.score_mgr.score)
            self.analytics.save_to_json()
            self.analytics.save_to_excel()
            self.state = GameState.GAME_OVER
            return

        self.notes.update(dt, elapsed)
        self.notes.adjust_difficulty(self.score_mgr.multiplier)
        self.particles.update(dt)
        self.hud.update(dt, mouse_pos)

        if self.tracker.detected and self.tracker.is_pinching():
            lane = self.tracker.get_pinch_lane()
            hit = self.notes.check_hit(lane)
            if hit:
                self.analytics.record_hit(lane)
                pts = self.score_mgr.hit()
                x = self.notes.lane_x[hit.lane]
                self.assets.play_sound()
                self.hud.add_feedback("PERFECT!" if self.score_mgr.multiplier > 2 else "GOOD!", x, Config.SCREEN_HEIGHT-90)
                self.particles.emit(x, Config.SCREEN_HEIGHT-100, (80, 160, 255))

        for n in self.notes.notes[:]:
            if n.active and n.y > Config.SCREEN_HEIGHT - 40:
                self.analytics.record_miss(n.lane)
                self.score_mgr.miss()
                n.active = False

    def render(self):
        if self.state == GameState.LOADING:
            self._render_loading()
        elif self.state == GameState.HOME:
            self._render_home()
        elif self.state == GameState.SETTINGS:
            self._render_settings()
        elif self.state == GameState.INSTRUCTIONS:
            self._render_instructions()
        elif self.state == GameState.CALIBRATION:
            self._render_calibration()
        elif self.state == GameState.PLAYING:
            self._render_playing()
        elif self.state == GameState.GAME_OVER:
            self._render_gameover()
        elif self.state == GameState.REPORT:
            self._render_report()

        if self.state == GameState.PLAYING and self.camera_surf:
            # Small camera preview only during gameplay
            cam_x, cam_y = Config.SCREEN_WIDTH-340, 20
            ModernUI.draw_glass_panel(self.screen, (cam_x-10, cam_y-10, 340, 260), (30, 40, 60), 220)
            self.screen.blit(self.camera_surf, (cam_x, cam_y))

        pygame.display.flip()

    def _render_loading(self):
        # Show HOME.png once assets are loaded; animated bg before that
        if self.assets and self.assets.images.get('home_img'):
            self.screen.blit(self.assets.images['home_img'], (0, 0))
            # Semi-transparent dark overlay so progress bar is readable over the image
            overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            self.screen.blit(overlay, (0, 0))
        else:
            ModernUI.draw_animated_background(self.screen, self.animation_time)

        # Title
        title_font = pygame.font.Font(None, 130)
        ModernUI.draw_neon_text(self.screen, "RHYTHM HERO AR",
                               (Config.SCREEN_WIDTH//2 - title_font.size("RHYTHM HERO AR")[0]//2, 180),
                               title_font, (120, 200, 255), True)

        sub_font = pygame.font.Font(None, 52)
        ModernUI.draw_neon_text(self.screen, "Modern Edition",
                               (Config.SCREEN_WIDTH//2 - sub_font.size("Modern Edition")[0]//2, 300),
                               sub_font, (180, 220, 255), True)

        # Modern progress bar
        bw, bh = 700, 50
        bx = (Config.SCREEN_WIDTH - bw)//2
        by = 420

        ModernUI.draw_glass_panel(self.screen, (bx, by, bw, bh), (40, 50, 70), 220)

        # Animated progress fill
        pw = int((bw - 10) * self.loading_progress)
        if pw > 0:
            prog_surf = ModernUI.create_gradient_surface(pw, bh-10, (80, 160, 255), (120, 200, 255), False)
            self.screen.blit(prog_surf, (bx + 5, by + 5))

            # Glow effect on progress
            glow_surf = pygame.Surface((pw, bh-10), pygame.SRCALPHA)
            pygame.draw.rect(glow_surf, (150, 200, 255, 100), (0, 0, pw, bh-10), border_radius=15)
            self.screen.blit(glow_surf, (bx + 5, by + 5))

        # Status text
        perc_font = pygame.font.Font(None, 44)
        status = "Initialising camera..." if (self.assets and self.loading_progress >= 1.0) else f"{int(self.loading_progress * 100)}%"
        ModernUI.draw_neon_text(self.screen, status,
                               (Config.SCREEN_WIDTH//2 - perc_font.size(status)[0]//2, by + bh + 25),
                               perc_font, (220, 240, 255), True)

    def _render_home(self):
        dt = self.clock.get_time() / 1000.0

        # Draw HOME.png as full background, or fall back to animated background
        home_img = self.assets.images.get('home_img')
        if home_img:
            self.screen.blit(home_img, (0, 0))
        else:
            ModernUI.draw_animated_background(self.screen, self.animation_time)

        mx, my = pygame.mouse.get_pos()
        play_img = self.assets.images.get('play_button')
        exit_img = self.assets.images.get('exit_button')

        if play_img and exit_img:
            # Smooth hover animation
            spd = 6.0
            self._play_hover = min(1.0, self._play_hover + spd * dt) if self.play_btn_rect.collidepoint(mx, my) else max(0.0, self._play_hover - spd * dt)
            self._exit_hover = min(1.0, self._exit_hover + spd * dt) if self.exit_btn_rect.collidepoint(mx, my) else max(0.0, self._exit_hover - spd * dt)

            for img, rect, hover in [
                (play_img, self.play_btn_rect, self._play_hover),
                (exit_img, self.exit_btn_rect, self._exit_hover),
            ]:
                # Scale up smoothly on hover (no overlays, no glow shapes)
                scale = 1.0 + hover * 0.07
                sw = int(rect.w * scale)
                sh = int(rect.h * scale)
                scaled_img = pygame.transform.smoothscale(img, (sw, sh))
                draw_x = rect.centerx - sw // 2
                draw_y = rect.centery - sh // 2
                self.screen.blit(scaled_img, (draw_x, draw_y))
        else:
            # Fallback: draw styled text buttons
            for b in self.home_buttons:
                b.draw(self.screen)

    def _render_settings(self):
        # Use background.jpg as the settings background
        settings_bg = self.assets.images.get('settings_bg')
        if settings_bg:
            self.screen.blit(settings_bg, (0, 0))
        else:
            ModernUI.draw_animated_background(self.screen, self.animation_time)
        
        # No heading, no glass panel — just sliders directly on background (frog style)
        self.diff_slider.draw(self.screen)
        self.time_slider.draw(self.screen)
        
        for b in self.settings_buttons:
            b.draw(self.screen)

    def _render_instructions(self):
        # Background: use instruct.png
        instruct_img = self.assets.images.get('instruct')
        if instruct_img:
            self.screen.blit(instruct_img, (0, 0))
        else:
            ModernUI.draw_animated_background(self.screen, self.animation_time)
            # Semi-transparent overlay as fallback
            overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))

        # Only show START button
        # Note: self.instr_buttons contains [self.start_btn, self.instr_back_btn]
        # User said "alone with the start button only", so I'll only draw start_btn
        self.start_btn.draw(self.screen)
        self.instr_back_btn.draw(self.screen)

    def _render_calibration(self):
        ModernUI.draw_animated_background(self.screen, self.animation_time)

        title_font = pygame.font.Font(None, 90)
        ModernUI.draw_neon_text(self.screen, "HAND CALIBRATION",
                               (Config.SCREEN_WIDTH//2 - title_font.size("HAND CALIBRATION")[0]//2, 50),
                               title_font, (220, 240, 255), True)

        if self.camera_surf:
            large = pygame.transform.scale(self.camera_surf, (720, 540))
            cx = (Config.SCREEN_WIDTH - 720) // 2
            cy = 140
            
            # Modern frame
            ModernUI.draw_glass_panel(self.screen, (cx-15, cy-15, 750, 570), (40, 50, 70), 220)
            self.screen.blit(large, (cx, cy))

        # Instructions
        instr_font = pygame.font.Font(None, 50)
        ModernUI.draw_neon_text(self.screen, "Show your hand clearly to the camera",
                               (Config.SCREEN_WIDTH//2 - instr_font.size("Show your hand clearly to the camera")[0]//2, 100),
                               instr_font, (180, 220, 255), True)

        # Status with glass panel
        msg_w = 600
        ModernUI.draw_glass_panel(self.screen, (Config.SCREEN_WIDTH//2 - msg_w//2, Config.SCREEN_HEIGHT - 110, msg_w, 80),
                                 (40, 40, 60), 220)
        
        msg_font = pygame.font.Font(None, 48)
        text_color = (100, 255, 150) if self.calib_is_good else (255, 100, 120)
        ModernUI.draw_neon_text(self.screen, self.calib_message,
                               (Config.SCREEN_WIDTH//2 - msg_font.size(self.calib_message)[0]//2, Config.SCREEN_HEIGHT - 85),
                               msg_font, text_color, True)

        # Back button
        self.calib_back_btn.draw(self.screen)

    def _render_playing(self):
        self.screen.fill(Config.BG_COLOR)

        vx = Config.SCREEN_WIDTH // 2
        colors = self.tracker.FINGER_COLORS

        # Enhanced lane lines with glow
        for i, ex in enumerate(self.notes.lane_x):
            c = colors[i]
            sx = vx + (ex - vx) * 0.28
            
            # Outer glow
            for j in range(3):
                glow_width = Config.LANE_LINE_WIDTH + 30 - j * 8
                glow_alpha = 40 - j * 10
                glow_surf = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
                pygame.draw.line(glow_surf, (*c, glow_alpha),
                               (sx, Config.VANISHING_POINT_Y), (ex, Config.SCREEN_HEIGHT),
                               glow_width)
                self.screen.blit(glow_surf, (0, 0))
            
            # Main line
            pygame.draw.line(self.screen, c,
                           (sx, Config.VANISHING_POINT_Y), (ex, Config.SCREEN_HEIGHT),
                           Config.LANE_LINE_WIDTH)

        # Hit line with glow
        hy = Config.SCREEN_HEIGHT - 100
        for i in range(3):
            glow_surf = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
            pygame.draw.line(glow_surf, (255, 240, 100, 60 - i*15), (0, hy), (Config.SCREEN_WIDTH, hy), 8 - i*2)
            self.screen.blit(glow_surf, (0, 0))
        pygame.draw.line(self.screen, (255, 255, 150), (0, hy), (Config.SCREEN_WIDTH, hy), 3)

        self.notes.render(self.screen)
        self.particles.render(self.screen)

        # Active lane indicators with modern style
        pinch = self.tracker.get_pinch_lane() if self.tracker.detected else None
        for i in range(Config.NUM_LANES):
            x = self.notes.lane_x[i]
            c = colors[i]
            if i == pinch:
                # Pulsing active indicator
                pulse = 1.0 + 0.2 * math.sin(self.animation_time * 8)
                for j in range(3):
                    glow_r = int(50 * pulse) + (3-j) * 8
                    glow_surf = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*c, 80 - j*20), (x, hy), glow_r)
                    self.screen.blit(glow_surf, (0, 0))
                pygame.draw.circle(self.screen, c, (x, hy), 32, 6)
                pygame.draw.circle(self.screen, (255, 255, 255), (x, hy), 18, 3)
            else:
                pygame.draw.circle(self.screen, (80, 80, 100), (x, hy), 28, 2)

        elapsed = time.time() - self.game_start
        self.hud.render(self.screen, self.score_mgr, elapsed, self.tracker.detected)
        self.hud.render_feedback(self.screen)

    def _render_gameover(self):
        # 1. Background with dark overlay
        bg = self.assets.images.get('settings_bg')
        if bg:
            self.screen.blit(bg, (0, 0))
        else:
            ModernUI.draw_animated_background(self.screen, self.animation_time)
        
        overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        # Frog style: text directly over dark overlay, no enclosing glass panel
        panel_y = 100  # anchor for text layout

        # TIME'S UP! title with glow
        title_font = pygame.font.Font(None, 110)
        title_text = "TIME'S UP!"
        title_color = (255, 42, 109)
        title_surf = title_font.render(title_text, True, title_color)
        for gi in range(3, 0, -1):
            gs = title_font.render(title_text, True, title_color)
            gsurf = pygame.Surface(gs.get_size(), pygame.SRCALPHA)
            gsurf.blit(gs, (0, 0))
            gsurf.set_alpha(30)
            self.screen.blit(gsurf, (Config.SCREEN_WIDTH//2 - gs.get_width()//2 - gi, panel_y + 28 - gi))
            self.screen.blit(gsurf, (Config.SCREEN_WIDTH//2 - gs.get_width()//2 + gi, panel_y + 28 + gi))
        self.screen.blit(title_surf, (Config.SCREEN_WIDTH//2 - title_surf.get_width()//2, panel_y + 28))

        # Final score — plain bright green (no shadow/glow, frog style)
        score_font = pygame.font.Font(None, 90)
        score_text = f"FINAL SCORE: {self.score_mgr.score:,}"
        score_surf = score_font.render(score_text, True, (0, 255, 80))
        self.screen.blit(score_surf, (Config.SCREEN_WIDTH//2 - score_surf.get_width()//2, panel_y + 160))

        # Stats
        stats_font = pygame.font.Font(None, 42)
        hits = self.score_mgr.hits
        max_streak = self.score_mgr.max_streak
        diff_name = self.selected_diff.name
        
        final_t = getattr(self, 'final_time_display', 0)
        time_text = f"Time: {final_t//60:02d}:{final_t%60:02d} / {self.selected_time//60:02d}:{self.selected_time%60:02d}"
        
        stats = [
            f"Successful Hits: {hits}",
            f"Max Streak: {max_streak}",
            f"Difficulty: {diff_name}",
            # time_text
        ]
        
        for i, stat in enumerate(stats):
            s_surf = stats_font.render(stat, True, (200, 230, 255))
            self.screen.blit(s_surf, (Config.SCREEN_WIDTH//2 - s_surf.get_width()//2, panel_y + 268 + i * 50))

        # Buttons
        for b in self.gameover_buttons:
            b.draw(self.screen)

    def _render_report(self):
        ModernUI.draw_animated_background(self.screen, self.animation_time)
        overlay = pygame.Surface((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        ModernUI.draw_glass_panel(self.screen, (100, 50, Config.SCREEN_WIDTH-200, Config.SCREEN_HEIGHT-150), (30, 30, 45, 230), (0, 255, 245), 2, 20)
        
        text = self.analytics.get_summary_text()
        y_off = 80
        font = pygame.font.Font(None, 34)
        for line in text.split('\n'):
            color = (0, 255, 245) if '─' not in line and 'Session' not in line else (200, 200, 255)
            if '─' in line:
                pygame.draw.line(self.screen, (60, 60, 80), (140, y_off+10), (Config.SCREEN_WIDTH-140, y_off+10))
                y_off += 25
                continue
            
            surf = font.render(line, True, color)
            self.screen.blit(surf, (140, y_off))
            y_off += 35
        
        # Back button — dedicated widget so menu_exit_btn rect is never mutated
        if self.report_back_btn:
            self.report_back_btn.draw(self.screen)

    def run(self):
        while self.running:
            dt = self.clock.tick(Config.FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()
        self.cleanup()

    def cleanup(self):
        if self.camera: self.camera.release()
        if self.tracker: self.tracker.cleanup()
        cv2.destroyAllWindows()
        pygame.quit()

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Rhythm Hero AR  –  MODERN UI EDITION")
    print("Enhanced with Glassmorphism, Neon Glows, and Smooth Animations")
    print("="*70)
    print("Controls:")
    print("  Thumb + Index   → lane 1")
    print("  Thumb + Middle  → lane 2")
    print("  Thumb + Ring    → lane 3")
    print("  Thumb + Pinky   → lane 4")
    print("  SPACE           → pause")
    print("  ESC             → back / quit")
    print("="*70 + "\n")

    try:
        game = RhythmHeroAR()
        game.run()
    except Exception as e:
        print("\nERROR:", e)
        import traceback
        traceback.print_exc()
    finally:
        print("\nGame closed. Thank you!")