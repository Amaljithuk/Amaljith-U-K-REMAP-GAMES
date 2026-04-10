"""
Lily Pad Leap - MODERN UI EDITION (Juggling Game Style)
Enhanced with modern buttons, glass panels, and juggling game aesthetics
Frog jumps when hand is folded (fist) and steers with hand tilt
"""
import pygame
import mediapipe as mp
import cv2
import numpy as np
import random
import math
from collections import deque
import sys
import time
from enum import Enum
from typing import List, Tuple, Optional
import os
import threading
import json
import pandas as pd
from datetime import datetime

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

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
    GAME_OVER = 6
    REPORT = 7

class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

# ============================================================================
# FROG ANALYTICS
# ============================================================================
class FrogAnalytics:
    """Tracks and analyses body movements during frog/lily-pad gameplay"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: datetime = datetime.now()
        self.end_time: datetime = datetime.now()
        self.hand_positions: list = []
        self.hand_velocities: list = []
        self.total_hand_distance: float = 0.0
        self.max_hand_speed: float = 0.0
        self.max_reach_left: float = 0.0
        self.max_reach_right: float = 0.0
        self.max_reach_up: float = 0.0
        self.max_reach_down: float = 0.0
        self.frame_count: int = 0
        self.jump_events: list = []        # successful landings
        self.difficulty_level: str = "Medium"
        self.duration_target: int = 300
        self.final_score: int = 0
        self.calories_burned: float = 0.0

    def start_session(self, difficulty: str, duration: int):
        self.reset()
        self.start_time = datetime.now()
        self.difficulty_level = difficulty
        self.duration_target = duration

    def record_frame(self, hand_landmarks, frame_width: int, frame_height: int):
        """Record hand movement data for a single frame."""
        self.frame_count += 1
        if not hand_landmarks:
            return
        indices = [0, 5, 9, 13, 17]
        cx = int(np.mean([hand_landmarks.landmark[i].x * frame_width for i in indices]))
        cy = int(np.mean([hand_landmarks.landmark[i].y * frame_height for i in indices]))
        current_pos = {'x': cx, 'y': cy, 'timestamp': time.time()}
        self.hand_positions.append(current_pos)
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
        center_x, center_y = frame_width / 2, frame_height / 2
        if cx < center_x:
            self.max_reach_left = max(self.max_reach_left, abs(cx - center_x))
        else:
            self.max_reach_right = max(self.max_reach_right, abs(cx - center_x))
        if cy < center_y:
            self.max_reach_up = max(self.max_reach_up, abs(cy - center_y))
        else:
            self.max_reach_down = max(self.max_reach_down, abs(cy - center_y))

    def record_jump(self, score: int):
        """Record a successful pad landing."""
        self.jump_events.append({'score': score, 'timestamp': time.time()})

    def end_session(self, final_score: int):
        self.end_time = datetime.now()
        self.final_score = final_score
        self._calculate_metrics()

    def _calculate_metrics(self):
        duration_minutes = (self.end_time - self.start_time).total_seconds() / 60
        avg_speed = np.mean(self.hand_velocities) if self.hand_velocities else 0
        intensity = 2.5 if avg_speed > 300 else (2.0 if avg_speed > 200 else (1.5 if avg_speed > 100 else 1.1))
        self.calories_burned = duration_minutes * 3.5 * intensity

    def generate_report(self):
        """Return a comprehensive analytics report dict."""
        duration = (self.end_time - self.start_time).total_seconds()
        avg_speed = np.mean(self.hand_velocities) if self.hand_velocities else 0
        total_jumps = len(self.jump_events)
        hit_rate = (total_jumps / max(self.frame_count / 60, 1)) * 100  # jumps per minute
        return {
            'session_info': {
                'date': self.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration_played': f"{int(duration // 60)}:{int(duration % 60):02d}",
                'duration_target': f"{int(self.duration_target // 60)}:{int(self.duration_target % 60):02d}",
                'difficulty': self.difficulty_level,
                'final_score': self.final_score,
                'calories_burned': f"{self.calories_burned:.1f} kcal",
            },
            'movement_metrics': {
                'total_distance': f"{self.total_hand_distance:.1f} pixels",
                'avg_speed': f"{avg_speed:.1f} px/s",
                'max_speed': f"{self.max_hand_speed:.1f} px/s",
                'frames_tracked': self.frame_count,
            },
            'performance_metrics': {
                'successful_jumps': total_jumps,
                'jumps_per_minute': f"{hit_rate:.1f}",
            },
            'range_of_motion': {
                'horizontal_reach': f"{self.max_reach_left + self.max_reach_right:.1f} pixels",
                'vertical_reach': f"{self.max_reach_up + self.max_reach_down:.1f} pixels",
            },
            'health_insights': self._generate_health_insights(duration, avg_speed, total_jumps),
        }

    def _generate_health_insights(self, duration, avg_speed, total_jumps):
        insights = []
        if duration >= 600:
            insights.append("🌟 Excellent stamina! 10+ minutes of continuous play.")
        elif duration >= 300:
            insights.append("✓ Great session! You maintained focus for 5+ minutes.")
        if avg_speed > 200:
            insights.append("🔥 High intensity hand movements detected.")
        if total_jumps >= 20:
            insights.append("🐸 Impressive! 20+ successful jumps shows great coordination.")
        insights.append(f"🔥 Estimated {self.calories_burned:.1f} calories burned this session.")
        return insights

    def save_to_json(self, filename=None):
        if filename is None:
            ts = self.start_time.strftime('%Y%m%d_%H%M%S') if self.start_time else 'unknown'
            filename = f"frog_report_{ts}.json"
        report = self.generate_report()
        if report:
            try:
                with open(filename, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"✓ Saved JSON report: {filename}")
                return filename
            except Exception as e:
                print(f"Error saving JSON report: {e}")
        return None

    def save_to_excel(self, filename="frog_analytics.xlsx"):
        """Append all metrics for this session as a single row to the cumulative Excel file."""
        report = self.generate_report()
        if not report:
            return None
        try:
            row = {}
            row.update(report['session_info'])
            row.update(report['movement_metrics'])
            row.update(report['performance_metrics'])
            row.update(report['range_of_motion'])
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
        report = self.generate_report()
        if not report:
            return "No data available"
        lines = [
            "F R O G   P E R F O R M A N C E   A N A L Y S I S",
            "─" * 50,
            f"Date: {report['session_info']['date']}",
            f"Difficulty: {report['session_info']['difficulty']}",
            f"Final Score: {report['session_info']['final_score']}",
            f"Duration: {report['session_info']['duration_played']}",
            "─" * 50,
            f"Successful Jumps: {report['performance_metrics']['successful_jumps']}",
            f"Jumps/min: {report['performance_metrics']['jumps_per_minute']}",
            f"Avg Hand Speed: {report['movement_metrics']['avg_speed']}",
            f"Calories: {report['session_info']['calories_burned']}",
            "─" * 50,
            "Health Insights:",
        ]
        for insight in report['health_insights']:
            lines.append(f"• {insight}")
        return "\n".join(lines)

# ============================================================================
# MODERN UI UTILITIES (Juggling Game Style)
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

# ============================================================================
# MODERN UI COMPONENTS (Juggling Game Style)
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
        # Label
        if self.label:
            label_font = pygame.font.Font(None, 58)
            label_surf = label_font.render(self.label, True, (0, 0, 0))
            surface.blit(label_surf, (self.x + self.width//2 - label_surf.get_width()//2, self.y - 50))
        
        # Panel background — narrower box centered within slider label
        panel_w = 260
        panel_rect = (self.x + (self.width - panel_w)//2, self.y, panel_w, 35)
        ModernUI.draw_glass_panel(surface, panel_rect, (0, 0, 0, 180), (100, 100, 120), 1, 10)
        
        # Value text
        if isinstance(self.options[self.current_index], Difficulty):
            val_text = self.options[self.current_index].name
            colors = {Difficulty.EASY: (100, 255, 100), Difficulty.MEDIUM: (255, 255, 100), 
                      Difficulty.HARD: (255, 100, 100)}
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
# SETTINGS
# ============================================================================
class Settings:
    def __init__(self):
        self.window_width = 1280
        self.window_height = 720
        self.fps = 60
        self.session_time_options = list(range(2, 21))  # 2 to 20 minutes
        self.selected_time = 5 * 60  # default 5 minutes in seconds
        self.selected_difficulty = Difficulty.MEDIUM
        self.calibration_duration = 3.0
        self.particle_count = 15
        self.gravity = 0.12
        self.fist_threshold = 0.13        # avg fingertip-to-wrist distance (normalised)
        self.fist_curl_min_fingers = 3     # how many fingers must individually be curled
        self.fist_frames_required = 3      # consecutive frames needed to confirm fist
        self.align_threshold = 35.0
        self.min_hand_size_calib = 0.12  # Lowered so user doesn't need to be very close
        self.min_hand_size_play = 0.10
        
        # Asset paths
        self.assets_dir = "asset"
        self.frog_image_path = resource_path(f"{self.assets_dir}/frog.png")
        self.frog_jump_image_path = resource_path(f"{self.assets_dir}/Frog_jump.png")
        self.frogs_bg_path = resource_path(f"{self.assets_dir}/frogs.jpeg")
        self.lily_bg_path = resource_path(f"{self.assets_dir}/lily.jpeg")
        self.water_bg_path = resource_path(f"{self.assets_dir}/water.jpg")
        self.frog_play_path = resource_path(f"{self.assets_dir}/frog_play.png")
        self.frog_exit_path = resource_path(f"{self.assets_dir}/frog_exit.png")
        self.instruction_image_path = resource_path(f"{self.assets_dir}/instruction1.jpg")
        
        self.click_sound_path = resource_path(f"{self.assets_dir}/click.mp3")
        self.frog_sound_path = resource_path(f"{self.assets_dir}/frog_sound.mp3")
        
        self.update_difficulty()

    def update_difficulty(self):
        if self.selected_difficulty == Difficulty.EASY:
            self.jump_speed = 0.06
            self.hitbox_radius = 80
            self.reach_radius_min = 180
            self.reach_radius_max = 250
        elif self.selected_difficulty == Difficulty.MEDIUM:
            self.jump_speed = 0.08
            self.hitbox_radius = 65
            self.reach_radius_min = 200
            self.reach_radius_max = 350
        else:  # HARD
            self.jump_speed = 0.10
            self.hitbox_radius = 50
            self.reach_radius_min = 250
            self.reach_radius_max = 450

# ============================================================================
# ASSET MANAGER
# ============================================================================
class AssetManager:
    def __init__(self, settings):
        self.settings = settings
        self.sounds = {}
        self.load_sounds()

    def load_sounds(self):
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"⚠ Mixer init failed: {e}")

        for name, path in [
            ('click', self.settings.click_sound_path),
            ('jump', self.settings.frog_sound_path),
        ]:
            try:
                self.sounds[name] = pygame.mixer.Sound(path)
                print(f"✓ Loaded {name} sound")
            except:
                print(f"⚠ Could not load {name} sound")
                self.sounds[name] = None

    def play_click(self):
        if self.sounds.get('click'):
            self.sounds['click'].play()

    def play_jump(self):
        if self.sounds.get('jump'):
            self.sounds['jump'].play()

# ============================================================================
# CALIBRATION
# ============================================================================
class Calibration:
    def __init__(self, settings):
        self.settings = settings
        self.is_calibrated = False
        self.start_time = None
        self.duration = settings.calibration_duration
        self.min_hand_size_norm = settings.min_hand_size_calib
        self.calibration_locked = False
        self.calibration_progress = 0.0
        self.temp_hand_bbox = None

    def start(self):
        self.is_calibrated = False
        self.calibration_locked = False
        self.calibration_progress = 0.0
        self.start_time = time.time()

    def get_hand_bounding_box(self, landmarks, frame_w, frame_h):
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

    def update(self, hand_detected, landmarks=None, frame_w=640, frame_h=480):
        if not hand_detected or not landmarks:
            self.calibration_progress = 0.0
            self.calibration_locked = False
            return 0.0, False, "Show your hand to the camera", None

        bbox, norm_size = self.get_hand_bounding_box(landmarks, frame_w, frame_h)
        is_good_size = norm_size >= self.min_hand_size_norm

        if is_good_size and self.start_time is not None:
            elapsed = time.time() - self.start_time
            self.calibration_progress = min(elapsed / self.duration, 1.0)
            if self.calibration_progress >= 1.0:
                self.calibration_locked = True
                self.is_calibrated = True
        elif not is_good_size:
            self.start_time = time.time()
            self.calibration_progress = 0.0
            self.calibration_locked = False

        if self.calibration_locked:
            message = "✓ LOCKED! Click CONFIRM to start"
            progress = 1.0
        elif self.calibration_progress > 0:
            remaining = self.duration * (1.0 - self.calibration_progress)
            message = f"Hold steady... {remaining:.1f}s"
            progress = self.calibration_progress
        else:
            message = "Move closer • Hand too small" if not is_good_size else "Hold hand steady to calibrate"
            progress = 0.0
            
        return progress, is_good_size, message, bbox

    def draw_hand_box(self, frame, bbox, is_good):
        if bbox is None:
            return frame
        x, y, w, h = bbox
        color = (0, 255, 0) if self.calibration_locked else (0, 255, 255) if is_good else (0, 0, 220)
        thick = 4 if self.calibration_locked else 3 if is_good else 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thick)
        return frame

# ============================================================================
# VISUAL EFFECTS
# ============================================================================
class Ripple:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.radius = 10
        self.max_radius = 120
        self.alpha = 255
        self.speed = 3

    def update(self):
        self.radius += self.speed
        self.alpha = max(0, 255 - (self.radius / self.max_radius) * 255)

    def draw(self, screen):
        if self.alpha > 0:
            surf = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (0, 255, 245, int(self.alpha)), (self.radius, self.radius), self.radius, 3)
            screen.blit(surf, (self.pos[0]-self.radius, self.pos[1]-self.radius))

class Particle:
    def __init__(self, pos, settings):
        self.pos = np.array(pos, dtype=float)
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(2.0, 4.0)
        self.vel = np.array([speed * math.cos(angle), speed * math.sin(angle) - 3])
        self.lifespan = 40
        self.settings = settings

    def update(self):
        self.vel[1] += self.settings.gravity
        self.pos += self.vel
        self.lifespan -= 1

    def draw(self, screen):
        if self.lifespan > 0:
            alpha = int(255 * (self.lifespan / 40))
            surf = pygame.Surface((10, 10), pygame.SRCALPHA)
            pygame.draw.circle(surf, (0, 255, 245, alpha), (5, 5), 5)
            screen.blit(surf, self.pos.astype(int))

# ============================================================================
# GAME OBJECTS
# ============================================================================
class Frog:
    def __init__(self, settings):
        self.settings = settings
        self.position = np.array([settings.window_width / 2.0, settings.window_height * 0.75])
        self.target_position = self.position.copy()
        self.jump_start_position = self.position.copy()
        self.jump_progress = 0.0
        self.is_jumping = False
        self.angle = 270.0  # Facing UP
        self.angle_buffer = deque(maxlen=8)

        # Load sitting frog image (frog.png)
        # Fallback sprite uses orange/brown so it contrasts with green lily pads
        self._fallback = pygame.Surface((80, 70), pygame.SRCALPHA)
        pygame.draw.ellipse(self._fallback, (160, 70, 10, 200), (12, 20, 56, 40))
        pygame.draw.ellipse(self._fallback, (220, 110, 20), (10, 15, 60, 40))
        pygame.draw.circle(self._fallback, (230, 140, 40), (40, 25), 22)
        pygame.draw.circle(self._fallback, (255, 255, 255), (32, 18), 8)
        pygame.draw.circle(self._fallback, (255, 255, 255), (48, 18), 8)
        pygame.draw.circle(self._fallback, (0, 0, 0), (32, 18), 4)
        pygame.draw.circle(self._fallback, (0, 0, 0), (48, 18), 4)
        pygame.draw.arc(self._fallback, (160, 70, 10), (30, 28, 20, 12), 0, math.pi, 2)

        def apply_orange_tint(surface):
            """Apply a warm orange tint specifically to the visible frog pixels."""
            tinted = surface.copy()
            # Create a mask of the frog's shape
            mask = pygame.mask.from_surface(surface)
            # Create a tint surface that only exists where the frog is
            # setcolor specifies the color and alpha (60) for the frog area
            tint_surf = mask.to_surface(setcolor=(255, 140, 0, 60), unsetcolor=(0, 0, 0, 0))
            # Add the tint to the original image
            tinted.blit(tint_surf, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
            return tinted

        try:
            loaded = pygame.image.load(settings.frog_image_path).convert_alpha()
            scaled = pygame.transform.smoothscale(loaded, (160, 140))
            self.sit_image = apply_orange_tint(scaled)
            print("✓ Loaded frog.png (sitting) with orange tint")
        except:
            print("⚠ frog.png not found - using generated sprite")
            self.sit_image = self._fallback

        try:
            loaded_jump = pygame.image.load(settings.frog_jump_image_path).convert_alpha()
            scaled_jump = pygame.transform.smoothscale(loaded_jump, (160, 140))
            self.jump_image = apply_orange_tint(scaled_jump)
            print("✓ Loaded Frog_jump.png with orange tint")
        except:
            print("⚠ Frog_jump.png not found - using sitting sprite")
            self.jump_image = self.sit_image

        self.base_image = self.sit_image

    def update_angle(self, new_angle):
        self.angle_buffer.append(new_angle)
        self.angle = np.mean(self.angle_buffer)

    def start_jump(self, target_pos):
        self.jump_start_position = self.position.copy()
        self.target_position = target_pos
        self.jump_progress = 0.0
        self.is_jumping = True

    def update_jump(self):
        if self.is_jumping:
            self.jump_progress += self.settings.jump_speed
            if self.jump_progress >= 1.0:
                self.jump_progress = 1.0
                self.is_jumping = False
                self.position = self.target_position.copy()
            else:
                t = self.jump_progress
                dist_vec = self.target_position - self.jump_start_position
                dist = np.linalg.norm(dist_vec)
                arc_height = dist * 0.3
                x = self.jump_start_position[0] + t * dist_vec[0]
                y = self.jump_start_position[1] + t * dist_vec[1] - arc_height * math.sin(math.pi * t)
                self.position = np.array([x, y])

    def draw(self, screen):
        self.base_image = self.jump_image if self.is_jumping else self.sit_image
        # Rotate image - rotate() handles alpha transparency for corners if the source has alpha
        rotated = pygame.transform.rotate(self.base_image, -(self.angle + 90))
        rect = rotated.get_rect(center=self.position.astype(int))
        screen.blit(rotated, rect)
        
        # Direction indicator
        indicator_dist = 60
        indicator_x = self.position[0] + indicator_dist * math.cos(math.radians(self.angle))
        indicator_y = self.position[1] + indicator_dist * math.sin(math.radians(self.angle))
        pygame.draw.line(screen, (0, 255, 245), 
                        self.position.astype(int), 
                        (int(indicator_x), int(indicator_y)), 3)
        pygame.draw.circle(screen, (0, 255, 245), (int(indicator_x), int(indicator_y)), 6)

class LilyPad:
    def __init__(self, position, score=0):
        self.position = np.array(position, dtype=float)
        shrink_factor = max(0.6, 1.0 - (score / 5000))
        self.radius = 55 * shrink_factor

    def draw(self, screen, glowing=False):
        pos_int = self.position.astype(int)
        
        # Shadow
        shadow_surf = pygame.Surface((self.radius*3, self.radius*2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 60), 
                           (0, 0, self.radius*3, self.radius*2))
        screen.blit(shadow_surf, (pos_int[0]-self.radius*1.5, pos_int[1]-self.radius*0.7+5))
        
        # Glow effect when aligned
        if glowing:
            for i in range(3):
                glow_surf = pygame.Surface((int(self.radius*3), int(self.radius*2)), pygame.SRCALPHA)
                pygame.draw.ellipse(glow_surf, (255, 220, 0, 50-i*12),
                                   (0, 0, int(self.radius*3), int(self.radius*2)))
                screen.blit(glow_surf, (pos_int[0]-self.radius*1.5-i*4, pos_int[1]-self.radius*0.7-i*4))
        
        # Main pad - bright gold glow color when aligned
        color = (30, 160, 80) if not glowing else (255, 210, 0)
        pygame.draw.ellipse(screen, color, 
                           (pos_int[0]-self.radius, pos_int[1]-self.radius*0.7, 
                            self.radius*2, self.radius*1.4))
        pygame.draw.ellipse(screen, (20, 100, 50), 
                           (pos_int[0]-self.radius, pos_int[1]-self.radius*0.7, 
                            self.radius*2, self.radius*1.4), 3)
        
        # Details
        center_x = pos_int[0]
        center_y = int(pos_int[1] - self.radius*0.2)
        for i in range(5):
            angle = (360 / 5) * i - 90
            end_x = center_x + (self.radius*0.7) * math.cos(math.radians(angle))
            end_y = center_y + (self.radius*0.5) * math.sin(math.radians(angle))
            pygame.draw.line(screen, (20, 100, 50), (center_x, center_y), (end_x, end_y), 2)

class PadManager:
    def __init__(self, settings):
        self.settings = settings
        center = np.array([settings.window_width / 2.0, settings.window_height * 0.8])
        self.current_pad = LilyPad(center)
        self.target_pad = None
        self.spawn_new_target(0)

    def spawn_new_target(self, score):
        angle = random.uniform(230, 310)
        dist = random.uniform(self.settings.reach_radius_min, self.settings.reach_radius_max)

        offset = np.array([dist * math.cos(math.radians(angle)),
                           dist * math.sin(math.radians(angle))])
        new_pos = self.current_pad.position + offset

        # Keep pads in the safe central zone:
        #   Left boundary (370) clears the Score HUD (x=20..260) and camera PIP (x=10..350)
        #   Right boundary (win_w-130) clears the EXIT button (x≈1180)
        #   Top boundary (150) clears the Score/Time HUD (y=20..130)
        #   Bottom boundary (win_h-130) clears the alignment indicator (y≈610..690)
        W, H = self.settings.window_width, self.settings.window_height
        new_pos[0] = np.clip(new_pos[0], 370, W - 130)
        new_pos[1] = np.clip(new_pos[1], 150, H - 130)
        self.target_pad = LilyPad(new_pos, score)

    def draw(self, screen, aligned):
        self.current_pad.draw(screen)
        if self.target_pad:
            self.target_pad.draw(screen, aligned)

# ============================================================================
# MAIN GAME ENGINE
# ============================================================================
class GameEngine:
    def _init_camera_thread(self):
        """Runs in a background thread; initialises camera without blocking the render loop."""
        cap = self._get_camera()
        self.cap = cap
        self.camera_ready = True
        print("✓ Camera ready (background thread)")

    def _get_camera(self):
        """Attempts to find an external camera (index 1), falls back to system camera (index 0)."""
        for idx in [ 0, 1]:
            print(f"--- Attempting to open camera index {idx} ---")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✓ Successfully opened camera at index {idx}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    return cap
                else:
                    print(f"⚠ Camera at index {idx} opened but could not read frame.")
                    cap.release()
            else:
                print(f"⚠ Could not open camera at index {idx}")
        print("✖ No cameras found!")
        return cv2.VideoCapture(0)

    def __init__(self):
        pygame.init()
        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.window_width, self.settings.window_height), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("Lily Leap - Modern Edition")
        self.clock = pygame.time.Clock()
        # No static splash – the animated LOADING state handles it

        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Camera init runs in a background thread so the loading screen shows immediately
        self.cap = None
        self.camera_ready = False
        threading.Thread(target=self._init_camera_thread, daemon=True).start()
        
        self.state = GameState.LOADING
        self.animation_time = 0.0
        self.loading_progress = 0.0
        self.loading_start = time.time()
        self.assets = AssetManager(self.settings)
        
        self.analytics = FrogAnalytics()
        self.reset_game()
        
        self.camera_surf = None
        self.aligned_status = False
        self.was_fist = False
        
        # Calibration
        self.calib = None
        self.calib_progress = 0.0
        self.calib_is_good = False
        self.calib_message = ""
        self.calib_locked = False
        
        self.create_ui()
        self.frogs_bg = None
        self.lily_bg = None
        self.water_bg = None
        self.instruction_img = None
        self.flogo_img = None
        self.frog_play_img = None
        self.frog_exit_img = None
        self._load_bg_assets()

    def _load_bg_assets(self):
        """Load all background and UI image assets"""
        W, H = self.settings.window_width, self.settings.window_height
        s = self.settings

        def load_bg(path, size):
            try:
                img = pygame.image.load(path).convert()
                return pygame.transform.smoothscale(img, size)
            except:
                print(f"⚠ Could not load {path}")
                return None

        def load_img(path, size):
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.smoothscale(img, size)
            except:
                print(f"⚠ Could not load {path}")
                return None

        self.frogs_bg = load_bg(s.frogs_bg_path, (W, H))
        self.lily_bg = load_bg(s.lily_bg_path, (W, H))
        self.water_bg = load_bg(s.water_bg_path, (W, H))

        # Instruction screen image (shown fullscreen, no text)
        self.instruction_img = load_bg(s.instruction_image_path, (W, H))

        # flogo (game logo)
        self.flogo_img = load_img(resource_path(f"{s.assets_dir}/flogo.png"), (600, 360))

        # Image buttons
        self.frog_play_img = load_img(s.frog_play_path, (360, 190))
        self.frog_exit_img = load_img(s.frog_exit_path, (320, 130))

        # Rect hot-zones for image buttons
        cx = self.settings.window_width // 2
        cy = self.settings.window_height // 2
        self.frog_play_rect = pygame.Rect(cx - 180, cy - 80, 360, 190)
        self.frog_exit_rect = pygame.Rect(self.settings.window_width - 340, 15, 320, 130)

    def create_ui(self):
        cx = self.settings.window_width // 2
        
        # Home buttons
        self.play_btn = ModernButton(cx-150, 450, 300, 70, "START GAME", primary=True)
        self.exit_btn = ModernButton(cx-150, 540, 300, 70, "EXIT", primary=False)
        
        # Settings
        self.diff_slider = ModernSlider(cx-200, 280, 400, list(Difficulty), 1, "DIFFICULTY")
        times_display = [f"{t}min" for t in self.settings.session_time_options]
        self.time_slider = ModernSlider(cx-200, 380, 400, times_display, 3, "TIME LIMIT")
        self.next_btn = ModernButton(cx-110, 520, 220, 60, "NEXT", primary=True)
        self.back_settings_btn = ModernButton(60, self.settings.window_height-90, 160, 55, "BACK", primary=False)
        
        # Instructions
        self.start_btn = ModernButton(cx-140, self.settings.window_height-130, 280, 70, "START", primary=True)
        self.back_instr_btn = ModernButton(60, self.settings.window_height-80, 160, 55, "BACK", primary=False)
        
        # Game Over buttons
        self.play_again_btn = ModernButton(cx-290, 555, 260, 70, "PLAY AGAIN", primary=True)
        self.menu_btn = ModernButton(cx+30, 555, 260, 70, "MENU", primary=False)
        self.view_report_btn = ModernButton(self.settings.window_width-270, 20, 250, 60, "VIEW REPORT", primary=True)
        self.report_back_btn = ModernButton(cx-120, self.settings.window_height-100, 240, 60, "BACK", primary=False)
        
        # Playing exit button
        self.exit_play_btn = ModernButton(self.settings.window_width - 100, 20, 80, 40, "EXIT", primary=False)
        
        # Calibration back button
        self.calib_back_btn = ModernButton(60, self.settings.window_height - 80, 160, 55, "BACK", primary=False)

    def reset_game(self):
        self.frog = Frog(self.settings)
        self.pad_manager = PadManager(self.settings)
        self.score = 0
        self.timer = 0.0
        self.ripples = []
        self.particles = []
        self.aligned_status = False
        self.was_fist = False
        # Fist debounce buffer: stores per-frame bool; fist only confirmed when all agree
        self.fist_buffer = deque(maxlen=self.settings.fist_frames_required)
        # True when the user is too far from the camera (hand too small)
        self.hand_too_far = False

    def handle_events(self):
        mx, my = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state in (GameState.SETTINGS, GameState.INSTRUCTIONS, GameState.GAME_OVER, GameState.REPORT):
                        self.state = GameState.HOME
                    elif self.state == GameState.PLAYING:
                        self.state = GameState.GAME_OVER
                    else:
                        pygame.quit()
                        sys.exit()
                # Arrow key navigation in Settings
                elif self.state == GameState.SETTINGS:
                    if event.key in (pygame.K_LEFT, pygame.K_DOWN):
                        if self.diff_slider.current_index > 0:
                            self.diff_slider.current_index -= 1
                            self.diff_slider.value_animation = 1.0
                    elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                        if self.diff_slider.current_index < len(self.diff_slider.options) - 1:
                            self.diff_slider.current_index += 1
                            self.diff_slider.value_animation = 1.0
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if time.time() - getattr(self, 'last_click', 0) > 0.18:
                    self.last_click = time.time()
                    self.assets.play_click()
                    self._handle_clicks(mx, my)

    def _handle_clicks(self, mx, my):
        if self.state == GameState.HOME:
            if self.frog_play_img and self.frog_play_rect.collidepoint(mx, my):
                self.state = GameState.SETTINGS
            elif self.frog_exit_img and self.frog_exit_rect.collidepoint(mx, my):
                pygame.quit()
                sys.exit()
            if self.play_btn.is_clicked((mx, my), True):
                self.state = GameState.SETTINGS
            if self.exit_btn.is_clicked((mx, my), True):
                pygame.quit()
                sys.exit()
        
        elif self.state == GameState.SETTINGS:
            self.diff_slider.handle_click((mx, my), True)
            self.time_slider.handle_click((mx, my), True)
            
            if self.next_btn.is_clicked((mx, my), True):
                self.settings.selected_difficulty = self.diff_slider.get_value()
                idx = self.time_slider.current_index
                chosen_minutes = self.settings.session_time_options[idx]
                self.settings.selected_time = chosen_minutes * 60
                self.settings.update_difficulty()
                self.state = GameState.INSTRUCTIONS
            
            if self.back_settings_btn.is_clicked((mx, my), True):
                self.state = GameState.HOME
        
        elif self.state == GameState.INSTRUCTIONS:
            if self.start_btn.is_clicked((mx, my), True):
                if not self.camera_ready:
                    # Camera still initialising – do nothing yet; update() will auto-advance
                    pass
                else:
                    self.calib = Calibration(self.settings)
                    self.calib.start()
                    self.state = GameState.CALIBRATION
            
            if self.back_instr_btn.is_clicked((mx, my), True):
                self.state = GameState.SETTINGS
        
        elif self.state == GameState.CALIBRATION:
            if self.calib_back_btn.is_clicked((mx, my), True):
                self.state = GameState.INSTRUCTIONS
        
        elif self.state == GameState.PLAYING:
            if self.exit_play_btn.is_clicked((mx, my), True):
                self.analytics.end_session(self.score)
                self.analytics.save_to_json()
                self.analytics.save_to_excel()
                self.state = GameState.GAME_OVER
        
        elif self.state == GameState.GAME_OVER:
            if self.play_again_btn.is_clicked((mx, my), True):
                self.reset_game()
                self.calib = Calibration(self.settings)
                self.calib.start()
                self.state = GameState.CALIBRATION
            
            if self.menu_btn.is_clicked((mx, my), True):
                self.state = GameState.HOME
            
            if self.view_report_btn.is_clicked((mx, my), True):
                self.state = GameState.REPORT
        
        elif self.state == GameState.REPORT:
            if self.report_back_btn.is_clicked((mx, my), True):
                self.state = GameState.GAME_OVER

    def update(self, dt):
        self.animation_time += dt
        mx, my = pygame.mouse.get_pos()
        
        if self.state == GameState.LOADING:
            elapsed = time.time() - self.loading_start
            self.loading_progress = min(elapsed / 2.5, 1.0)
            # Drain a frame if the camera is already ready (warms up the buffer)
            if self.cap is not None:
                self.cap.read()
            if self.loading_progress >= 1.0:
                self.state = GameState.HOME
            return
        
        # Update buttons
        if self.state == GameState.HOME:
            self.play_btn.update((mx, my), dt)
            self.exit_btn.update((mx, my), dt)
        elif self.state == GameState.SETTINGS:
            self.diff_slider.update((mx, my), dt)
            self.time_slider.update((mx, my), dt)
            self.next_btn.update((mx, my), dt)
            self.back_settings_btn.update((mx, my), dt)
        elif self.state == GameState.INSTRUCTIONS:
            self.start_btn.update((mx, my), dt)
            self.back_instr_btn.update((mx, my), dt)
        elif self.state == GameState.CALIBRATION:
            self.calib_back_btn.update((mx, my), dt)
        elif self.state == GameState.PLAYING:
            self.exit_play_btn.update((mx, my), dt)
        elif self.state == GameState.GAME_OVER:
            self.play_again_btn.update((mx, my), dt)
            self.menu_btn.update((mx, my), dt)
            self.view_report_btn.update((mx, my), dt)
        elif self.state == GameState.REPORT:
            self.report_back_btn.update((mx, my), dt)
        
        # Camera processing – skip if camera not ready yet
        if self.cap is None or not self.camera_ready:
            return
        success, frame = self.cap.read()
        if success and frame is not None:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if self.state == GameState.CALIBRATION:
                hand_detected = results.multi_hand_landmarks is not None
                landmarks = results.multi_hand_landmarks[0] if hand_detected else None
                
                progress, is_good, message, bbox = self.calib.update(
                    hand_detected, landmarks, 640, 480
                )
                
                self.calib_progress = progress
                self.calib_is_good = is_good
                self.calib_message = message
                self.calib_locked = self.calib.calibration_locked
                
                frame = self.calib.draw_hand_box(frame, bbox, is_good)
                
                if self.calib.is_calibrated:
                    self.reset_game()
                    self.analytics.start_session(
                        self.settings.selected_difficulty.name,
                        self.settings.selected_time
                    )
                    self.state = GameState.PLAYING
            
            elif self.state == GameState.PLAYING:
                # Record hand movement data every frame
                if results.multi_hand_landmarks:
                    self.analytics.record_frame(
                        results.multi_hand_landmarks[0], 640, 480
                    )
                self._update_playing(results, dt)
            
            # Draw hand landmarks on the frame for the PIP window
            annotated = rgb.copy()
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Draw connections (skeleton)
                    self.mp_drawing.draw_landmarks(
                        annotated,
                        hand_lms,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(0, 255, 200), thickness=2, circle_radius=3),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(
                            color=(255, 255, 255), thickness=2)
                    )
                    # Colour-code fingertips (indices 4,8,12,16,20)
                    h_px, w_px = annotated.shape[:2]
                    tip_colors = [
                        (255, 80, 80),   # thumb  – red
                        (80, 200, 255),  # index  – cyan
                        (255, 220, 50),  # middle – yellow
                        (180, 80, 255),  # ring   – purple
                        (80, 255, 120),  # pinky  – green
                    ]
                    for tip_idx, tip_lm_idx in enumerate([4, 8, 12, 16, 20]):
                        lm_tip = hand_lms.landmark[tip_lm_idx]
                        cx_tip = int(lm_tip.x * w_px)
                        cy_tip = int(lm_tip.y * h_px)
                        cv2.circle(annotated, (cx_tip, cy_tip), 8, tip_colors[tip_idx], -1)
                        cv2.circle(annotated, (cx_tip, cy_tip), 8, (255, 255, 255), 1)
                    # Wrist dot
                    lm_wrist = hand_lms.landmark[0]
                    wx = int(lm_wrist.x * w_px)
                    wy = int(lm_wrist.y * h_px)
                    cv2.circle(annotated, (wx, wy), 10, (0, 200, 255), -1)
                    cv2.circle(annotated, (wx, wy), 10, (255, 255, 255), 2)
                    # Gesture label – mirrors the two-layer fist check used in gameplay
                    tip_lm_idx = [8, 12, 16, 20]
                    pip_lm_idx = [6, 10, 14, 18]
                    palm_lm = hand_lms.landmark[0]
                    palm_pt = np.array([palm_lm.x, palm_lm.y])
                    dists_g = [np.linalg.norm(np.array([hand_lms.landmark[t].x,
                                                         hand_lms.landmark[t].y]) - palm_pt)
                               for t in tip_lm_idx]
                    avg_ok_g = np.mean(dists_g) < self.settings.fist_threshold
                    curl_ok_g = sum(
                        1 for tip, pip in zip(tip_lm_idx, pip_lm_idx)
                        if hand_lms.landmark[tip].y > hand_lms.landmark[pip].y
                    ) >= self.settings.fist_curl_min_fingers
                    gesture_label = "FIST" if (avg_ok_g and curl_ok_g) else "OPEN"
                    g_color = (80, 255, 80) if gesture_label == "OPEN" else (255, 80, 80)
                    cv2.putText(annotated, gesture_label, (10, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(annotated, gesture_label, (10, 30),
                                cv2.FONT_HERSHEY_DUPLEX, 0.9, g_color, 2, cv2.LINE_AA)

            # Create camera surface
            small_cam = cv2.resize(annotated, (320, 240))
            self.camera_surf = pygame.surfarray.make_surface(small_cam.swapaxes(0, 1))

    def _update_playing(self, results, dt):
        hand_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
        frame_is_fist = False
        self.aligned_status = False
        self.hand_too_far = False

        if hand_detected:
            landmarks = results.multi_hand_landmarks[0]
            lm_all = landmarks.landmark

            # Use wrist (0) → middle-finger MCP (9) distance as hand size.
            # This landmark pair is STABLE whether the hand is open or a fist,
            # so making a fist no longer falsely triggers the "too far" warning.
            wrist = np.array([lm_all[0].x, lm_all[0].y])
            mid_mcp = np.array([lm_all[9].x, lm_all[9].y])
            norm_size = np.linalg.norm(mid_mcp - wrist)

            if norm_size < self.settings.min_hand_size_play:
                # Hand too small → user too far back; pause timer, show warning
                self.hand_too_far = True
                hand_detected = False
            else:
                lm = landmarks.landmark

                # ── Tilt-based steering ──────────────────────────────────────
                # Use the angle of the wrist(0)→middle-MCP(9) vector as the
                # hand's tilt. A small tilt maps to a large frog direction
                # change via the sensitivity multiplier – no lateral movement
                # needed, just rotate your wrist.
                wrist_pt = np.array([lm[0].x, lm[0].y])
                mcp9_pt  = np.array([lm[9].x, lm[9].y])
                hand_vec  = mcp9_pt - wrist_pt
                # Angle of hand in screen space (y increases downward)
                # Neutral = pointing straight up ≈ 270°
                hand_angle = math.degrees(math.atan2(hand_vec[1], hand_vec[0])) % 360
                neutral_angle = 270.0
                deviation = hand_angle - neutral_angle
                # Wrap deviation to [-180, 180]
                if deviation > 180:  deviation -= 360
                if deviation < -180: deviation += 360
                # Amplify: small tilt → big direction change
                steering_sensitivity = 2.5
                target_angle = 270.0 + deviation * steering_sensitivity
                self.frog.update_angle(target_angle)

                # ── Improved fist detection ──────────────────────────────────
                # Layer 1: average fingertip-to-wrist distance (backstop)
                tip_indices = [8, 12, 16, 20]
                palm = np.array([lm[0].x, lm[0].y])
                dists = [np.linalg.norm(np.array([lm[t].x, lm[t].y]) - palm)
                         for t in tip_indices]
                avg_ok = np.mean(dists) < self.settings.fist_threshold

                # Layer 2: per-finger curl check.
                # PIP joints: index=6, middle=10, ring=14, pinky=18
                # Tip joints: index=8, middle=12, ring=16, pinky=20
                # In image coords y increases downward, so tip.y > pip.y means curled.
                pip_indices = [6, 10, 14, 18]
                curled_count = sum(
                    1 for tip, pip in zip(tip_indices, pip_indices)
                    if lm[tip].y > lm[pip].y
                )
                curl_ok = curled_count >= self.settings.fist_curl_min_fingers

                frame_is_fist = avg_ok and curl_ok

                if self.pad_manager.target_pad:
                    vec_to_pad = self.pad_manager.target_pad.position - self.frog.position
                    angle_to_pad = math.degrees(math.atan2(vec_to_pad[1], vec_to_pad[0])) % 360
                    diff = abs(self.frog.angle - angle_to_pad) % 360
                    self.aligned_status = min(diff, 360 - diff) < self.settings.align_threshold

        # Layer 3: temporal debounce – fist only confirmed if ALL recent frames agree
        self.fist_buffer.append(frame_is_fist)
        is_fist = len(self.fist_buffer) == self.settings.fist_frames_required and all(self.fist_buffer)

        fist_pressed = is_fist and not self.was_fist

        if fist_pressed and self.aligned_status and not self.frog.is_jumping:
            self.frog.start_jump(self.pad_manager.target_pad.position)
            self.assets.play_jump()

        self.was_fist = is_fist

        # Only update game physics/timer when the player is in range
        if not self.hand_too_far:
            self.frog.update_jump()

            self.frog.position[0] = np.clip(self.frog.position[0], 80, self.settings.window_width - 80)

            scroll_limit = self.settings.window_height * 0.5
            if self.frog.position[1] < scroll_limit:
                shift = scroll_limit - self.frog.position[1]
                self.frog.position[1] += shift
                self.pad_manager.current_pad.position[1] += shift
                if self.pad_manager.target_pad:
                    self.pad_manager.target_pad.position[1] += shift
                for r in self.ripples:
                    r.pos[1] += shift

            if not self.frog.is_jumping and self.pad_manager.target_pad:
                dist = np.linalg.norm(self.frog.position - self.pad_manager.target_pad.position)
                if dist < self.pad_manager.target_pad.radius:
                    self.score += 5
                    self.analytics.record_jump(self.score)
                    self.ripples.append(Ripple(self.frog.position))
                    for _ in range(self.settings.particle_count):
                        self.particles.append(Particle(self.frog.position, self.settings))
                    self.pad_manager.current_pad = self.pad_manager.target_pad
                    self.pad_manager.spawn_new_target(self.score)

            for r in self.ripples:
                r.update()
            self.ripples = [r for r in self.ripples if r.alpha > 0]

            for p in self.particles:
                p.update()
            self.particles = [p for p in self.particles if p.lifespan > 0]

            self.timer += dt
            if self.timer >= self.settings.selected_time:
                self.analytics.end_session(self.score)
                self.analytics.save_to_json()
                self.analytics.save_to_excel()
                self.state = GameState.GAME_OVER

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
        
        pygame.display.flip()

    def _render_loading(self):
        W, H = self.settings.window_width, self.settings.window_height
        # Use lily.jpeg as the loading background if loaded, else gradient
        if hasattr(self, 'lily_bg') and self.lily_bg:
            self.screen.blit(self.lily_bg, (0, 0))
            # Light dark overlay so text pops
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 100))
            self.screen.blit(ov, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        title_font = pygame.font.Font(None, 130)
        ModernUI.draw_neon_text(self.screen, "LILY LEAP",
                               (W//2 - title_font.size("LILY LEAP")[0]//2, 170),
                               title_font, (0, 255, 245), True)
        
        sub_font = pygame.font.Font(None, 52)
        ModernUI.draw_neon_text(self.screen, "Rehabilitation Edition",
                               (W//2 - sub_font.size("Rehabilitation Edition")[0]//2, 280),
                               sub_font, (255, 235, 160), True)
        
        # Progress bar
        bw, bh = 700, 50
        bx = (W - bw)//2
        by = 430
        
        ModernUI.draw_glass_panel(self.screen, (bx, by, bw, bh), (0, 0, 0, 180), (0, 255, 245), 1, 25)
        
        pw = int((bw - 10) * self.loading_progress)
        if pw > 0:
            prog_surf = ModernUI.create_gradient_surface(pw, bh-10, (0, 200, 180), (0, 255, 245), False)
            self.screen.blit(prog_surf, (bx + 5, by + 5))
        
        perc_font = pygame.font.Font(None, 44)
        ModernUI.draw_neon_text(self.screen, f"{int(self.loading_progress*100)}%",
                               (W//2 - perc_font.size(f"{int(self.loading_progress*100)}%")[0]//2, 
                                by + bh + 25),
                               perc_font, (0, 255, 245), True)

    def _render_home(self):
        if self.frogs_bg:
            self.screen.blit(self.frogs_bg, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        # Draw flogo if available
        if self.flogo_img:
            flogo_rect = self.flogo_img.get_rect(centerx=self.settings.window_width//2, y=50)
            self.screen.blit(self.flogo_img, flogo_rect)
        
        # Draw image buttons with hover effects
        mx, my = pygame.mouse.get_pos()
        
        def draw_img_btn(img, rect, hover):
            if hover:
                scale = 1.05
                sw = int(rect.w * scale)
                sh = int(rect.h * scale)
                scaled = pygame.transform.smoothscale(img, (sw, sh))
                draw_x = rect.centerx - sw // 2
                draw_y = rect.centery - sh // 2
                self.screen.blit(scaled, (draw_x, draw_y))
            else:
                # Always scale to rect size so the button never changes size on hover
                scaled = pygame.transform.smoothscale(img, (rect.w, rect.h))
                self.screen.blit(scaled, rect.topleft)
        
        if self.frog_play_img:
            # Lower the play button (moved down 80px from center)
            play_rect = pygame.Rect(self.frog_play_rect.x,
                                    self.frog_play_rect.y + 80,
                                    self.frog_play_rect.w,
                                    self.frog_play_rect.h)
            draw_img_btn(self.frog_play_img, play_rect,
                         play_rect.collidepoint(mx, my))
        else:
            self.play_btn.draw(self.screen)
        
        if self.frog_exit_img:
            draw_img_btn(self.frog_exit_img, self.frog_exit_rect,
                         self.frog_exit_rect.collidepoint(mx, my))
        else:
            self.exit_btn.draw(self.screen)

    def _render_settings(self):
        if self.frogs_bg:
            self.screen.blit(self.frogs_bg, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        # No heading — just sliders
        self.diff_slider.draw(self.screen)
        self.time_slider.draw(self.screen)
        self.next_btn.draw(self.screen)
        self.back_settings_btn.draw(self.screen)

    def _render_instructions(self):
        W, H = self.settings.window_width, self.settings.window_height
        
        # Show instruction.png fullscreen if available, else fallback to dark gradient
        if hasattr(self, 'instruction_img') and self.instruction_img:
            self.screen.blit(self.instruction_img, (0, 0))
        elif self.frogs_bg:
            self.screen.blit(self.frogs_bg, (0, 0))
            overlay = pygame.Surface((W, H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        # Only show START button (no text)
        self.start_btn.draw(self.screen)
        self.back_instr_btn.draw(self.screen)

    def _render_calibration(self):
        W, H = self.settings.window_width, self.settings.window_height
        if self.frogs_bg:
            self.screen.blit(self.frogs_bg, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        # Bold orange-gold calibration title – no shadow, beautiful font
        try:
            beauty_font_title = pygame.font.SysFont("Segoe UI", 74, bold=True)
            beauty_font_sub = pygame.font.SysFont("Segoe UI", 32)
        except:
            beauty_font_title = pygame.font.SysFont("Verdana", 70, bold=True)
            beauty_font_sub = pygame.font.SysFont("Verdana", 30)
            
        ModernUI.draw_neon_text(self.screen, "HAND CALIBRATION",
                               (W//2 - beauty_font_title.size("HAND CALIBRATION")[0]//2, 10),
                               beauty_font_title, (255, 255, 255), glow=False, shadow=False)
        
        # Subtitle instruction
        # ModernUI.draw_neon_text(self.screen, "Show your hand to the camera – a small distance is fine",
        #                        (W//2 - beauty_font_sub.size("Show your hand to the camera – a small distance is fine")[0]//2, 60),
        #                        beauty_font_sub, (255, 255, 255), glow=False, shadow=False)
        
        if self.camera_surf:
            # Shift camera feed down slightly and resize for better layout
            large = pygame.transform.scale(self.camera_surf, (640, 480))
            cx = (W - 640) // 2
            cy = 125
            ModernUI.draw_glass_panel(self.screen, (cx-15, cy-15, 670, 510), (0, 0, 0, 160), (0, 255, 245), 2, 20)
            self.screen.blit(large, (cx, cy))
        
        # Progress bar for calibration
        if not self.calib_locked and self.calib_progress > 0:
            bw, bh = 500, 30
            bx = (W - bw) // 2
            by = H - 105
            ModernUI.draw_glass_panel(self.screen, (bx, by, bw, bh), (0, 0, 0, 180), (0, 255, 245), 1, 15)
            pw = int((bw - 10) * self.calib_progress)
            if pw > 0:
                prog_surf = ModernUI.create_gradient_surface(pw, bh-10, (0, 200, 180), (0, 255, 245), False)
                self.screen.blit(prog_surf, (bx + 5, by + 5))
        
        # Back button
        self.calib_back_btn.draw(self.screen)
        msg_font = pygame.font.SysFont("Segoe UI", 36, bold=True)
        text_color = (80, 255, 80) if self.calib_locked else (255, 255, 255) if self.calib_is_good else (255, 180, 80)
        msg_surf = msg_font.render(self.calib_message, True, text_color)
        shadow_surf = msg_font.render(self.calib_message, True, (0, 0, 0))
        self.screen.blit(shadow_surf, (W//2 - msg_surf.get_width()//2 + 2, H - 55 + 2))
        self.screen.blit(msg_surf, (W//2 - msg_surf.get_width()//2, H - 55))

    def _render_playing(self):
        if self.water_bg:
            self.screen.blit(self.water_bg, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        # Effects and objects
        for r in self.ripples:
            r.draw(self.screen)
        
        self.pad_manager.draw(self.screen, self.aligned_status)
        
        for p in self.particles:
            p.draw(self.screen)
        
        self.frog.draw(self.screen)
        
        # HUD
        ModernUI.draw_glass_panel(self.screen, (20, 20, 240, 110), (0, 0, 0, 180), (0, 255, 245), 1, 15)
        score_font = pygame.font.Font(None, 44)
        ModernUI.draw_neon_text(self.screen, f"SCORE: {self.score}",
                               (35, 30), score_font, (0, 255, 100), False)
        time_font = pygame.font.Font(None, 44)
        elapsed = int(self.timer)
        time_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
        ModernUI.draw_neon_text(self.screen, f"TIME:  {time_str}",
                               (35, 75), time_font, (0, 255, 245), False)
        
        # Camera PIP
        if self.camera_surf:
            pip_x = 20
            pip_y = self.settings.window_height - 260
            ModernUI.draw_glass_panel(self.screen, (pip_x-10, pip_y-10, 340, 260), (0, 0, 0, 180), (0, 255, 245), 1, 15)
            self.screen.blit(self.camera_surf, (pip_x, pip_y))
        
        # Exit button
        self.exit_play_btn.draw(self.screen)
        
        # Alignment indicator
        indicator_y = self.settings.window_height - 80
        ModernUI.draw_glass_panel(self.screen, (self.settings.window_width//2 - 200, indicator_y - 30, 400, 80),
                                 (0, 0, 0, 180), (0, 255, 245), 1, 15)
        
        color = (255, 230, 0) if self.aligned_status else (255, 80, 80)  # Bright gold when aligned
        pulse = 1.0 + 0.2 * math.sin(self.animation_time * 6)
        
        circle_surf = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(circle_surf, (*color, 150), (25, 25), int(20 * pulse))
        pygame.draw.circle(circle_surf, color, (25, 25), 15)
        self.screen.blit(circle_surf, (self.settings.window_width//2 - 25, indicator_y - 10))
        
        inst_font = pygame.font.Font(None, 28)
        inst_text = "ALIGNED! Make a FIST to JUMP!" if self.aligned_status else "Steer hand to align with pad"
        inst_surf = inst_font.render(inst_text, True, color)
        self.screen.blit(inst_surf, inst_surf.get_rect(centerx=self.settings.window_width//2,
                                                       y=indicator_y + 20))

        # ── MOVE CLOSER warning overlay ──────────────────────────────────────
        if getattr(self, 'hand_too_far', False):
            W, H = self.settings.window_width, self.settings.window_height
            # Dark semi-transparent full-screen overlay
            warn_ov = pygame.Surface((W, H), pygame.SRCALPHA)
            warn_ov.fill((0, 0, 0, 160))
            self.screen.blit(warn_ov, (0, 0))

            # Pulsing panel
            pulse_scale = 1.0 + 0.06 * math.sin(self.animation_time * 5)
            pw, ph = int(580 * pulse_scale), int(160 * pulse_scale)
            px = W // 2 - pw // 2
            py = H // 2 - ph // 2
            ModernUI.draw_glass_panel(self.screen, (px, py, pw, ph),
                                      (120, 0, 0, 200), (255, 80, 80), 3, 24)

            # Icon + text
            warn_font = pygame.font.SysFont("Segoe UI", 54, bold=True)
            warn_surf = warn_font.render("⚠  MOVE CLOSER", True, (255, 80, 80))
            self.screen.blit(warn_surf,
                             warn_surf.get_rect(centerx=W // 2, centery=H // 2 - 18))

            sub_font = pygame.font.SysFont("Segoe UI", 26)
            sub_surf = sub_font.render("Bring your hand closer to the camera to resume",
                                       True, (255, 200, 200))
            self.screen.blit(sub_surf,
                             sub_surf.get_rect(centerx=W // 2, centery=H // 2 + 38))

    def _render_gameover(self):
        W, H = self.settings.window_width, self.settings.window_height
        
        if self.frogs_bg:
            self.screen.blit(self.frogs_bg, (0, 0))
        else:
            ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))
        
        # No glass panel — text renders directly over the dark overlay
        panel_y = 100  # anchor for text layout
        
        # "TIME'S UP!" title in red like game.py
        title_font = pygame.font.Font(None, 110)
        title_text = "TIME'S UP!"
        title_surf = title_font.render(title_text, True, (255, 42, 109))
        # Glow
        for gi in range(3, 0, -1):
            glow_surf = title_font.render(title_text, True, (255, 42, 109))
            gs = pygame.Surface(glow_surf.get_size(), pygame.SRCALPHA)
            gs.blit(glow_surf, (0, 0))
            gs.set_alpha(30)
            self.screen.blit(gs, (W//2 - glow_surf.get_width()//2 - gi, panel_y + 28 - gi))
            self.screen.blit(gs, (W//2 - glow_surf.get_width()//2 + gi, panel_y + 28 + gi))
        self.screen.blit(title_surf, (W//2 - title_surf.get_width()//2, panel_y + 28))
        
        # Final score in green like game.py
        score_font = pygame.font.Font(None, 90)
        score_text = f"FINAL SCORE: {self.score:,}"
        score_surf = score_font.render(score_text, True, (0, 255, 80))
        self.screen.blit(score_surf, (W//2 - score_surf.get_width()//2, panel_y + 160))
        
        # Stats
        stats_font = pygame.font.Font(None, 38)
        jumps = self.score // 5
        time_used = int(self.timer)
        diff_name = self.settings.selected_difficulty.name
        time_text = f"Time: {time_used//60:02d}:{time_used%60:02d} / {self.settings.selected_time//60:02d}:{self.settings.selected_time%60:02d}"
        
        for i, stat in enumerate([
            f"Successful Jumps: {jumps}",
            f"Difficulty: {diff_name}",
            time_text
        ]):
            s = stats_font.render(stat, True, (200, 230, 255))
            self.screen.blit(s, (W//2 - s.get_width()//2, panel_y + 268 + i * 50))
        
        self.play_again_btn.draw(self.screen)
        self.menu_btn.draw(self.screen)
        self.view_report_btn.draw(self.screen)

    def _render_report(self):
        W, H = self.settings.window_width, self.settings.window_height
        ModernUI.draw_gradient_background(self.screen, (10, 20, 40), (30, 60, 90))
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))

        ModernUI.draw_glass_panel(self.screen,
            (100, 50, W - 200, H - 150),
            (30, 30, 45, 230), (0, 255, 245), 2, 20)

        text = self.analytics.get_summary_text()
        y_off = 80
        font = pygame.font.Font(None, 34)
        for line in text.split('\n'):
            color = (0, 255, 245) if '\u2500' not in line else (200, 200, 255)
            if '\u2500' in line:
                pygame.draw.line(self.screen, (60, 60, 80),
                                 (140, y_off + 10), (W - 140, y_off + 10))
                y_off += 25
                continue
            surf = font.render(line, True, color)
            self.screen.blit(surf, (140, y_off))
            y_off += 35

        self.report_back_btn.draw(self.screen)

    def run(self):
        while True:
            dt = self.clock.tick(self.settings.fps) / 1000.0
            self.handle_events()
            self.update(dt)
            self.render()

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        self.hands.close()
        cv2.destroyAllWindows()
        pygame.quit()

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("LILY LEAP - MODERN REHABILITATION EDITION")
    print("="*70)
    print("Controls:")
    print("  • Tilt hand LEFT/RIGHT to steer")
    print("  • Make a FIST to jump when aligned")
    print("  • ESC to return to menu")
    print("="*70 + "\n")
    
    try:
        game = GameEngine()
        game.run()
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'game' in locals():
            game.cleanup()
        print("\nGame closed. Thank you!")