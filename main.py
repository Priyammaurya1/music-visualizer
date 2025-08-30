import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import math
from typing import Optional, Tuple, List
import os
import sys
import argparse
import wave
import pyaudio
from scipy import signal
from scipy.fft import fft
import colorsys
from pydub import AudioSegment
from pydub.playback import play
import librosa
import soundfile as sf

class EnhancedMusicVisualizerApp:
    def __init__(self):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Screen setup - Resizable Windows window
        self.SCREEN_WIDTH = 1000
        self.SCREEN_HEIGHT = 700
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("🎵 Music Visualizer - Drag corners to resize")
        
        # MediaPipe setup - OPTIMIZED for fast detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,  # Lower for faster detection
            min_tracking_confidence=0.5    # Lower for faster detection
        )
        
        # Performance optimization
        self.frame_skip = 0  # Skip frames for performance
        self.mp_draw = mp.solutions.drawing_utils
        
        # Input setup (camera or video)
        self.cap = None
        self.is_video = False
        # CLI: --video path.mp4 to replicate/look like reference
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--video", type=str, default=None)
        try:
            args, _ = parser.parse_known_args(sys.argv[1:])
            self.video_path = args.video
        except SystemExit:
            self.video_path = None
        if self.video_path and os.path.exists(self.video_path):
            self.init_video(self.video_path)
        else:
            self.init_camera()
        
        # Audio setup (PyAudio streaming)
        self.audio_thread = None
        self.audio_data = None
        self.original_audio_data = None
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.is_playing = False
        self.current_song = None
        self.audio_length = 0
        self.playback_speed = 1.0
        self.playback_pos = 0.0  # floating-point sample index
        self.last_output_chunk = np.zeros(self.chunk_size, dtype=np.int16)
        
        # PyAudio engine state
        self.pyaudio_instance = None
        self.audio_stream = None
        self.stop_event = threading.Event()
        self.lpf_y = 0.0  # LPF state
        self.lpf_cutoff_hz = 10000.0 # LPF cutoff in Hz
        # (Keep reverb/pan objects but disabled by default)
        self.reverb_buffer = np.zeros(int(self.sample_rate * 0.6), dtype=np.float32)
        self.reverb_idx = 0
        self.reverb_feedback = 0.0
        self.reverb_mix = 0.0
        self.pan_phase = 0.0
        
        # Gesture control variables
        self.visualizer_active = False
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.gesture_stability_counter = 0
        self.required_stability = 3  # Much faster activation - only 3 frames needed
        
        # Control parameters with smoothing
        self.speed = 1.0
        self.volume = 0.5
        self.frequency_boost = 1.0
        self.freq_sensitivity = 28.0  # higher -> small movement makes big change
        self.speed_smooth = 1.0
        self.volume_smooth = 0.5
        self.frequency_smooth = 1.0
        self.smoothing_factor = 0.1
        self.frequency_hz = 50.0
        
        # Visualization parameters
        self.fft_data = np.zeros(512)
        self.bars_count = 64
        self.bar_width = self.SCREEN_WIDTH // self.bars_count
        self.max_visual_bars = 128
        self.bar_energy = np.zeros(self.max_visual_bars, dtype=np.float32)  # per-line energy (normalized)
        self.bar_smoothing = 0.78  # slightly snappier
        # Equalizer-like bar dynamics
        self.max_bar_px = 100  # cap height in pixels
        self.bar_height = np.zeros(self.max_visual_bars, dtype=np.float32)
        self.bar_attack = 0.7   # faster rise
        self.bar_decay = 0.88   # slightly slower fall
        
        # Enhanced visual effects
        self.particles = []
        self.beat_detection_threshold = 0.0
        self.last_beat_time = 0
        self.color_shift = 0.0
        self.pulse_intensity = 0.0
        self.is_beat = False
        
        # Colors (professional dark theme)
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.NEON_BLUE = (0, 191, 255)
        self.NEON_GREEN = (0, 255, 127)
        self.NEON_PURPLE = (191, 0, 255)
        self.NEON_ORANGE = (255, 165, 0)
        self.NEON_PINK = (255, 20, 147)
        self.DARK_GRAY = (20, 20, 25)
        self.LIGHT_GRAY = (100, 100, 105)
        self.ACCENT_BLUE = (64, 156, 255)
        
        # Font setup
        self.font_large = pygame.font.Font(None, 54)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 28)
        self.font_tiny = pygame.font.Font(None, 20)
        
        # UI smoothing for stable overlays
        self.left_line_pos = None
        self.right_line_pos = None
        self.ui_smoothing = 0.4  # higher -> more stable

        # Load music file
        self.load_music_file()
        
        # Initialize camera
        self.init_camera()
        
        # Test camera
        if self.cap is not None and self.cap.isOpened():
            print(f"Camera initialized successfully: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        else:
            print("Warning: Camera initialization failed!")
        
        # Frame rate and timing
        self.clock = pygame.time.Clock()
        self.running = True
        self.frame_count = 0

    def init_camera(self):
        """Robust camera initialization with multiple backends and indices."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        indices_to_try = [0, 1, 2, 3]
        backends = [
            getattr(cv2, 'CAP_DSHOW', 0),  # DirectShow (Windows)
            getattr(cv2, 'CAP_MSMF', 0),   # Media Foundation (Windows)
            0                              # Default
        ]

        opened = False
        for backend in backends:
            for idx in indices_to_try:
                try:
                    if backend != 0:
                        cap = cv2.VideoCapture(idx, backend)
                    else:
                        cap = cv2.VideoCapture(idx)

                    if cap is not None and cap.isOpened():
                        # Configure basic properties
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        self.cap = cap
                        opened = True
                        break
                except Exception:
                    continue
            if opened:
                break

        if not opened:
            print("Warning: Could not open camera. The screen may appear black.")

    def init_video(self, path: str):
        """Initialize video file as input and loop it."""
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Failed to open video: {path}. Falling back to camera.")
                self.init_camera()
                return
            self.cap = cap
            self.is_video = True
            print(f"Video input: {path}")
        except Exception as e:
            print(f"Error opening video {path}: {e}. Falling back to camera.")
            self.init_camera()

    def load_music_file(self):
        """Load music file (supports MP3, WAV, etc.)"""
        # Look for music files in the current directory
        music_files = []
        supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(fmt) for fmt in supported_formats):
                music_files.append(file)
        
        if music_files:
            # Use the first found music file
            music_file = music_files[0]
            print(f"Loading music file: {music_file}")
            
            try:
                # Load audio using librosa for better compatibility
                audio_data, sr = librosa.load(music_file, sr=self.sample_rate, mono=True)
                
                # Convert to 16-bit integers
                self.original_audio_data = (audio_data * 32767).astype(np.int16)
                self.audio_data = self.original_audio_data.copy()
                self.audio_length = len(self.audio_data)
                self.audio_pos = 0
                self.current_song = music_file
                
                print(f"Successfully loaded: {music_file}")
                print(f"Duration: {len(audio_data) / sr:.1f} seconds")
                
            except Exception as e:
                print(f"Error loading {music_file}: {e}")
                self.load_fallback_audio()
        else:
            print("No music files found. Using fallback audio.")
            self.load_fallback_audio()
    
    def load_fallback_audio(self):
        """Generate fallback audio if no music file is found"""
        duration = 180  # 3 minutes
        sample_rate = self.sample_rate
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create a more musical fallback track
        audio = np.zeros_like(t)
        
        # Chord progression (Am - F - C - G)
        chord_duration = 4.0  # 4 seconds per chord
        chords = [
            [220, 261.63, 329.63],  # Am
            [174.61, 220, 261.63],  # F
            [261.63, 329.63, 392],  # C
            [196, 246.94, 293.66]   # G
        ]
        
        for i, chord in enumerate(chords):
            start_time = i * chord_duration
            end_time = (i + 1) * chord_duration
            
            # Repeat the progression
            for rep in range(int(duration / (len(chords) * chord_duration)) + 1):
                chord_start = start_time + rep * len(chords) * chord_duration
                chord_end = end_time + rep * len(chords) * chord_duration
                
                if chord_start >= duration:
                    break
                    
                chord_mask = (t >= chord_start) & (t < min(chord_end, duration))
                
                for freq in chord:
                    audio[chord_mask] += 0.15 * np.sin(2 * np.pi * freq * t[chord_mask])
        
        # Add bass line
        bass_pattern = np.sin(2 * np.pi * 55 * t) * np.sin(2 * np.pi * 0.5 * t)  # A bass with rhythm
        audio += 0.2 * bass_pattern
        
        # Add some high-frequency sparkle
        sparkle = 0.05 * np.sin(2 * np.pi * 1760 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
        audio += sparkle
        
        # Normalize and convert
        audio = audio / np.max(np.abs(audio)) * 0.7
        self.original_audio_data = (audio * 32767).astype(np.int16)
        self.audio_data = self.original_audio_data.copy()
        self.audio_length = len(self.audio_data)
        self.audio_pos = 0
        self.current_song = "Generated Music"

    def detect_gesture(self, landmarks) -> bool:
        """Detect index finger and thumb extended gesture (like in the screenshot)"""
        if not landmarks:
            return False
        
        # Get landmark positions
        tips = [landmarks[i] for i in [4, 8, 12, 16, 20]]  # Thumb, Index, Middle, Ring, Pinky tips
        mcps = [landmarks[i] for i in [2, 5, 9, 13, 17]]   # MCP joints
        
        # Check finger extensions - we want thumb and index finger extended
        thumb_extended = tips[0].y < mcps[0].y - 0.005  # Thumb extended (more sensitive)
        index_extended = tips[1].y < mcps[1].y - 0.005  # Index finger extended (more sensitive)
        
        # Other fingers should be folded (not extended) - more lenient
        middle_folded = tips[2].y > mcps[2].y + 0.005
        ring_folded = tips[3].y > mcps[3].y + 0.005
        pinky_folded = tips[4].y > mcps[4].y + 0.005
        
        # Check that thumb and index are extended while others are folded
        return (thumb_extended and index_extended and 
                middle_folded and ring_folded and pinky_folded)

    def calculate_finger_distance(self, landmarks, finger1_tip: int, finger2_tip: int) -> float:
        """Enhanced distance calculation with depth consideration"""
        if not landmarks:
            return 0.0
        
        tip1 = landmarks[finger1_tip]
        tip2 = landmarks[finger2_tip]
        
        # 3D distance calculation
        distance = math.sqrt((tip1.x - tip2.x)**2 + (tip1.y - tip2.y)**2 + (tip1.z - tip2.z)**2)
        return distance

    def update_audio_parameters(self):
        """Enhanced parameter updates - VISUAL ONLY to prevent restarts"""
        if not self.visualizer_active:
            return
        
        # Left hand controls frequency (pure frequency control)
        if self.left_hand_landmarks:
            left_distance = self.calculate_finger_distance(self.left_hand_landmarks, 4, 8)
            # Map pinch distance from [0.02, 0.2] to [50, 1000] Hz for intuitive control
            min_dist, max_dist = 0.02, 0.2
            normalized_dist = (left_distance - min_dist) / (max_dist - min_dist)
            normalized_dist = max(0.0, min(1.0, normalized_dist))  # Clip to 0-1 range

            self.lpf_cutoff_hz = 50.0 + (normalized_dist**2) * 950.0 # Exp mapping for LPF

            freq_hz = 50.0 + normalized_dist * 950.0
            self.frequency_hz = freq_hz
            # Convert to a visual boost factor for bars (kept modest)
            target_freq = 0.8 + (freq_hz / 1000.0) * 0.8
            old_freq = self.frequency_boost
            self.frequency_boost = target_freq
            
            # Show visual feedback only (no audio restart)
            if abs(old_freq - self.frequency_boost) > 0.1:
                print(f"🎛️ Frequency: {int(freq_hz)} Hz")
        
        # Right hand controls speed (VISUAL FEEDBACK ONLY)
        if self.right_hand_landmarks:
            right_distance = self.calculate_finger_distance(self.right_hand_landmarks, 4, 8)
            target_speed = max(0.5, min(2.0, right_distance * 8))
            old_speed = self.speed
            self.speed = target_speed  # Direct assignment for responsiveness
            self.playback_speed = self.speed
            
            # Show visual feedback only (no audio restart)
            if abs(old_speed - self.speed) > 0.1:
                print(f"⚡ Speed: {self.speed:.1f}x (visual only)")
        
        # Volume is controlled by number of lines (distance between finger lines)
        # This is handled in draw_volume_bars_between_hands() function

    def calculate_hand_distance(self, left_landmarks, right_landmarks) -> float:
        """Calculate distance between hands using multiple points"""
        if not left_landmarks or not right_landmarks:
            return 0.0
        
        # Use multiple points for more stable distance calculation
        left_points = [left_landmarks[0], left_landmarks[5], left_landmarks[17]]  # Wrist, Index MCP, Pinky MCP
        right_points = [right_landmarks[0], right_landmarks[5], right_landmarks[17]]
        
        total_distance = 0
        for lp, rp in zip(left_points, right_points):
            dist = math.sqrt((lp.x - rp.x)**2 + (lp.y - rp.y)**2 + (lp.z - rp.z)**2)
            total_distance += dist
        
        return total_distance / len(left_points)
    
    def apply_audio_effects(self):
        """Apply real-time audio effects based on gesture controls"""
        if self.original_audio_data is None:
            return
        
        # Reset to original audio data
        self.audio_data = self.original_audio_data.copy()
        
        # Apply volume effect
        if self.volume != 1.0:
            self.audio_data = (self.audio_data * self.volume).astype(np.int16)
        
        # Apply frequency boost (simple EQ)
        if self.frequency_boost != 1.0:
            # Simple frequency boost by amplifying
            boosted_audio = self.audio_data * self.frequency_boost
            self.audio_data = np.clip(boosted_audio, -32767, 32767).astype(np.int16)

    def process_frame(self):
        """Enhanced frame processing with stability checking"""
        if self.cap is None or not self.cap.isOpened():
            # Try to recover camera if it was lost
            print("Camera not opened, attempting to initialize...")
            self.init_camera()
            if self.cap is None or not self.cap.isOpened():
                print("Failed to initialize camera, returning black frame")
                return np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)

        ret, frame = self.cap.read()
        if self.is_video and not ret:
            # loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        if not ret:
            # Attempt one quick retry
            ret, frame = self.cap.read()
            if not ret:
                # Reinitialize camera if still failing
                print("Failed to read frame, attempting to recover camera...")
                self.init_camera()
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to recover camera, returning black frame")
                    return np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb_frame)
        
        # Reset hand landmarks
        current_left = None
        current_right = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            gestures_detected = 0
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                if self.detect_gesture(hand_landmarks.landmark):
                    gestures_detected += 1
                    
                    if hand_label == "Left":
                        current_right = hand_landmarks.landmark  # Camera mirrored
                    else:
                        current_left = hand_landmarks.landmark   # Camera mirrored
                
                # Draw hand overlays with proper labels
                self.draw_hand_overlay(frame, hand_landmarks, hand_label)
            
            # Stability checking for activation - need both hands showing the gesture
            if gestures_detected >= 2:
                self.gesture_stability_counter += 1
                if self.gesture_stability_counter >= self.required_stability:
                    if not self.visualizer_active:
                        self.visualizer_active = True
                        self.start_music_playback()  # Just start, don't restart
                        print("🎵 Visualizer activated - music playing!")
                    self.left_hand_landmarks = current_left
                    self.right_hand_landmarks = current_right
            else:
                self.gesture_stability_counter = max(0, self.gesture_stability_counter - 1)
                if self.gesture_stability_counter <= 0:
                    if self.visualizer_active:
                        self.visualizer_active = False
                        self.stop_music_playback()
                        print("⏹️ Visualizer deactivated - music stopped!")
        else:
            self.gesture_stability_counter = max(0, self.gesture_stability_counter - 2)
            if self.gesture_stability_counter <= 0:
                if self.visualizer_active:
                    self.visualizer_active = False
                    self.stop_music_playback()
        
        return frame

    def draw_hand_overlay(self, frame, hand_landmarks, hand_label):
        """Draw TouchDesigner-style overlays on hands"""
        if not self.visualizer_active:
            return
            
        h, w, _ = frame.shape
        landmarks = hand_landmarks.landmark
        
        # Get thumb tip (4) and index finger tip (8) positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        
        # Draw line between thumb and index finger
        cv2.line(frame, thumb_pos, index_pos, (255, 255, 255), 2)
        
        # Draw circles at finger tips
        cv2.circle(frame, thumb_pos, 8, (255, 255, 255), 2)
        cv2.circle(frame, index_pos, 8, (255, 255, 255), 2)
        
        # Calculate distance between fingers
        distance = math.sqrt((thumb_pos[0] - index_pos[0])**2 + (thumb_pos[1] - index_pos[1])**2)
        distance_normalized = distance / 100  # Normalize for display
        
        # Determine control type and value based on hand (camera is mirrored)
        if hand_label == "Left":  # This is actually right hand in camera view - controls speed
            control_type = "speed"
            control_value = f"{self.speed:.3f}"  # 3 decimals, e.g., 1.032
        else:  # This is actually left hand in camera view - controls frequency
            control_type = "frequency"
            control_value = f"{int(self.frequency_hz)}"  # percentage-like value
        
        # Position text above the hand - more minimal positioning
        text_x = (thumb_pos[0] + index_pos[0]) // 2
        text_y = min(thumb_pos[1], index_pos[1]) - 30
        
        # Draw control label with minimal iPhone style
        cv2.putText(frame, control_type, (text_x - 20, text_y - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Draw control value with clean styling
        cv2.putText(frame, control_value, (text_x - 8, text_y + 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def draw_volume_overlay(self, frame):
        """Draw volume text BELOW the thin lines"""
        if not self.visualizer_active or not (self.left_hand_landmarks and self.right_hand_landmarks):
            return
            
        h, w, _ = frame.shape
        
        # Get finger positions for control lines (same as the lines)
        left_thumb = self.left_hand_landmarks[4]
        left_index = self.left_hand_landmarks[8]
        right_thumb = self.right_hand_landmarks[4]
        right_index = self.right_hand_landmarks[8]
        
        # Calculate midpoints of finger control lines (same as lines positioning)
        left_mid_x = int((left_thumb.x + left_index.x) / 2 * w)
        left_mid_y = int((left_thumb.y + left_index.y) / 2 * h)
        right_mid_x = int((right_thumb.x + right_index.x) / 2 * w)
        right_mid_y = int((right_thumb.y + right_index.y) / 2 * h)
        
        # Center position between the finger lines
        center_x = (left_mid_x + right_mid_x) // 2
        center_y = (left_mid_y + right_mid_y) // 2
        
        # Draw volume text BELOW the lines
        volume_text = "Volume"
        volume_value = f"{int(self.volume * 15)}"
        
        # Position volume text ABOVE the thin lines (-60px above center)
        cv2.putText(frame, volume_text, (center_x - 40, center_y - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(frame, volume_value, (center_x - 10, center_y - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def start_music_playback(self):
        """Start streaming playback using PyAudio."""
        if self.is_playing or self.audio_data is None:
            return
        try:
            if self.pyaudio_instance is None:
                self.pyaudio_instance = pyaudio.PyAudio()
            
            self.stop_event.clear()
            self.playback_pos = 0.0
            
            def callback(in_data, frame_count, time_info, status):
                if self.stop_event.is_set() or self.audio_data is None:
                    return (np.zeros(frame_count * 2, dtype=np.int16).tobytes(), pyaudio.paContinue)
            
                # Resample audio to apply speed changes
                left = np.empty(frame_count, dtype=np.float32)
                for i in range(frame_count):
                    idx = int(self.playback_pos)
                    frac = self.playback_pos - idx
                    if idx + 1 >= len(self.audio_data):
                        self.playback_pos = 0.0
                        idx = 0
                        frac = 0.0
                    # Linear interpolation
                    s0 = self.audio_data[idx]
                    s1 = self.audio_data[idx + 1]
                    sample = (1.0 - frac) * s0 + frac * s1
                    left[i] = sample
                    self.playback_pos += max(0.5, min(2.0, self.speed))
                
                # Convert to float for processing
                l_float = left.astype(np.float32) / 32767.0

                # Store pre-effect chunk for FFT visualization
                self.last_output_chunk = (l_float * 32767.0).astype(np.int16)

                # Apply Low-Pass Filter based on gesture
                RC = 1.0 / (2 * math.pi * max(50.0, self.lpf_cutoff_hz))
                dt = 1.0 / self.sample_rate
                alpha_lp = dt / (RC + dt)
                y = np.empty_like(l_float)
                prev_y = self.lpf_y
                for i in range(frame_count):
                    x = l_float[i]
                    prev_y = prev_y + alpha_lp * (x - prev_y)
                    y[i] = prev_y
                self.lpf_y = prev_y

                # Apply volume with soft clipping
                y_vol = y * self.volume
                y_vol_clipped = np.tanh(y_vol)

                # Final stereo output
                left_out = np.clip(y_vol_clipped, -1, 1)
                right_out = left_out
                interleaved = np.empty(frame_count * 2, dtype=np.int16)
                interleaved[0::2] = (left_out * 32767).astype(np.int16)
                interleaved[1::2] = (right_out * 32767).astype(np.int16)
                return (interleaved.tobytes(), pyaudio.paContinue)
            
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=2,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=callback
            )
            self.audio_stream.start_stream()
            self.is_playing = True
            print("🎵 Music started (streaming)!")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
    
    def stop_music_playback(self):
        """Stop streaming playback."""
        if not self.is_playing:
            return
        try:
            self.stop_event.set()
            if self.audio_stream is not None:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            self.is_playing = False
            print("⏹️ Music stopped.")
        except Exception as e:
            print(f"Error stopping audio: {e}")
    
    def restart_music_from_beginning(self):
        """Reset playback position to beginning without reloading."""
        self.playback_pos = 0.0
        print("🔄 Music position reset to start")
    
    def apply_real_speed_change(self):
        """DISABLED - Speed change causes restarts, visual feedback only"""
        # This function is disabled to prevent music restarts
        # The speed value is shown visually but doesn't affect actual playback
        pass
    
    def apply_real_frequency_change(self):
        """DISABLED - Frequency change causes restarts, visual feedback only"""
        # This function is disabled to prevent music restarts  
        # The frequency value is shown visually but doesn't affect actual playback
        pass
    
    def apply_realtime_audio_effects(self):
        """Apply real-time audio effects for speed and frequency"""
        if not self.is_playing:
            return
            
        # For now, we'll focus on volume control which works
        # Speed and frequency would need more advanced audio processing
        # The FFT visualization will react to the parameter changes
        pass

    def generate_enhanced_fft(self):
        """Enhanced FFT generation using the last output audio chunk."""
        if self.last_output_chunk is None or len(self.last_output_chunk) == 0:
            return
        chunk = self.last_output_chunk.astype(np.float32)
        if len(chunk) < self.chunk_size:
            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
        windowed = chunk * np.hanning(len(chunk))
        fft_result = np.abs(fft(windowed))[:self.chunk_size//2]
        low_freq_energy = np.sum(fft_result[:50])
        mid_freq_energy = np.sum(fft_result[50:150])
        total_energy = low_freq_energy + mid_freq_energy * self.frequency_boost
        if total_energy > self.beat_detection_threshold * 1.3:
            current_time = time.time()
            beat_interval = 0.3 / max(self.speed, 0.5)
            if current_time - self.last_beat_time > beat_interval:
                self.last_beat_time = current_time
                self.pulse_intensity = min(1.0, total_energy / 10000)
                self.is_beat = True
        self.beat_detection_threshold = 0.85 * self.beat_detection_threshold + 0.15 * total_energy
        self.pulse_intensity *= 0.92
        if self.pulse_intensity < 0.1:
            self.is_beat = False
        
        smoothed_fft = fft_result[:512] * (1 + self.frequency_boost * 0.3)
        self.fft_data = 0.6 * self.fft_data + 0.4 * smoothed_fft

    def spawn_particles(self):
        """Spawn particles on beat detection with enhanced visual effects."""
        for _ in range(15):  # More particles for stronger beats
            particle = {
                'x': self.SCREEN_WIDTH // 2 + np.random.randint(-150, 150),
                'y': self.SCREEN_HEIGHT // 2 + np.random.randint(-150, 150),
                'vx': np.random.uniform(-8, 8),
                'vy': np.random.uniform(-8, 8),
                'life': 1.0,
                'size': np.random.randint(4, 12),
                'color': self.hsv_to_rgb(np.random.random(), 1.0, 1.0),
                'beat_intensity': self.pulse_intensity  # Store beat intensity
            }
            self.particles.append(particle)

    def update_particles(self):
        """Update particle system with beat-reactive behavior."""
        for particle in self.particles[:]:
            # Update position
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            
            # Apply beat-reactive effects
            if self.is_beat:
                # Particles pulse with the beat
                beat_scale = 1.0 + self.pulse_intensity * 0.5
                particle['size'] *= beat_scale
                particle['vx'] *= 1.02  # Slight acceleration on beat
                particle['vy'] *= 1.02
            
            # Decay life and size
            particle['life'] -= 0.015
            particle['size'] *= 0.98
            
            # Add some gravity effect
            particle['vy'] += 0.1
            
            # Remove dead particles
            if particle['life'] <= 0 or particle['size'] < 1:
                self.particles.remove(particle)

    def draw_particles(self):
        """Draw particles with beat-reactive effects."""
        for particle in self.particles:
            if particle['life'] > 0:
                # Calculate alpha based on life
                alpha = int(particle['life'] * 255)
                
                # Beat-reactive color intensity
                if self.is_beat:
                    color_intensity = min(255, 200 + int(self.pulse_intensity * 55))
                    particle_color = (color_intensity, color_intensity, 255, alpha)
                else:
                    particle_color = (*particle['color'], alpha)
                
                # Draw particle with size variation
                size = int(particle['size'])
                if size > 0:
                    # Draw main particle
                    pygame.draw.circle(self.screen, particle_color, 
                                     (int(particle['x']), int(particle['y'])), size)
                    
                    # Add glow effect on strong beats
                    if self.is_beat and self.pulse_intensity > 0.6:
                        glow_size = size + 3
                        glow_color = (*particle_color[:3], 50)  # Semi-transparent
                        pygame.draw.circle(self.screen, glow_color,
                                         (int(particle['x']), int(particle['y'])), glow_size)

    def draw_enhanced_visualizer(self, camera_frame):
        """Draw enhanced visualizer overlay on camera feed"""
        if camera_frame is not None:
            # Draw volume overlay on camera frame first (before pygame conversion)
            if self.visualizer_active:
                self.draw_volume_overlay(camera_frame)
            
            # Convert to pygame and display
            camera_surface = self.convert_cv_to_pygame(camera_frame)
            camera_surface = pygame.transform.scale(camera_surface, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.screen.blit(camera_surface, (0, 0))
        
        if self.visualizer_active:
            # Generate enhanced FFT from last played audio
            self.generate_enhanced_fft()
            
            # Update color shift
            self.color_shift = (self.color_shift + 0.005) % 1.0
            
            # Update audio position
            self.audio_pos += self.speed * self.chunk_size / 60
            if self.audio_pos >= len(self.audio_data):
                self.audio_pos = 0
        
        # ALWAYS draw lines LAST so they appear on top of everything
        if self.visualizer_active:
            self.draw_visualizer_dots_between_hands()
    
    def convert_cv_to_pygame(self, cv_frame):
        """Convert OpenCV frame to pygame surface"""
        frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        # Use frombuffer for reliability
        surface = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), 'RGB')
        return surface

    def draw_gradient_background(self):
        """Draw animated gradient background"""
        for y in range(0, self.SCREEN_HEIGHT, 4):
            progress = y / self.SCREEN_HEIGHT
            hue = (self.color_shift + progress * 0.1) % 1.0
            color = self.hsv_to_rgb(hue, 0.1, 0.05 + progress * 0.02)
            pygame.draw.rect(self.screen, color, (0, y, self.SCREEN_WIDTH, 4))

    def draw_visualizer_dots_between_hands(self):
        """Draw THIN vertical lines between finger control lines like in the image"""
        if not self.visualizer_active or not (self.left_hand_landmarks and self.right_hand_landmarks):
            return
        
        # Get finger positions for control lines (thumb and index finger)
        left_thumb = self.left_hand_landmarks[4]
        left_index = self.left_hand_landmarks[8]
        right_thumb = self.right_hand_landmarks[4]
        right_index = self.right_hand_landmarks[8]
        
        # Calculate midpoints of finger control lines
        raw_left_mid = (
            (left_thumb.x + left_index.x) / 2 * self.SCREEN_WIDTH,
            (left_thumb.y + left_index.y) / 2 * self.SCREEN_HEIGHT,
        )
        raw_right_mid = (
            (right_thumb.x + right_index.x) / 2 * self.SCREEN_WIDTH,
            (right_thumb.y + right_index.y) / 2 * self.SCREEN_HEIGHT,
        )
        # Exponential smoothing for stability
        if self.left_line_pos is None:
            self.left_line_pos = raw_left_mid
        else:
            lx = self.left_line_pos[0] * (1 - self.ui_smoothing) + raw_left_mid[0] * self.ui_smoothing
            ly = self.left_line_pos[1] * (1 - self.ui_smoothing) + raw_left_mid[1] * self.ui_smoothing
            self.left_line_pos = (lx, ly)
        if self.right_line_pos is None:
            self.right_line_pos = raw_right_mid
        else:
            rx = self.right_line_pos[0] * (1 - self.ui_smoothing) + raw_right_mid[0] * self.ui_smoothing
            ry = self.right_line_pos[1] * (1 - self.ui_smoothing) + raw_right_mid[1] * self.ui_smoothing
            self.right_line_pos = (rx, ry)
        left_mid_x, left_mid_y = int(self.left_line_pos[0]), int(self.left_line_pos[1])
        right_mid_x, right_mid_y = int(self.right_line_pos[0]), int(self.right_line_pos[1])
        
        # Calculate distance between finger control lines
        fingers_distance = math.sqrt((right_mid_x - left_mid_x)**2 + (right_mid_y - left_mid_y)**2)
        
        # Line count based on distance - closer hands = fewer lines, farther = more lines
        dot_count = max(3, min(30, int(fingers_distance // 15)))
        
        # Update volume based on line count (distance between hands)
        # Remove upper limit per request; downstream soft-limiter avoids harsh clipping
        self.volume = (dot_count / 15.0)
        
        # Map live FFT into per-line energies (visualizer behavior)
        if len(self.fft_data) > 0:
            # Smooth continuous spectrum and interpolate across bars for fluid motion
            spectrum = self.fft_data[:300].astype(np.float32)
            n = max(1, dot_count)
            for i in range(n):
                # log-ish mapping across frequency bins
                frac = i / max(n - 1, 1)
                band_f = 5 + frac * (len(spectrum) - 1)
                b0 = int(max(0, min(len(spectrum) - 1, math.floor(band_f))))
                b1 = int(max(0, min(len(spectrum) - 1, b0 + 1)))
                w = band_f - b0
                energy_raw = (1 - w) * spectrum[b0] + w * spectrum[b1]
                # Normalize and emphasize low/mid slightly; modulate by frequency depth
                energy = (energy_raw / 1200.0) * (0.55 + 0.45 * self.frequency_boost)
                # Temporal smoothing of spectral energy
                self.bar_energy[i] = (
                    self.bar_smoothing * self.bar_energy[i]
                    + (1.0 - self.bar_smoothing) * float(energy)
                )
                # Envelope: fast attack, slow decay for visualizer feel
                target_px = float(self.max_bar_px) * max(0.0, min(1.0, self.bar_energy[i]))
                if target_px > self.bar_height[i]:
                    self.bar_height[i] = (
                        (1 - self.bar_attack) * self.bar_height[i]
                        + self.bar_attack * target_px
                    )
                else:
                    self.bar_height[i] *= self.bar_decay
        
        # Draw dots between the finger control lines
        for i in range(dot_count):
            # Position dots evenly between left and right finger midpoints
            t = i / max(dot_count - 1, 1)  # Interpolation factor (0 to 1)
            dot_x = int(left_mid_x + t * (right_mid_x - left_mid_x))
            dot_y = int(left_mid_y + t * (right_mid_y - left_mid_y))

            if self.is_beat:
                # On beat: draw thin vertical line (2px wide, height based on beat intensity)
                line_height = int(max(8, min(20, self.pulse_intensity * 25)))
                pygame.draw.rect(self.screen, (255, 255, 255),
                               (dot_x - 1, dot_y - line_height // 2, 2, line_height))
            else:
                # No beat: draw small dot (2px diameter)
                pygame.draw.circle(self.screen, (255, 255, 255), (dot_x, dot_y), 1)

        print(f"Drew {dot_count} beat-reactive elements between hands, volume: {self.volume:.2f}")

    def draw_enhanced_center_overlay(self):
        """Draw clean center visualization like TouchDesigner"""
        if not self.visualizer_active:
            return
            
        # Remove ugly rings and spokes - keep it minimal like TouchDesigner
        pass

    def draw_enhanced_waveforms_overlay(self):
        """Draw clean waveforms like TouchDesigner"""
        if not self.visualizer_active or self.audio_data is None:
            return
        
        # Remove ugly waveforms for now - keep it clean like TouchDesigner
        pass



    def hsv_to_rgb(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to RGB"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(r * 255), int(g * 255), int(b * 255))

    def handle_events(self):
        """Handle events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.visualizer_active = not self.visualizer_active
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.SCREEN_WIDTH = event.w
                self.SCREEN_HEIGHT = event.h
                self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
                self.bar_width = self.SCREEN_WIDTH // self.bars_count

    def run(self):
        """Main enhanced application loop"""
        print("🎵 Gesture-Controlled Music Visualizer Started")
        print("Show index finger + thumb extended with both hands to activate the visualizer")
        print("Press ESC to quit")
        
        previous_active = self.visualizer_active
        while self.running:
            self.handle_events()
            self.update_audio_parameters()
            
            # Get camera frame
            camera_frame = self.process_frame()
            
            # Draw everything (camera feed + visualizer overlay)
            self.draw_enhanced_visualizer(camera_frame)

            # Track activation edge (placeholder for future audio engine hook)
            if self.visualizer_active != previous_active:
                previous_active = self.visualizer_active

            pygame.display.flip()
            self.clock.tick(60)
            self.frame_count += 1
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    app = EnhancedMusicVisualizerApp()
    app.run()

if __name__ == "__main__":
    main()
