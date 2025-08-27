import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import time
import math
from typing import Optional, Tuple, List
import os
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
        
        # Screen setup (resizable)
        self.SCREEN_WIDTH = 1200
        self.SCREEN_HEIGHT = 800
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Gesture Music Visualizer")
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Camera setup
        self.cap = None
        self.init_camera()
        
        # Audio setup
        self.audio_thread = None
        self.audio_data = None
        self.original_audio_data = None
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.is_playing = False
        self.current_song = None
        self.audio_length = 0
        self.playback_speed = 1.0
        
        # Audio effects
        self.audio_engine = None
        self.audio_stream = None
        
        # Gesture control variables
        self.visualizer_active = False
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.gesture_stability_counter = 0
        self.required_stability = 10  # Frames needed for stable activation
        
        # Control parameters with smoothing
        self.speed = 1.0
        self.volume = 0.5
        self.frequency_boost = 1.0
        self.speed_smooth = 1.0
        self.volume_smooth = 0.5
        self.frequency_smooth = 1.0
        self.smoothing_factor = 0.1
        
        # Visualization parameters
        self.fft_data = np.zeros(512)
        self.bars_count = 64
        self.bar_width = self.SCREEN_WIDTH // self.bars_count
        
        # Enhanced visual effects
        self.particles = []
        self.beat_detection_threshold = 0.0
        self.last_beat_time = 0
        self.color_shift = 0.0
        self.pulse_intensity = 0.0
        
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
        """Enhanced parameter updates with smoothing and real-time audio effects"""
        if not self.visualizer_active:
            return
        
        # Left hand controls speed
        if self.left_hand_landmarks:
            left_distance = self.calculate_finger_distance(self.left_hand_landmarks, 4, 8)
            target_speed = max(0.1, min(3.0, left_distance * 8))
            self.speed = self.speed * (1 - self.smoothing_factor) + target_speed * self.smoothing_factor
            self.playback_speed = self.speed
        
        # Right hand controls frequency boost
        if self.right_hand_landmarks:
            right_distance = self.calculate_finger_distance(self.right_hand_landmarks, 4, 8)
            target_freq = max(0.1, min(3.0, right_distance * 8))
            self.frequency_boost = self.frequency_boost * (1 - self.smoothing_factor) + target_freq * self.smoothing_factor
        
        # Both hands distance controls volume
        if self.left_hand_landmarks and self.right_hand_landmarks:
            hands_distance = self.calculate_hand_distance(self.left_hand_landmarks, self.right_hand_landmarks)
            target_volume = max(0.0, min(1.0, hands_distance * 3))
            self.volume = self.volume * (1 - self.smoothing_factor) + target_volume * self.smoothing_factor
        
        # Apply real-time audio effects
        self.apply_audio_effects()

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
                
                # Enhanced hand landmark drawing
                self.draw_enhanced_landmarks(frame, hand_landmarks)
            
            # Stability checking for activation - need both hands showing the gesture
            if gestures_detected >= 2:
                self.gesture_stability_counter += 1
                if self.gesture_stability_counter >= self.required_stability:
                    self.visualizer_active = True
                    self.left_hand_landmarks = current_left
                    self.right_hand_landmarks = current_right
            else:
                self.gesture_stability_counter = max(0, self.gesture_stability_counter - 2)
                if self.gesture_stability_counter <= 0:
                    self.visualizer_active = False
        else:
            self.gesture_stability_counter = max(0, self.gesture_stability_counter - 3)
            if self.gesture_stability_counter <= 0:
                self.visualizer_active = False
        
        return frame

    def draw_enhanced_landmarks(self, frame, hand_landmarks):
        """Draw minimal hand landmarks (invisible tracking for gesture detection only)"""
        # Remove all visual hand tracking elements for clean UI
        # Hand detection still works in background for gesture control
        pass

    def generate_enhanced_fft(self):
        """Enhanced FFT generation with beat detection"""
        if self.audio_data is None:
            return
        
        chunk_start = int(self.audio_pos)
        chunk_end = min(chunk_start + self.chunk_size, len(self.audio_data))
        
        if chunk_end > chunk_start:
            chunk = self.audio_data[chunk_start:chunk_end]
            
            # Apply frequency boost
            if self.frequency_boost != 1.0:
                chunk = chunk * self.frequency_boost
                chunk = np.clip(chunk, -32767, 32767).astype(np.int16)
            
            if len(chunk) < self.chunk_size:
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), 'constant')
            
            # Enhanced windowing
            windowed = chunk * np.hanning(len(chunk))
            fft_result = np.abs(fft(windowed))[:self.chunk_size//2]
            
            # Beat detection
            current_energy = np.sum(fft_result[:50])  # Low frequency energy
            if current_energy > self.beat_detection_threshold * 1.5:
                current_time = time.time()
                if current_time - self.last_beat_time > 0.2:  # Minimum beat interval
                    self.last_beat_time = current_time
                    self.pulse_intensity = 1.0
                    self.spawn_particles()
            
            self.beat_detection_threshold = 0.9 * self.beat_detection_threshold + 0.1 * current_energy
            self.pulse_intensity *= 0.95  # Decay pulse
            
            # Smooth FFT data
            self.fft_data = 0.7 * self.fft_data + 0.3 * fft_result[:512]

    def spawn_particles(self):
        """Spawn particles on beat detection"""
        for _ in range(10):
            particle = {
                'x': self.SCREEN_WIDTH // 2 + np.random.randint(-100, 100),
                'y': self.SCREEN_HEIGHT // 2 + np.random.randint(-100, 100),
                'vx': np.random.uniform(-5, 5),
                'vy': np.random.uniform(-5, 5),
                'life': 1.0,
                'size': np.random.randint(3, 8),
                'color': self.hsv_to_rgb(np.random.random(), 1.0, 1.0)
            }
            self.particles.append(particle)

    def update_particles(self):
        """Update particle system"""
        for particle in self.particles[:]:
            particle['x'] += particle['vx']
            particle['y'] += particle['vy']
            particle['life'] -= 0.02
            particle['size'] *= 0.99
            
            if particle['life'] <= 0 or particle['size'] < 1:
                self.particles.remove(particle)

    def draw_enhanced_visualizer(self, camera_frame):
        """Draw enhanced visualizer overlay on camera feed"""
        if camera_frame is not None:
            # Resize camera frame to fit screen
            camera_surface = self.convert_cv_to_pygame(camera_frame)
            camera_surface = pygame.transform.scale(camera_surface, (self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            self.screen.blit(camera_surface, (0, 0))
        
        if self.visualizer_active:
            # Generate enhanced FFT
            self.generate_enhanced_fft()
            
            # Update color shift
            self.color_shift = (self.color_shift + 0.005) % 1.0
            
            # Draw clean visualizer elements (TouchDesigner style)
            self.draw_enhanced_frequency_bars_overlay()
            
            # Update audio position
            self.audio_pos += self.speed * self.chunk_size / 60
            if self.audio_pos >= len(self.audio_data):
                self.audio_pos = 0
    
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

    def draw_enhanced_frequency_bars_overlay(self):
        """Draw clean frequency bars like TouchDesigner"""
        if not self.visualizer_active:
            return
            
        # Center frequency bars vertically like in TouchDesigner
        center_x = self.SCREEN_WIDTH // 2
        center_y = self.SCREEN_HEIGHT // 2
        
        # Use fewer, wider bars for cleaner look
        bars_to_show = 32
        bar_width = 8
        bar_spacing = 12
        
        # Calculate total width to center the bars
        total_width = bars_to_show * bar_spacing
        start_x = center_x - total_width // 2
        
        for i in range(bars_to_show):
            fft_index = int(i * len(self.fft_data) / bars_to_show)
            magnitude = self.fft_data[fft_index] if fft_index < len(self.fft_data) else 0
            
            # Clean scaling
            bar_height = min(magnitude / 2000 * 150, 200)  # Max height 200px
            
            # Clean white bars like TouchDesigner
            bar_x = start_x + i * bar_spacing
            bar_y = center_y - bar_height // 2
            
            # Draw clean white bars
            if bar_height > 2:
                pygame.draw.rect(self.screen, (255, 255, 255), 
                               (bar_x, bar_y, bar_width, bar_height))

    def draw_enhanced_center_overlay(self):
        """Draw clean center visualization like TouchDesigner"""
        if not self.visualizer_active:
            return
            
        # Remove ugly rings and spokes - keep it minimal like TouchDesigner
        pass

    def draw_particles(self):
        """Remove particle effects for clean TouchDesigner look"""
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
