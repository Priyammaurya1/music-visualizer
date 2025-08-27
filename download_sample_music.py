"""
Simple script to create a sample music file for testing the visualizer.
Since we can't distribute copyrighted music, this creates a sample track.
"""

import numpy as np
import soundfile as sf

def create_sample_music():
    """Create a sample music track"""
    # Parameters
    duration = 120  # 2 minutes
    sample_rate = 44100
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a more complex musical composition
    audio = np.zeros_like(t)
    
    # Bass line (moving root notes)
    bass_notes = [55, 65.4, 73.4, 82.4]  # A, C, D, E (low octave)
    for i, note in enumerate(bass_notes):
        start_time = i * (duration / len(bass_notes))
        end_time = (i + 1) * (duration / len(bass_notes))
        mask = (t >= start_time) & (t < end_time)
        
        # Add harmonic content to bass
        audio[mask] += 0.3 * np.sin(2 * np.pi * note * t[mask])
        audio[mask] += 0.15 * np.sin(2 * np.pi * note * 2 * t[mask])  # Octave
    
    # Chord progression
    chord_progression = [
        [220, 261.63, 329.63],  # Am
        [261.63, 329.63, 392],  # C
        [293.66, 369.99, 440],  # D
        [329.63, 415.30, 493.88]  # E
    ]
    
    for i, chord in enumerate(chord_progression):
        start_time = i * (duration / len(chord_progression))
        end_time = (i + 1) * (duration / len(chord_progression))
        mask = (t >= start_time) & (t < end_time)
        
        for freq in chord:
            # Add some rhythm
            rhythm = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * t[mask])  # 2 Hz rhythm
            audio[mask] += 0.1 * np.sin(2 * np.pi * freq * t[mask]) * rhythm
    
    # Lead melody (higher frequency)
    melody_notes = [440, 493.88, 523.25, 587.33, 659.25, 698.46, 783.99, 880]  # A major scale
    melody_pattern = [0, 2, 4, 2, 1, 3, 5, 4, 2, 0]  # Note indices
    
    note_duration = duration / len(melody_pattern) / 2  # Each note plays for half the allocated time
    
    for i, note_idx in enumerate(melody_pattern):
        start_time = i * (duration / len(melody_pattern))
        end_time = start_time + note_duration
        mask = (t >= start_time) & (t < end_time)
        
        if np.any(mask):
            freq = melody_notes[note_idx]
            # Envelope for smoother notes
            envelope = np.exp(-3 * (t[mask] - start_time) / note_duration)
            audio[mask] += 0.15 * np.sin(2 * np.pi * freq * t[mask]) * envelope
    
    # Add some percussion-like elements
    beat_times = np.arange(0, duration, 0.5)  # Beat every 0.5 seconds
    for beat_time in beat_times:
        beat_idx = int(beat_time * sample_rate)
        if beat_idx < len(audio):
            # Kick drum simulation
            kick_length = int(0.1 * sample_rate)
            kick_envelope = np.exp(-10 * np.linspace(0, 1, kick_length))
            kick_sound = 0.2 * np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, kick_length)) * kick_envelope
            
            end_idx = min(beat_idx + kick_length, len(audio))
            actual_length = end_idx - beat_idx
            audio[beat_idx:end_idx] += kick_sound[:actual_length]
    
    # Add hi-hat like sounds
    hihat_times = np.arange(0.25, duration, 0.25)  # Every quarter beat
    for hihat_time in hihat_times:
        hihat_idx = int(hihat_time * sample_rate)
        if hihat_idx < len(audio):
            hihat_length = int(0.05 * sample_rate)
            hihat_envelope = np.exp(-20 * np.linspace(0, 1, hihat_length))
            # High frequency noise for hi-hat
            hihat_sound = 0.05 * np.random.normal(0, 1, hihat_length) * hihat_envelope
            
            end_idx = min(hihat_idx + hihat_length, len(audio))
            actual_length = end_idx - hihat_idx
            audio[hihat_idx:end_idx] += hihat_sound[:actual_length]
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV file
    sf.write('sample_music.wav', audio, sample_rate)
    print("Created sample_music.wav - a 2-minute sample track for the visualizer")
    print("Duration: 2 minutes")
    print("Format: 44.1kHz WAV")

if __name__ == "__main__":
    create_sample_music()
