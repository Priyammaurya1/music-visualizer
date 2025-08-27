# Gesture-Controlled Music Visualizer

A sophisticated real-time music visualizer controlled by hand gestures using computer vision. Inspired by TouchDesigner interfaces, this Python application provides an industry-level full-screen experience with finger gesture controls for music parameters.

## Features

- **Full-Screen Camera Integration**: Your camera feed fills the entire window with visualizer overlays
- **Gesture Recognition**: Activate visualizer with peace signs from both hands
- **Real-time Music Controls**:
  - Left hand thumb-index distance: Controls playback speed (0.1x - 3.0x)
  - Right hand thumb-index distance: Controls frequency boost (0.1x - 3.0x)
  - Both hands distance: Controls volume (0% - 100%)
- **Professional Visualizer Effects**:
  - Real-time FFT frequency spectrum bars overlaid on camera feed
  - Animated center circle with audio-reactive pulsing
  - Multi-directional waveform displays (top, left, right)
  - Particle effects and beat detection
  - Smooth color gradients and neon effects
- **Music File Support**: Automatically loads MP3, WAV, FLAC, and other audio formats
- **Resizable Window**: Fully resizable interface just like any desktop application

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `pyaudio` installation on Windows, you may need to install it using:
```bash
pip install pipwin
pipwin install pyaudio
```

## Usage

1. **Add your music** (optional): Place any MP3, WAV, FLAC, or other audio files in the same folder as the application. The app will automatically load the first music file it finds.

2. **Run the application**:
```bash
python main_enhanced.py
```

3. **Position yourself**: Sit or stand in front of your camera so your hands are clearly visible

4. **Activate the visualizer**: Show peace signs (✌️) with both hands simultaneously 

5. **Control the music**:
   - **Speed Control**: Adjust distance between thumb and index finger on your left hand
   - **Frequency Control**: Adjust distance between thumb and index finger on your right hand  
   - **Volume Control**: Move your hands closer together or farther apart

6. **Resize the window**: Drag the window edges to resize just like any other application

7. Press `ESC` to quit or `SPACE` to toggle visualizer (for testing)

## System Requirements

- Python 3.7+
- Webcam
- Windows/macOS/Linux

## Dependencies

- **OpenCV**: Camera capture and image processing
- **MediaPipe**: Hand tracking and landmark detection
- **Pygame**: Graphics rendering and window management
- **NumPy**: Numerical computations
- **SciPy**: Signal processing and FFT
- **PyAudio**: Audio processing
- **Librosa**: Advanced audio analysis and loading
- **Pydub**: Audio file format support
- **SoundFile**: Audio file I/O

## Controls Summary

| Gesture | Control | Range |
|---------|---------|-------|
| Both hands peace sign | Activate visualizer | On/Off |
| Left thumb-index distance | Playback speed | 0.1x - 3.0x |
| Right thumb-index distance | Frequency boost | 0.1x - 3.0x |
| Both hands distance | Volume | 0% - 100% |

## Technical Details

- **Frame Rate**: 60 FPS
- **Audio Processing**: Real-time FFT analysis
- **Hand Detection**: MediaPipe with 70% confidence threshold
- **Gesture Recognition**: Peace sign detection using finger landmark analysis
- **Audio Generation**: Synthetic multi-frequency test audio (expandable to load music files)

## Customization

The application is designed to be easily customizable:

- Modify colors in the `MusicVisualizerApp` class
- Adjust gesture sensitivity by changing confidence thresholds
- Add support for music file loading by extending the `load_test_audio()` method
- Customize visualizer effects in the drawing methods

## Troubleshooting

1. **Camera not detected**: Ensure your webcam is connected and not being used by another application
2. **Poor gesture recognition**: Ensure good lighting and clear hand visibility
3. **Performance issues**: Close other applications and ensure adequate system resources

## Future Enhancements

- Music file loading and playback
- Additional gesture controls
- Customizable visualizer themes
- Export visualization recordings
- MIDI controller integration

## License

This project is open source and available under the MIT License.
