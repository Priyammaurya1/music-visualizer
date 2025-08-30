# Music Visualizer with Hand Gesture Control

A Python-based music visualizer that uses computer vision to detect hand gestures and create dynamic visualizations that respond to music beats.

![Music Visualizer Demo](https://i.imgur.com/example.gif)

## Features

- **Hand Gesture Control**: Control the visualizer using simple hand gestures
- **Beat-Reactive Visualization**: Clean vertical bars that expand and contract with the music
- **Real-time Audio Analysis**: FFT-based frequency analysis for responsive visualization
- **Dynamic Controls**:
  - Left hand controls frequency response
  - Right hand controls playback speed
  - Distance between hands controls volume and number of bars

## Requirements

- Python 3.8+
- OpenCV
- MediaPipe
- PyGame
- NumPy
- SciPy
- LibROSA
- PyAudio
- PyDub

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/music-visualizer.git
   cd music-visualizer
   ```

2. Install the required packages:
   ```
   pip install opencv-python mediapipe pygame numpy scipy librosa pyaudio pydub
   ```

3. Place your music files (MP3, WAV, etc.) in the project directory.

## Usage

1. Run the application:
   ```
   python main.py
   ```

2. To activate the visualizer:
   - Show both hands to the camera
   - Extend your thumb and index finger on both hands (pinch gesture)
   - The visualizer will activate and start playing music

3. Controls:
   - **Left Hand**: Controls frequency response (move thumb and index finger closer/further)
   - **Right Hand**: Controls playback speed (move thumb and index finger closer/further)
   - **Both Hands**: Distance between hands controls volume and number of bars
   - **Press ESC**: Exit the application

## How It Works

The application uses:

1. **MediaPipe** for hand landmark detection
2. **PyAudio** for real-time audio processing
3. **FFT (Fast Fourier Transform)** for frequency analysis
4. **PyGame** for visualization rendering

The visualizer displays thin vertical bars that respond to different frequency ranges of the music. During quiet moments, the bars appear as small dots, and they dynamically expand into lines as the music plays, creating a clean, professional visualization effect.

## Customization

You can customize the visualizer by modifying parameters in the code:

- Adjust frequency bands in the `frequency_bands` dictionary
- Modify visualization parameters like `bar_smoothing` and `max_bar_px`
- Change colors by updating the color constants

## Troubleshooting

- **Camera Issues**: If the camera doesn't initialize, try changing the camera index in the `init_camera` method
- **Audio Issues**: Make sure you have a music file in the project directory
- **Performance Issues**: Adjust `frame_skip` parameter for better performance on slower systems

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by professional music visualization tools like TouchDesigner
- Uses MediaPipe's hand tracking technology
