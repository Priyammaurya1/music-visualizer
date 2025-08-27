# Gesture-Controlled Music Visualizer Demo

## Quick Start Guide

1. **Launch the Application**
   ```bash
   python main.py
   ```

2. **Position Yourself**
   - Sit or stand 2-3 feet from your camera
   - Ensure good lighting on your hands
   - Keep your hands visible in the camera frame

3. **Activate the Visualizer**
   - Show peace signs (✌️) with **both hands** simultaneously
   - You should see the visualizer activate with frequency bars and animations
   - The camera feed border will turn **green** when active

4. **Control Parameters**

   ### Speed Control (Left Hand)
   - Use your **left hand** thumb and index finger
   - **Close distance** = Slower playback (0.1x speed)
   - **Increase distance** = Faster playback (up to 3.0x speed)
   - Watch the blue "Speed" bar in the bottom-left

   ### Frequency/Bass Control (Right Hand)
   - Use your **right hand** thumb and index finger
   - **Close distance** = Less frequency boost (0.1x)
   - **Increase distance** = More frequency boost (up to 3.0x)
   - Watch the orange "Frequency" bar in the bottom-left

   ### Volume Control (Both Hands)
   - Move your **entire hands** closer or farther apart
   - **Hands close together** = Low volume (0%)
   - **Hands far apart** = High volume (100%)
   - Watch the green "Volume" bar in the bottom-left

## Visual Elements

- **Frequency Bars**: Real-time audio spectrum analysis
- **Center Circle**: Audio-reactive pulsing visualization
- **Waveform**: Audio waveform display at top and bottom
- **Camera Feed**: Small window showing your hands (top-right)
- **Control Overlay**: Real-time parameter values (bottom-left)

## Tips for Best Experience

1. **Consistent Gestures**: Keep peace signs stable for reliable detection
2. **Smooth Movements**: Make gradual changes for smooth parameter transitions
3. **Good Lighting**: Ensure your hands are well-lit for accurate tracking
4. **Stable Position**: Stay within camera frame for continuous control

## Keyboard Shortcuts

- **SPACE**: Toggle visualizer on/off (for testing)
- **ESC**: Exit application

## Troubleshooting

### Visualizer Won't Activate
- Ensure both hands are showing clear peace signs
- Check that all fingers are visible to the camera
- Improve lighting conditions
- Move closer to or further from camera

### Poor Gesture Recognition
- Clean camera lens
- Remove jewelry or accessories on hands
- Ensure hands contrast well with background
- Check that MediaPipe is detecting hand landmarks (visible dots on hands)

### Performance Issues
- Close other applications using camera/audio
- Ensure adequate system resources
- Lower camera resolution if needed
- Check that graphics drivers are updated

## Customization

The application can be easily customized by modifying `main.py`:

- **Colors**: Change color constants in the `__init__` method
- **Sensitivity**: Adjust distance scaling factors in `update_audio_parameters()`
- **Visualizer Effects**: Modify drawing methods for different visual styles
- **Audio**: Replace `load_test_audio()` with music file loading

Enjoy your gesture-controlled music visualizer experience!
