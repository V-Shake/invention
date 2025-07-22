# ArUco markers Detection & AR Breadboard Visualizer

An augmented reality application that detects ArUco markers and visualizes electronic components with 3D models. This project was developed as part of our course **Invention Design 2** in the **Interaction Design program (4th semester)** at **HfG Schwäbisch Gmünd**.


## Overview

This AR application uses OpenCV to detect ArUco markers that represent real electronic components (Arduino, breadboard, resistors, LEDs, etc.) and overlays them with:
- Component identification labels
- 3D breadboard models with realistic textures
- Interactive AR visualization with multiple display modes

The system detects ArUco markers using OpenCV2 and renders 3D models with UV texture mapping for an immersive augmented reality experience.

## Features

- **ArUco Marker Detection**: Detects and identifies multiple electronic components
- **3D Model Visualization**: Renders textured 3D models over detected markers
- **Multiple AR Modes**: 
  - Basic marker detection with component labels
  - Enhanced 3D visualization
  - Clean UI with modern typography
  - Textured models with UV mapping
- **Real-time Performance**: Optimized rendering for smooth AR experience
- **Component Recognition**: Identifies Arduino, breadboard, LEDs, resistors, potentiometers, and jumper wires

## Installation

### Prerequisites

- Python 3.7 or higher
- A webcam or external camera
- Printed ArUco markers (DICT_6X6_250)

### Setup

1. **Clone or download this repository**
2. **Create a virtual environment (recommended)**:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install required dependencies**:
   ```powershell
   pip install opencv-python opencv-contrib-python numpy pillow
   ```

### Required Packages

- `opencv-python` - Computer vision and ArUco marker detection
- `opencv-contrib-python` - Additional OpenCV modules for ArUco
- `numpy` - Numerical computations and matrix operations  
- `pillow` - Enhanced typography and image processing

## Usage

### Running the Application

1. **Navigate to the project directory**:
   ```powershell
   cd path\to\invention
   ```

2. **Activate your virtual environment** (if using one):
   ```powershell
   venv\Scripts\activate
   ```

3. **Run the main application**:
   ```powershell
   python src\main.py
   ```

4. **Choose your preferred AR mode**:
   - **Option 1**: Basic ArUco marker detection with component labels
   - **Option 2**: AR 3D visualization with enhanced OpenCV rendering
   - **Option 3**: Clean UI with black text and custom fonts
   - **Option 4**: Textured AR with UV-mapped breadboard models

### ArUco Markers

Print ArUco markers using the DICT_6X6_250 dictionary. Each marker ID corresponds to a specific component:

- **ID 0**: Arduino Leonardo
- **ID 1**: Breadboard
- **ID 2**: LED
- **ID 3**: 220Ω Resistor
- **ID 4**: Potentiometer  
- **ID 5**: Jumper Wires

**Recommended marker size**: 2.5 inches (6.35 cm) for optimal detection.

### Controls

- **'q'**: Quit the application
- **Hold markers**: Position ArUco markers in front of the camera for detection
- **Multiple markers**: The system can detect and render multiple components simultaneously

## Project Structure

```
invention/
├── src/
│   ├── main.py              # Main application with menu system
│   ├── ar_test.py           # Basic AR visualization 
│   ├── ar_modern_ui.py      # Clean UI with enhanced typography
│   ├── ar_textured.py       # Advanced textured rendering
│   └── camera_utils.py      # Camera detection utilities
├── assets/
│   ├── models/
│   │   ├── breadboard.obj   # 3D breadboard model
│   │   └── *.glb            # Additional 3D models
│   └── images/
│       └── uv.jpeg          # UV texture for breadboard
└── README.md
```

## 3D Models

The 3D breadboard model used in this project is sourced from Sketchfab:
- **Model**: Arduino Breadboard Low Poly
- **Source**: https://skfb.ly/6XnzP
- **Credit**: Used for educational purposes in AR experimentation

The model features:
- UV-mapped textures for realistic appearance
- Optimized geometry for real-time rendering
- Proper scaling for ArUco marker alignment

## Technical Details

### AR Pipeline

1. **Camera Initialization**: Automatic detection of best available camera
2. **Marker Detection**: ArUco marker detection using OpenCV
3. **Pose Estimation**: 3D pose estimation using `solvePnP`
4. **3D Rendering**: Model projection and rendering with lighting
5. **UI Overlay**: Modern typography using PIL/Pillow

### Performance Optimizations

- Dynamic level-of-detail (LOD) rendering
- Face normal caching for lighting calculations
- Efficient depth sorting for proper rendering
- Optimized camera settings for real-time performance

### Rendering Features

- **Realistic Lighting**: Directional lighting with shadow calculations
- **UV Texture Mapping**: Detailed surface textures on 3D models
- **Depth Sorting**: Proper face rendering order for 3D models
- **Modern Typography**: Enhanced text rendering with system fonts

## Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Try different camera indices or ensure camera permissions
   - Use external USB camera for better performance

2. **Markers not detected**:
   - Ensure good lighting conditions
   - Print markers at recommended 2.5-inch size
   - Keep markers flat and unobstructed

3. **Poor performance**:
   - Close other applications using the camera
   - Try the basic detection mode (Option 1)
   - Use external camera with higher resolution

4. **Missing dependencies**:
   ```powershell
   pip install --upgrade opencv-python opencv-contrib-python numpy pillow
   ```

## Development

This project demonstrates:
- Computer vision techniques with OpenCV
- 3D graphics programming and rendering
- Real-time AR application development
- Modern UI design principles
- Performance optimization strategies

### Future Enhancements

- Support for additional 3D model formats
- Advanced lighting and shadow effects  
- Multi-marker calibration
- Mobile platform support
- Component animation and interaction

## Academic Context

**Course**: Invention Design 2  
**Program**: Interaction Design (4th Semester)  
**Institution**: HfG Schwäbisch Gmünd  
**Focus**: Exploring augmented reality as a medium for visualizing and understanding electronic components and circuits.

## License

This project is for educational purposes. 3D models are credited to their respective creators on Sketchfab.

---

*Developed with passion for new technology experimentation - starting with ArUco marker detection, then exploring AR visualization because the possibilities seemed fascinating*✨
