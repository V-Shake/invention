# ArUco markers Detection & AR Breadboard Visualizer

An augmented reality application that detects ArUco markers and visualizes electronic components with 3D models.

## 🎓 Project Background

Developed for course Invention Design 2 in the Interaction Design program (4th semester) at HfG Schwäbisch Gmünd. Created to test the functionality of our project concept CircuitSpace - an innovative approach to visualizing and understanding electronic components through augmented reality.

🔗 **University Project**: https://ausstellung.hfg-gmuend.de/s-2525/projekte/circuitspace/studiengang:ig

## Features

- **ArUco Marker Detection**: Real-time detection of electronic components
- **3D Model Visualization**: Renders textured 3D models over detected markers
- **Multiple AR Modes**: Basic detection, 3D visualization, and textured rendering
- **Component Recognition**: Arduino, breadboard, LEDs, resistors, potentiometers, jumper wires

## Installation

### Prerequisites
- Python 3.7+
- Webcam
- Printed ArUco markers (DICT_6X6_250)

### Setup
1. **Create virtual environment**:
   ```bash
   py -m venv .venv
   .\.venv\Scripts\activate.bat
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**Run the application**:
```bash
python src\main.py
```

**Choose AR mode**:
- **1**: Basic marker detection
- **2**: 3D visualization

**Controls**: Press 'q' to quit

## ArUco Markers

Print markers using DICT_6X6_250:
- **ID 0**: Arduino  
- **ID 1**: Breadboard
- **ID 2**: LED
- **ID 3**: Resistor
- **ID 4**: Potentiometer
- **ID 5**: Jumper Wires

## Project Structure

```
src/
├── main.py              # Main application
├── ar_test.py           # Basic AR mode
├── ar_modern_ui.py      # Enhanced UI mode  
├── ar_textured.py       # Textured rendering
└── camera_utils.py      # Camera utilities
assets/
├── models/              # 3D models (.obj, .glb)
└── images/              # Textures
```
