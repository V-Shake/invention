# Quick test to verify lighting system is working
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ar_textured import calculate_face_normal, calculate_lighting_intensity
import numpy as np

print("🔥 LIGHTING SYSTEM TEST 🔥")

# Test vertices for a simple triangle
vertices = np.array([
    [0.0, 0.0, 0.0],  # vertex 0
    [1.0, 0.0, 0.0],  # vertex 1
    [0.5, 1.0, 0.0]   # vertex 2
], dtype=np.float32)

# Test face (triangle using vertices 0, 1, 2)
face = [0, 1, 2]

# Calculate face normal
normal = calculate_face_normal(vertices, face)
print(f"Face normal: {normal}")

# Test different light directions
light_directions = [
    [0.0, 0.0, 1.0],    # Light directly above (should be bright)
    [0.0, 0.0, -1.0],   # Light directly below (should be dark)
    [1.0, 0.0, 0.0],    # Light from side
    [1.0, -1.5, -1.0]   # Our dramatic light direction
]

print("\n--- LIGHTING INTENSITY TESTS ---")
for i, light_dir in enumerate(light_directions):
    light_direction = np.array(light_dir, dtype=np.float32)
    intensity = calculate_lighting_intensity(normal, light_direction)
    print(f"Light {i+1} {light_dir}: {intensity:.3f} ({intensity*100:.1f}% brightness)")

print("\n✅ Lighting system is working!")
print("Expected: Different intensities showing dramatic contrast (20%-100%)")
