@@ -0,0 +1,121 @@
"""
Debug Lighting System - Show exactly what's happening with face normals and lighting
"""
import cv2
import numpy as np
import math

def debug_face_normals_and_lighting():
    """Create a simple test to show how lighting should work with different face orientations"""
    print("🔧 DEBUG: Face Normal and Lighting Analysis")
    print("=" * 60)
    
    # Light direction (same as in AR app)
    light_direction = np.array([0.8, -0.6, -0.5])
    light_direction = light_direction / np.linalg.norm(light_direction)
    print(f"💡 Light Direction: {light_direction}")
    
    # Test different face normals (these represent different surfaces of a 3D object)
    test_faces = [
        ("🔼 TOP surface (facing up)", np.array([0, 0, 1])),           # Should be somewhat dark
        ("🔽 BOTTOM surface (facing down)", np.array([0, 0, -1])),     # Should be bright  
        ("▶️ RIGHT surface", np.array([1, 0, 0])),                     # Should be bright (light from right)
        ("◀️ LEFT surface", np.array([-1, 0, 0])),                    # Should be dark
        ("⬆️ FRONT surface", np.array([0, -1, 0])),                   # Should be bright (light from front)
        ("⬇️ BACK surface", np.array([0, 1, 0])),                     # Should be dark
        ("🎯 Surface facing light directly", light_direction),          # Should be brightest
        ("🔄 Surface facing away from light", -light_direction),       # Should be darkest
    ]
    
    print("Face Orientation → Normal → Dot Product → Angle → Expected Shadow")
    print("-" * 80)
    
    for description, normal in test_faces:
        # Calculate dot product
        dot_product = np.dot(normal, light_direction)
        
        # Calculate angle
        angle = np.arccos(np.clip(dot_product, -1, 1)) * 180 / np.pi
        
        # Calculate lighting intensity (same as AR app)
        if dot_product <= 0.0:
            lighting_intensity = 0.1  # 10% for shadows
            shadow_type = "🌑 SHADOW"
        else:
            intensity = dot_product ** 2
            lighting_intensity = 0.1 + (1.0 - 0.1) * intensity
            if lighting_intensity > 0.7:
                shadow_type = "☀️ BRIGHT"
            elif lighting_intensity > 0.4:
                shadow_type = "🌤️ MEDIUM"
            else:
                shadow_type = "🌑 SHADOW"
        
        print(f"{description:30} → {normal} → {dot_product:6.3f} → {angle:6.1f}° → {shadow_type} ({lighting_intensity:.3f})")
    
    print()
    print("🎯 What to Look For in AR:")
    print("• Red arrows = Surfaces in SHADOW (dot product ≤ 0)")
    print("• Green arrows = Surfaces that are LIT (dot product > 0)")
    print("• Face colors should range from very dark (0.1) to bright (1.0)")
    print("• If all faces look the same brightness, normals might be wrong!")

def test_simple_cube_lighting():
    """Test lighting with a simple cube to verify the system works"""
    print("\n🧊 SIMPLE CUBE LIGHTING TEST")
    print("=" * 40)
    
    # Define a simple cube with 6 faces
    # Each face has vertices and a normal
    cube_faces = [
        ("Front", np.array([0, -1, 0])),   # Facing towards camera
        ("Back", np.array([0, 1, 0])),    # Facing away from camera  
        ("Right", np.array([1, 0, 0])),   # Facing right
        ("Left", np.array([-1, 0, 0])),   # Facing left
        ("Top", np.array([0, 0, 1])),     # Facing up
        ("Bottom", np.array([0, 0, -1])), # Facing down
    ]
    
    # Light from top-right-front
    light_direction = np.array([0.8, -0.6, -0.5])
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    print(f"Light: {light_direction}")
    print("Face → Normal → Dot → Lighting → Expected Appearance")
    print("-" * 55)
    
    for face_name, normal in cube_faces:
        dot_product = np.dot(normal, light_direction)
        
        # Our lighting function
        if dot_product <= 0.0:
            lighting_intensity = 0.1
        else:
            intensity = dot_product ** 2
            lighting_intensity = 0.1 + 0.9 * intensity
        
        # Determine appearance
        if lighting_intensity > 0.7:
            appearance = "VERY BRIGHT ☀️"
        elif lighting_intensity > 0.5:
            appearance = "BRIGHT 🌤️"
        elif lighting_intensity > 0.3:
            appearance = "MEDIUM 🌥️"
        else:
            appearance = "DARK/SHADOW 🌑"
        
        print(f"{face_name:6} → {normal} → {dot_product:5.2f} → {lighting_intensity:.3f} → {appearance}")

if __name__ == "__main__":
    debug_face_normals_and_lighting()
    test_simple_cube_lighting()
    
    print("\n🚨 DEBUGGING CHECKLIST:")
    print("1. Are the face normals being calculated correctly?")
    print("2. Are the face normals pointing in the right direction?")
    print("3. Is the light direction vector correct?")
    print("4. Are faces being sorted properly (back-to-front)?")
    print("5. Are face colors actually different, or all the same?")
    print("\n💡 TIP: Look at the colored arrows in the AR view:")
    print("   Green arrows = faces that should be bright")
    print("   Red arrows = faces that should be dark/shadowed")