"""
Lighting System Demonstration for AR Breadboard
Shows how lighting creates shadows based on surface angles
"""
import numpy as np
import matplotlib.pyplot as plt

def calculate_lighting_intensity(normal, light_direction):
    """Calculate lighting intensity based on surface normal and light direction"""
    # Ensure vectors are normalized
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    light_direction = light_direction / (np.linalg.norm(light_direction) + 1e-6)
    
    # Calculate dot product (cosine of angle between normal and light)
    dot_product = np.dot(normal, light_direction)
    
    # Clamp to positive values (surfaces facing away from light get ambient light only)
    intensity = max(0.0, dot_product)
    
    # Add ambient lighting (minimum brightness)
    ambient = 0.3
    diffuse = 0.7
    
    return ambient + diffuse * intensity

def demo_lighting_effects():
    """Demonstrate how lighting creates shadows and highlights"""
    print("🔆 AR Lighting System - Creating Realistic Shadows")
    print("=" * 60)
    
    # Light direction (top-right-front, same as in AR app)
    light_direction = np.array([0.5, -0.7, -0.5])
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    print(f"💡 Light Direction: {light_direction}")
    print(f"   Coming from: Top-Right-Front")
    print()
    
    # Test different surface orientations
    test_surfaces = [
        ("🔼 Surface facing UP (horizontal top)", np.array([0, 0, 1])),
        ("🔽 Surface facing DOWN (horizontal bottom)", np.array([0, 0, -1])),
        ("▶️ Surface facing RIGHT (vertical side)", np.array([1, 0, 0])),
        ("◀️ Surface facing LEFT (vertical side)", np.array([-1, 0, 0])),
        ("⬆️ Surface facing FORWARD (front face)", np.array([0, -1, 0])),
        ("⬇️ Surface facing BACKWARD (back face)", np.array([0, 1, 0])),
        ("🔍 Surface directly facing light", light_direction),
        ("🔄 Surface at 45° angle", np.array([0.707, 0, 0.707])),
    ]
    
    base_color = np.array([100, 150, 80])  # Example breadboard green color
    
    print("Surface Orientation → Angle → Intensity → Brightness Effect")
    print("-" * 70)
    
    max_intensity = 0
    min_intensity = 1
    
    for description, normal in test_surfaces:
        intensity = calculate_lighting_intensity(normal, light_direction)
        final_color = (base_color * intensity).astype(int)
        
        # Calculate angle between normal and light
        angle = np.arccos(np.clip(np.dot(normal, light_direction), -1, 1)) * 180 / np.pi
        
        # Determine brightness description
        if intensity > 0.8:
            brightness = "🌟 VERY BRIGHT"
        elif intensity > 0.6:
            brightness = "☀️ BRIGHT"
        elif intensity > 0.4:
            brightness = "🌤️ MEDIUM"
        else:
            brightness = "🌑 DARK/SHADOW"
        
        max_intensity = max(max_intensity, intensity)
        min_intensity = min(min_intensity, intensity)
        
        print(f"{description:35} → {angle:5.1f}° → {intensity:.3f} → {brightness}")
    
    print()
    print("🎯 Key Insights:")
    print(f"• 🌟 Brightest surfaces: {max_intensity:.3f} intensity (small angles to light)")
    print(f"• 🌑 Darkest surfaces: {min_intensity:.3f} intensity (large angles to light)")
    print("• 📐 Smaller angle = Surface faces light directly = BRIGHTER")
    print("• 📐 Larger angle = Surface faces away from light = DARKER")
    print("• 🏠 Creates realistic 3D depth and shadow effects")
    print("• ⚡ 30% minimum ambient light prevents completely black surfaces")
    
    print()
    print("🎨 Visual Effect in AR:")
    print("• Top surfaces get more light (brighter)")
    print("• Bottom and back surfaces get less light (darker/shadowed)")
    print("• Creates convincing 3D appearance with depth and volume")
    print("• Breadboard looks like a real physical object!")

if __name__ == "__main__":
    demo_lighting_effects()
