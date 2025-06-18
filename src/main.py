import cv2
import numpy as np
from ar_test import ar_main
from ar_modern_ui import ar_main_modern
from ar_textured import ar_main_textured
from PIL import Image, ImageDraw, ImageFont
from camera_utils import get_camera_with_fallback

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def create_modern_text_overlay(width, height, text, position, font_size=24, text_color=(0, 255, 255), center_text=False):
    """Create modern text overlay with custom fonts"""
    # Create transparent overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        # Try to use modern system fonts (same as ar_modern_ui)
        fonts_to_try = [
            "arial.ttf",
            "calibri.ttf", 
            "segoeui.ttf",
            "helvetica.ttf"
        ]
        
        font = None
        for font_name in fonts_to_try:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except:
                continue
        
        # Fallback to default font if no system fonts found
        if font is None:
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()
    
    # Get text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Use provided position
    x, y = position
    
    # Center text horizontally if requested
    if center_text:
        x = x - text_width // 2
    
    # Convert BGR color to RGB for PIL
    r, g, b = text_color
    rgb_color = (b, g, r, 255)  # Convert BGR to RGB and add alpha
    
    # Draw text
    draw.text((x, y), text, font=font, fill=rgb_color)
    
    return overlay

def blend_overlay_with_frame(frame, overlay):
    """Blend PIL overlay with OpenCV frame"""
    # Convert frame to PIL
    frame_pil = cv2_to_pil(frame)
    
    # Composite overlay onto frame
    composite = Image.alpha_composite(frame_pil.convert('RGBA'), overlay)
    
    # Convert back to OpenCV
    return pil_to_cv2(composite.convert('RGB'))

def basic_marker_detection():
    """Original ArUco marker detection function"""
    print("Detection of ArUco Markers")
    print("Press 'q' to quit the application")
    
    # Initialize the best available camera
    cap = get_camera_with_fallback()
    if cap is None:
        print("Error: Could not initialize any camera")
        return
    
    # Load the predefined dictionary for ArUco markers
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Create ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
      # Define component labels for each marker ID
    component_labels = {
        0: "1x Arduino Leonardo",
        1: "1x Breadboard", 
        2: "1x LED",
        3: "1x 220Ω Resistor",
        4: "1x Potentiometer",
        5: "5x Jumper Wires"
    }
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to grab frame")
            break
        
        # Convert frame to grayscale for ArUco detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
          # If markers are detected
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
              # Process each detected marker
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                
                # Get the center point of the marker
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))
                center = (center_x, center_y)
                
                # Calculate rotation angle using the marker corners
                # Using the vector from first corner to second corner
                dx = corner[0][1][0] - corner[0][0][0]
                dy = corner[0][1][1] - corner[0][0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Calculate marker size for bounding box
                marker_width = np.linalg.norm(corner[0][1] - corner[0][0])
                marker_height = np.linalg.norm(corner[0][2] - corner[0][1])
                
                # Add padding to bounding box
                padding = 40
                bbox_width = marker_width + padding
                bbox_height = marker_height + padding
                bbox_size = (bbox_width, bbox_height)
                  # Draw rotated bounding box
                rotated_corners = draw_rotated_rectangle(frame, center, bbox_size, angle, (0, 255, 255), 2)
                
                # Get component label for this marker ID
                component_name = component_labels.get(marker_id, f"Unknown Component (ID: {marker_id})")
                
                # Calculate rotated text position below the bounding box (centered)
                text_position = calculate_rotated_text_position_below(center, bbox_size, angle, offset_distance=30)
                text_x, text_y = text_position
                
                # Ensure text doesn't go off screen
                height, width = frame.shape[:2]
                if text_y > height - 50:  # Too close to bottom
                    # Move text above the bounding box instead
                    text_position = calculate_rotated_text_position(center, bbox_size, angle, offset_distance=50)
                    text_x, text_y = text_position
                    if text_y < 40:  # Still too close to top
                        text_y = 40
                
                # Boundary check for horizontal position (with some margin for centered text)
                text_width, _ = get_text_dimensions(component_name, font_size=28)
                if text_x - text_width//2 < 10:
                    text_x = 10 + text_width//2
                elif text_x + text_width//2 > width - 10:
                    text_x = width - 10 - text_width//2
                
                # Create modern font overlay for component label (centered)
                component_overlay = create_modern_text_overlay(
                    width, height, component_name, (text_x, text_y), 
                    font_size=28, text_color=(0, 255, 255), center_text=True  # Yellow color, centered
                )
                frame = blend_overlay_with_frame(frame, component_overlay)
                
                # Display additional marker information inside the bounding box with modern fonts
                info_text = f"ID: {marker_id}"
                position_text = f"X: {center_x}, Y: {center_y}"
                rotation_text = f"Angle: {angle:.1f}°"
                  # Create modern font overlays for info text (centered on marker)
                info_overlay = create_modern_text_overlay(
                    width, height, info_text, (center_x, center_y - 30), 
                    font_size=18, text_color=(0, 255, 0), center_text=True  # Green color, centered
                )
                frame = blend_overlay_with_frame(frame, info_overlay)
                
                position_overlay = create_modern_text_overlay(
                    width, height, position_text, (center_x, center_y - 5), 
                    font_size=16, text_color=(0, 255, 0), center_text=True  # Green color, centered
                )
                frame = blend_overlay_with_frame(frame, position_overlay)                
                rotation_overlay = create_modern_text_overlay(
                    width, height, rotation_text, (center_x, center_y + 20), 
                    font_size=16, text_color=(0, 255, 0), center_text=True  # Green color, centered
                )
                frame = blend_overlay_with_frame(frame, rotation_overlay)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                print(f"Marker {marker_id} ({component_name}): Position({center_x}, {center_y}), Rotation: {angle:.1f}°")
        
        # Display the frame
        cv2.imshow('ArUco Marker Detection', frame)
        
        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed")

def draw_rotated_rectangle(frame, center, size, angle, color, thickness=2):
    """Draw a rotated rectangle around a marker"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate half dimensions
    half_width = size[0] / 2
    half_height = size[1] / 2
    
    # Define rectangle corners relative to center
    corners = np.array([
        [-half_width, -half_height],  # top-left
        [half_width, -half_height],   # top-right
        [half_width, half_height],    # bottom-right
        [-half_width, half_height]    # bottom-left
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Rotate corners
    rotated_corners = corners @ rotation_matrix.T
    
    # Translate to center position
    rotated_corners += np.array(center)
    
    # Convert to integer coordinates
    rotated_corners = rotated_corners.astype(int)
    
    # Draw the rotated rectangle
    cv2.polylines(frame, [rotated_corners], True, color, thickness)
    
    return rotated_corners

def calculate_rotated_text_position_below(center, size, angle, offset_distance=50):
    """Calculate text position below a rotated rectangle, centered"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate the bottom-center of the rotated rectangle
    half_height = size[1] / 2
    
    # Get the bottom-center point of the rotated rectangle
    bottom_center_local = np.array([0, half_height + offset_distance])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Rotate the bottom-center point
    rotated_bottom = bottom_center_local @ rotation_matrix.T
    
    # Translate to actual center position
    text_position = rotated_bottom + np.array(center)
    
    return text_position.astype(int)

def get_text_dimensions(text, font_size=28):
    """Get approximate text dimensions for positioning"""
    # Rough estimation: each character is about 0.6 * font_size wide
    # and height is approximately font_size
    width = len(text) * int(font_size * 0.6)
    height = font_size
    return width, height

def calculate_rotated_text_position(center, size, angle, offset_distance=50):
    """Calculate text position above a rotated rectangle"""
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate the top-center of the rotated rectangle
    half_height = size[1] / 2
    
    # Get the top-center point of the rotated rectangle
    top_center_local = np.array([0, -half_height - offset_distance])
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Rotate the top-center point
    rotated_top = top_center_local @ rotation_matrix.T
    
    # Translate to actual center position
    text_position = rotated_top + np.array(center)
    
    return text_position.astype(int)

def main():
    print("=== ArUco Marker Detection & AR Application ===")
    print("Choose an option:")
    print("1. Basic ArUco Marker Detection")
    print("2. AR 3D Visualization (Enhanced OpenCV)")
    print("3. AR Clean UI (Black Text, Custom Fonts)")
    print("4. AR Textured Model (UV Mapped Breadboard)")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                print("\nStarting Basic ArUco Marker Detection...")
                basic_marker_detection()
                break
            elif choice == "2":
                print("\nStarting AR 3D Visualization...")
                ar_main()
                break
            elif choice == "3":
                print("\nStarting Clean AR with Black Text...")
                ar_main_modern()
                break
            elif choice == "4":
                print("\nStarting Textured AR with UV Mapping...")
                ar_main_textured()
                break
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()