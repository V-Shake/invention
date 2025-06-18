import cv2
import numpy as np
from ar_test import ar_main
from ar_modern_ui import ar_main_modern
from ar_textured import ar_main_textured
from PIL import Image, ImageDraw, ImageFont

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def create_modern_text_overlay(width, height, text, position, font_size=24, text_color=(0, 255, 255)):
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
    
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
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
        3: "1x LED",
        4: "1x 220 Ohm Resistor",
        5: "1x Potentiometer",
        6: "5x Jumper Wires"
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
                
                # Get bounding box coordinates
                corner_points = corner[0].astype(int)
                x_min = np.min(corner_points[:, 0])
                y_min = np.min(corner_points[:, 1])
                x_max = np.max(corner_points[:, 0])
                y_max = np.max(corner_points[:, 1])
                
                # Add padding to bounding box
                padding = 20
                x_min -= padding
                y_min -= padding
                x_max += padding
                y_max += padding
                  # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Get component label for this marker ID
                component_name = component_labels.get(marker_id, f"Unknown Component (ID: {marker_id})")
                
                # Calculate text position above the bounding box
                text_x = x_min
                text_y = y_min - 40  # More space for modern font
                
                # Ensure text doesn't go off screen
                if text_y < 40:
                    text_y = y_max + 40
                
                # Create modern font overlay for component label
                height, width = frame.shape[:2]
                component_overlay = create_modern_text_overlay(
                    width, height, component_name, (text_x, text_y), 
                    font_size=28, text_color=(0, 255, 255)  # Yellow color
                )
                frame = blend_overlay_with_frame(frame, component_overlay)
                
                # Get the center point of the marker
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))
                
                # Calculate rotation angle (simplified approach)
                # Using the vector from first corner to second corner
                dx = corner[0][1][0] - corner[0][0][0]
                dy = corner[0][1][1] - corner[0][0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Display additional marker information inside the bounding box with modern fonts
                info_text = f"ID: {marker_id}"
                position_text = f"X: {center_x}, Y: {center_y}"
                rotation_text = f"Angle: {angle:.1f}°"
                
                # Create modern font overlays for info text
                info_overlay = create_modern_text_overlay(
                    width, height, info_text, (center_x - 50, center_y - 30), 
                    font_size=18, text_color=(0, 255, 0)  # Green color
                )
                frame = blend_overlay_with_frame(frame, info_overlay)
                
                position_overlay = create_modern_text_overlay(
                    width, height, position_text, (center_x - 50, center_y - 5), 
                    font_size=16, text_color=(0, 255, 0)  # Green color
                )
                frame = blend_overlay_with_frame(frame, position_overlay)
                
                rotation_overlay = create_modern_text_overlay(
                    width, height, rotation_text, (center_x - 50, center_y + 20), 
                    font_size=16, text_color=(0, 255, 0)  # Green color
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