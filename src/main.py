import cv2
import numpy as np
from ar_test import ar_main
from ar_modern_ui import ar_main_modern
from ar_textured import ar_main_textured

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
                text_y = y_min - 10
                
                # Ensure text doesn't go off screen
                if text_y < 20:
                    text_y = y_max + 25
                
                # Draw component label
                cv2.putText(frame, component_name, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Get the center point of the marker
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))
                
                # Calculate rotation angle (simplified approach)
                # Using the vector from first corner to second corner
                dx = corner[0][1][0] - corner[0][0][0]
                dy = corner[0][1][1] - corner[0][0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Display additional marker information inside the bounding box
                info_text = f"ID: {marker_id}"
                position_text = f"X: {center_x}, Y: {center_y}"
                rotation_text = f"Angle: {angle:.1f}°"
                
                # Draw additional information on the frame
                cv2.putText(frame, info_text, (center_x - 50, center_y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, position_text, (center_x - 50, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(frame, rotation_text, (center_x - 50, center_y + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
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