import cv2
import numpy as np
from ar_test import ar_main

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
            
            # Calculate pose for each marker
            for i, corner in enumerate(corners):
                # Get the center point of the marker
                center_x = int(np.mean(corner[0][:, 0]))
                center_y = int(np.mean(corner[0][:, 1]))
                
                # Calculate rotation angle (simplified approach)
                # Using the vector from first corner to second corner
                dx = corner[0][1][0] - corner[0][0][0]
                dy = corner[0][1][1] - corner[0][0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Display marker information
                marker_id = ids[i][0]
                info_text = f"ID: {marker_id}"
                position_text = f"X: {center_x}, Y: {center_y}"
                rotation_text = f"Angle: {angle:.1f}°"
                
                # Draw information on the frame
                cv2.putText(frame, info_text, (center_x - 50, center_y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, position_text, (center_x - 50, center_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, rotation_text, (center_x - 50, center_y + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                
                print(f"Marker {marker_id}: Position({center_x}, {center_y}), Rotation: {angle:.1f}°")
        
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
    print("2. AR 3D Axes Visualization")
    print("3. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\nStarting Basic ArUco Marker Detection...")
                basic_marker_detection()
                break
            elif choice == "2":
                print("\nStarting AR 3D Axes Visualization...")
                ar_main()
                break
            elif choice == "3":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()