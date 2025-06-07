import cv2
import numpy as np
import math

def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, marker_size=0.0635):
    """Draw XYZ axes on the frame"""
    # 3D points for XYZ axes
    axis_length = marker_size * 1.5
    axis_3d_points = np.array([
        [0, 0, 0],                    # Origin
        [axis_length, 0, 0],          # X-axis
        [0, axis_length, 0],          # Y-axis
        [0, 0, -axis_length]          # Z-axis
    ], dtype=np.float32)
    
    # Project 3D axis points to 2D
    axis_2d_points, _ = cv2.projectPoints(
        axis_3d_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    axis_2d_points = np.int32(axis_2d_points).reshape(-1, 2)
    
    # Get points
    origin = tuple(axis_2d_points[0])
    x_axis = tuple(axis_2d_points[1])
    y_axis = tuple(axis_2d_points[2])
    z_axis = tuple(axis_2d_points[3])
    
    # Draw axes with arrows
    cv2.arrowedLine(frame, origin, x_axis, (0, 0, 255), 5, tipLength=0.3)  # X: Red
    cv2.arrowedLine(frame, origin, y_axis, (0, 255, 0), 5, tipLength=0.3)  # Y: Green
    cv2.arrowedLine(frame, origin, z_axis, (255, 0, 0), 5, tipLength=0.3)  # Z: Blue
    
    # Add labels
    cv2.putText(frame, 'X', (x_axis[0]+10, x_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, 'Y', (y_axis[0]+10, y_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, 'Z', (z_axis[0]+10, z_axis[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    return frame

def ar_main():
    print("AR Visualizer - XYZ Axes")
    print("Hold up ArUco markers in front of the camera")
    print("Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Camera calibration parameters (approximate values for typical webcam)
    camera_matrix = np.array([[800.0, 0.0, frame_width/2],
                             [0.0, 800.0, frame_height/2],
                             [0.0, 0.0, 1.0]], dtype=np.float32)
    
    # Distortion coefficients (assuming no distortion)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Marker size in meters (2.5 inches ≈ 0.0635 meters)
    marker_size = 0.0635
    
    # 3D points for marker corners in marker coordinate system
    marker_3d_points = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, _ = detector.detectMarkers(gray)
            
            # If markers are detected
            if ids is not None:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                print(f"Detected {len(ids)} marker(s): {ids.flatten()}")
                
                # Process each detected marker
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    
                    # Estimate pose using solvePnP
                    success, rvec, tvec = cv2.solvePnP(
                        marker_3d_points,
                        corner[0],
                        camera_matrix,
                        dist_coeffs
                    )
                    
                    if success:
                        # Draw XYZ axes
                        frame = draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, marker_size)
                        
                        # Calculate distance
                        distance = np.linalg.norm(tvec.flatten())
                        
                        # Convert rotation vector to rotation matrix for Euler angles
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        
                        # Calculate Euler angles (simplified)
                        sy = math.sqrt(rotation_matrix[0,0]**2 + rotation_matrix[1,0]**2)
                        singular = sy < 1e-6
                        
                        if not singular:
                            x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                            y = math.atan2(-rotation_matrix[2,0], sy)
                            z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
                        else:
                            x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                            y = math.atan2(-rotation_matrix[2,0], sy)
                            z = 0
                        
                        # Convert to degrees
                        euler_angles = np.array([x, y, z]) * 180.0 / np.pi
                        
                        # Display pose information on frame
                        text_y = 30 + i * 120
                        cv2.putText(frame, f"Marker ID: {marker_id}", (10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        tx, ty, tz = tvec.flatten()
                        cv2.putText(frame, f"Position: X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}", 
                                   (10, text_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.putText(frame, f"Rotation: X={euler_angles[0]:.1f}, Y={euler_angles[1]:.1f}, Z={euler_angles[2]:.1f}", 
                                   (10, text_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.putText(frame, f"Distance: {distance:.3f}m", 
                                   (10, text_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Print to console
                        print(f"Marker {marker_id}: Distance={distance:.3f}m, "
                              f"Angles=({euler_angles[0]:.1f}°, {euler_angles[1]:.1f}°, {euler_angles[2]:.1f}°)")
            
            # Add instructions
            cv2.putText(frame, "Hold ArUco marker in view - Press 'q' to quit", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('AR Visualizer - XYZ Axes', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("AR Visualizer closed")

if __name__ == "__main__":
    ar_main()
