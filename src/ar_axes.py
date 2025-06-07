import cv2
import numpy as np
import math

class ARVisualizer:
    def __init__(self):
        # Camera calibration parameters (approximate values for typical webcam)
        self.camera_matrix = np.array([[800.0, 0.0, 320.0],
                                     [0.0, 800.0, 240.0],
                                     [0.0, 0.0, 1.0]], dtype=np.float32)
        
        # Distortion coefficients (assuming no distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Marker size in meters (2.5 inches ≈ 0.0635 meters)
        self.marker_size = 0.0635
        
        # 3D points for marker corners in marker coordinate system
        self.marker_3d_points = np.array([
            [-self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2,  self.marker_size/2, 0],
            [ self.marker_size/2, -self.marker_size/2, 0],
            [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)
        
        # 3D points for XYZ axes
        axis_length = self.marker_size * 1.5
        self.axis_3d_points = np.array([
            [0, 0, 0],                    # Origin
            [axis_length, 0, 0],          # X-axis
            [0, axis_length, 0],          # Y-axis
            [0, 0, -axis_length]          # Z-axis
        ], dtype=np.float32)
    
    def estimate_pose(self, corners):
        success, rvec, tvec = cv2.solvePnP(
            self.marker_3d_points,
            corners[0],
            self.camera_matrix,
            self.dist_coeffs
        )
        return success, rvec, tvec
    
    def draw_axes(self, frame, rvec, tvec):
        # Project 3D axis points to 2D
        axis_2d_points, _ = cv2.projectPoints(
            self.axis_3d_points, rvec, tvec,
            self.camera_matrix, self.dist_coeffs
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
    
    def get_pose_info(self, rvec, tvec):
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation = tvec.flatten()
        
        # Calculate Euler angles
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
        
        euler_angles = np.array([x, y, z]) * 180.0 / np.pi
        distance = np.linalg.norm(translation)
        
        return {
            'translation': translation,
            'euler_angles': euler_angles,
            'distance': distance
        }
    
    def draw_pose_info(self, frame, pose_info, marker_id, position=(10, 30)):
        x, y = position
        line_height = 25
        
        cv2.putText(frame, f"Marker ID: {marker_id}", (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        tx, ty, tz = pose_info['translation']
        cv2.putText(frame, f"Position: X={tx:.3f}, Y={ty:.3f}, Z={tz:.3f}", 
                   (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        rx, ry, rz = pose_info['euler_angles']
        cv2.putText(frame, f"Rotation: X={rx:.1f}, Y={ry:.1f}, Z={rz:.1f}", 
                   (x, y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Distance: {pose_info['distance']:.3f}m", 
                   (x, y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

def ar_main():
    print("AR Visualizer - XYZ Axes")
    print("Hold up ArUco markers in front of the camera")
    print("Press 'q' to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    ar_viz = ARVisualizer()
    ar_viz.camera_matrix[0, 2] = frame_width / 2
    ar_viz.camera_matrix[1, 2] = frame_height / 2
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    
                    success, rvec, tvec = ar_viz.estimate_pose([corner])
                    
                    if success:
                        frame = ar_viz.draw_axes(frame, rvec, tvec)
                        pose_info = ar_viz.get_pose_info(rvec, tvec)
                        frame = ar_viz.draw_pose_info(frame, pose_info, marker_id, 
                                                    (10, 30 + i * 120))
                        
                        print(f"Marker {marker_id}: Distance={pose_info['distance']:.3f}m")
            
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
