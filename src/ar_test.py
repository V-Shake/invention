import cv2
import numpy as np
import math
import os

def load_obj_model(obj_path):
    """Load 3D model from OBJ file"""
    try:
        if not os.path.exists(obj_path):
            print(f"Model file not found: {obj_path}")
            return None, None
        
        print(f"Loading 3D model: {obj_path}")
        
        # Simple OBJ parser that ignores materials
        vertices = []
        faces = []
        
        with open(obj_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):  # Vertex
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):  # Face
                    parts = line.split()
                    face_vertices = []
                    for part in parts[1:]:
                        # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                        vertex_index = int(part.split('/')[0]) - 1  # OBJ indices start at 1
                        face_vertices.append(vertex_index)
                    if len(face_vertices) >= 3:
                        faces.append(face_vertices)
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Convert faces to handle variable face sizes (triangulate if needed)
        triangulated_faces = []
        for face in faces:
            if len(face) == 3:
                triangulated_faces.append(face)
            elif len(face) == 4:  # Quad - convert to two triangles
                triangulated_faces.append([face[0], face[1], face[2]])
                triangulated_faces.append([face[0], face[2], face[3]])
            elif len(face) > 4:  # Polygon - fan triangulation
                for i in range(1, len(face) - 1):
                    triangulated_faces.append([face[0], face[i], face[i + 1]])
        
        faces = np.array(triangulated_faces, dtype=np.int32)
        
        print(f"Model loaded: {len(vertices)} vertices, {len(faces)} faces")
        
        # Scale model to fit on 2.5 inch marker
        # Get bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        size = max_coords - min_coords
        max_size = np.max(size)
          # Scale to fit within marker size (2.5 inches = 0.0635 meters)
        marker_size_meters = 0.0635
        scale_factor = (marker_size_meters * 1.2) / max_size  # 120% of marker size for bigger model
          # Center and scale the model
        center = (min_coords + max_coords) / 2
        vertices = (vertices - center) * scale_factor
        
        # Apply Blender coordinate system correction (rotate 90° around X-axis)
        # Blender: Z-up, Y-forward → OpenCV: Y-down, Z-forward
        rotation_90_x = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # Apply rotation to all vertices
        vertices = vertices @ rotation_90_x.T
        
        # Lift model slightly above marker (5mm)
        vertices[:, 2] += 0.005
        
        print(f"Model scaled by factor: {scale_factor:.4f}")
        print(f"Model bounds: X=({np.min(vertices[:,0]):.4f}, {np.max(vertices[:,0]):.4f}), "
              f"Y=({np.min(vertices[:,1]):.4f}, {np.max(vertices[:,1]):.4f}), "
              f"Z=({np.min(vertices[:,2]):.4f}, {np.max(vertices[:,2]):.4f})")
        
        return vertices, faces
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if vertices is None or len(vertices) == 0:
        return None
    
    # Project 3D points to 2D
    projected_points, _ = cv2.projectPoints(
        vertices, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(int)

def draw_3d_model(frame, vertices, faces, projected_vertices):
    """Draw 3D model faces on the frame"""
    if projected_vertices is None or faces is None:
        return frame
    
    try:
        # Draw each face
        for face in faces:
            if len(face) >= 3:  # Valid face (triangle or more)
                # Get 2D points for this face
                face_points = []
                valid_face = True
                
                for vertex_idx in face:
                    if vertex_idx < len(projected_vertices):
                        point = projected_vertices[vertex_idx]
                        # Check if point is within frame bounds
                        if 0 <= point[0] < frame.shape[1] and 0 <= point[1] < frame.shape[0]:
                            face_points.append(point)
                        else:
                            valid_face = False
                            break
                    else:
                        valid_face = False
                        break
                
                if valid_face and len(face_points) >= 3:
                    face_points = np.array(face_points, dtype=np.int32)
                    
                    # Draw filled polygon with breadboard-like colors
                    # Use a more realistic breadboard color (brownish-green)
                    breadboard_color = (20, 100, 40)  # Dark green-brown
                    cv2.fillPoly(frame, [face_points], breadboard_color)
                    
                    # Draw wireframe edges with copper-like color for traces
                    edge_color = (0, 165, 255)  # Orange/copper color
                    cv2.polylines(frame, [face_points], True, edge_color, 1)
        
        return frame
        
    except Exception as e:
        print(f"Error drawing 3D model: {e}")
        return frame

def draw_text_3d(frame, text, position_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Draw beautiful 3D text with modern UI styling"""
    try:
        # Project text position to 2D
        text_points, _ = cv2.projectPoints(
            np.array([position_3d], dtype=np.float32),
            rvec, tvec, camera_matrix, dist_coeffs
        )
        
        text_2d = tuple(text_points[0][0].astype(int))
        
        # Modern font settings - cleaner look
        font = cv2.FONT_HERSHEY_SIMPLEX  # Clean, modern font
        font_scale = 0.9
        thickness = 2
        
        # Modern color scheme
        text_color = (255, 255, 255)      # Pure white
        accent_color = (0, 120, 255)      # Modern blue
        bg_color = (20, 20, 30)           # Dark modern background
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate centered position
        text_x = text_2d[0] - text_width // 2
        text_y = text_2d[1]
        
        # Modern card-like background with rounded corners effect
        padding_x, padding_y = 20, 12
        corner_radius = 8
        
        # Background coordinates
        bg_x1 = text_x - padding_x
        bg_y1 = text_y - text_height - padding_y
        bg_x2 = text_x + text_width + padding_x
        bg_y2 = text_y + baseline + padding_y
        
        # Create modern glass-morphism effect
        overlay = frame.copy()
        
        # Main background with slight transparency
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        
        # Simulate rounded corners with multiple rectangles
        corner_size = corner_radius
        # Top-left corner
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x1 + corner_size, bg_y1 + corner_size), bg_color, -1)
        # Top-right corner  
        cv2.rectangle(overlay, (bg_x2 - corner_size, bg_y1), (bg_x2, bg_y1 + corner_size), bg_color, -1)
        # Bottom-left corner
        cv2.rectangle(overlay, (bg_x1, bg_y2 - corner_size), (bg_x1 + corner_size, bg_y2), bg_color, -1)
        # Bottom-right corner
        cv2.rectangle(overlay, (bg_x2 - corner_size, bg_y2 - corner_size), (bg_x2, bg_y2), bg_color, -1)
        
        # Modern accent border - thin and elegant
        border_thickness = 1
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), accent_color, border_thickness)
        
        # Subtle top highlight for depth
        highlight_color = (255, 255, 255, 40)  # Very subtle white
        cv2.line(overlay, (bg_x1 + 2, bg_y1 + 1), (bg_x2 - 2, bg_y1 + 1), (60, 60, 80), 1)
        
        # Blend with frame for glass effect
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add subtle glow effect behind text
        glow_offset = 1
        glow_color = (accent_color[0]//3, accent_color[1]//3, accent_color[2]//3)
        for offset in range(1, 3):
            cv2.putText(frame, text, (text_x + offset, text_y + offset), 
                       font, font_scale, glow_color, thickness + 1)
        
        # Main text with crisp rendering
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Modern minimal indicator - single elegant line
        indicator_y = bg_y1 + (bg_y2 - bg_y1) // 2
        cv2.line(frame, (bg_x1 + 6, indicator_y), (bg_x1 + 14, indicator_y), accent_color, 2)
        cv2.line(frame, (bg_x2 - 14, indicator_y), (bg_x2 - 6, indicator_y), accent_color, 2)
        
        return frame
        
    except Exception as e:
        print(f"Error drawing 3D text: {e}")
        return frame

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
    print("AR Visualizer - 3D Breadboard Model")
    print("Hold up ArUco markers in front of the camera")
    print("Press 'q' to quit")
    
    # Load 3D model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "assets", "models", "breadboard.obj")
    model_path = os.path.normpath(model_path)
    vertices, faces = load_obj_model(model_path)
    
    if vertices is None:
        print("Failed to load 3D model. Using axes visualization instead.")
        use_3d_model = False
    else:
        use_3d_model = True
    
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
    # Following your suggestion for 2.5 inch marker
    marker_3d_points = np.array([
        [0.0, 0.0, 0.0],                    # top left
        [marker_size, 0.0, 0.0],           # top right (2.5 inches)
        [marker_size, marker_size, 0.0],   # bottom right
        [0.0, marker_size, 0.0]            # bottom left
    ], dtype=np.float32)
    
    # Center the marker coordinate system
    marker_3d_points -= np.array([marker_size/2, marker_size/2, 0])
    
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
                        if use_3d_model:
                            # Project 3D model vertices to 2D
                            projected_vertices = project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs)
                            
                            # Draw 3D breadboard model
                            frame = draw_3d_model(frame, vertices, faces, projected_vertices)
                              # Draw 3D text above the model (higher position for better visibility)
                            text_position_3d = np.array([0.0, 0.0, 0.04])  # 4cm above marker
                            frame = draw_text_3d(frame, "BREADBOARD", text_position_3d, 
                                               rvec, tvec, camera_matrix, dist_coeffs)
                        else:
                            # Fallback to axes visualization
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
            model_status = "3D Model" if use_3d_model else "Axes Only"
            cv2.putText(frame, f"AR Mode: {model_status} - Press 'q' to quit", 
                       (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow('AR Visualizer - 3D Breadboard', frame)
            
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
