"""
Visual Lighting Test - Create a simple scene to demonstrate lighting effects
"""
import cv2
import numpy as np

def calculate_face_normal(vertices, face):
    """Calculate the normal vector of a face for lighting calculations"""
    if len(face) < 3:
        return np.array([0, 0, 1])
    
    v1 = vertices[face[0]]
    v2 = vertices[face[1]]
    v3 = vertices[face[2]]
    
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    normal = np.cross(edge1, edge2)
    norm_length = np.linalg.norm(normal)
    if norm_length > 0:
        normal = normal / norm_length
    else:
        normal = np.array([0, 0, 1])
    
    return normal

def calculate_lighting_intensity(normal, light_direction):
    """Calculate lighting intensity based on surface normal and light direction"""
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    light_direction = light_direction / (np.linalg.norm(light_direction) + 1e-6)
    
    dot_product = np.dot(normal, light_direction)
    intensity = max(0.0, dot_product)
    
    # Enhanced contrast for more visible shadows
    ambient = 0.1  # Very dark shadows
    diffuse = 0.9  # High contrast
    
    return ambient + diffuse * intensity

def create_test_cube():
    """Create a simple cube for lighting demonstration"""
    # Define cube vertices
    vertices = np.array([
        [-0.02, -0.02, -0.02],  # 0: bottom-left-back
        [ 0.02, -0.02, -0.02],  # 1: bottom-right-back
        [ 0.02,  0.02, -0.02],  # 2: top-right-back
        [-0.02,  0.02, -0.02],  # 3: top-left-back
        [-0.02, -0.02,  0.02],  # 4: bottom-left-front
        [ 0.02, -0.02,  0.02],  # 5: bottom-right-front
        [ 0.02,  0.02,  0.02],  # 6: top-right-front
        [-0.02,  0.02,  0.02],  # 7: top-left-front
    ])
    
    # Define cube faces (triangles)
    faces = [
        # Front face
        [4, 5, 6], [4, 6, 7],
        # Back face
        [1, 0, 3], [1, 3, 2],
        # Left face
        [0, 4, 7], [0, 7, 3],
        # Right face
        [5, 1, 2], [5, 2, 6],
        # Top face
        [3, 7, 6], [3, 6, 2],
        # Bottom face
        [0, 1, 5], [0, 5, 4],
    ]
    
    return vertices, faces

def lighting_demo():
    """Demonstrate lighting on a simple cube"""
    print("🔆 Visual Lighting Test")
    print("This demonstrates the lighting system working on a simple cube")
    print("You should see different faces with different brightness levels")
    print("Press 'q' to quit")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera parameters (approximate)
    camera_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Create test cube
    vertices, faces = create_test_cube()
    
    # Light direction (same as AR app)
    light_direction = np.array([0.8, -1.0, -0.6])
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Get pose estimation
            marker_size = 0.05  # 5cm marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            
            rvec = rvec[0]
            tvec = tvec[0]
            
            # Project cube vertices
            projected_points, _ = cv2.projectPoints(
                vertices, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_points = projected_points.reshape(-1, 2).astype(int)
            
            # Draw cube faces with lighting
            for i, face in enumerate(faces):
                # Get face vertices
                face_3d = [vertices[j] for j in face]
                face_2d = [projected_points[j] for j in face]
                
                # Calculate lighting
                face_normal = calculate_face_normal(vertices, face)
                lighting_intensity = calculate_lighting_intensity(face_normal, light_direction)
                
                # Color based on face (for identification)
                base_colors = [
                    (0, 100, 200),    # Front faces - blue
                    (0, 100, 200),
                    (200, 100, 0),    # Back faces - orange
                    (200, 100, 0),
                    (0, 200, 100),    # Left faces - green
                    (0, 200, 100),
                    (200, 0, 100),    # Right faces - purple
                    (200, 0, 100),
                    (100, 200, 0),    # Top faces - yellow-green
                    (100, 200, 0),
                    (100, 0, 200),    # Bottom faces - purple-blue
                    (100, 0, 200),
                ]
                
                base_color = base_colors[i % len(base_colors)]
                
                # Apply lighting
                lit_color = tuple(int(c * lighting_intensity) for c in base_color)
                lit_color = tuple(max(0, min(255, c)) for c in lit_color)
                
                # Draw face
                face_points = np.array(face_2d, dtype=np.int32)
                cv2.fillPoly(frame, [face_points], lit_color)
                cv2.polylines(frame, [face_points], True, (255, 255, 255), 1)
                
                # Show lighting intensity as text
                center = np.mean(face_2d, axis=0).astype(int)
                cv2.putText(frame, f"{lighting_intensity:.2f}", 
                           tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add instructions
        cv2.putText(frame, "Hold ArUco marker to see lighting demo", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Numbers show lighting intensity (0.1=dark, 1.0=bright)", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Lighting Demo - Cube with Shadows', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    lighting_demo()
