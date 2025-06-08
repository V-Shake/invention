"""
High Contrast Lighting Test - Make shadows very obvious
"""
import cv2
import numpy as np
from ar_textured import *

def extreme_lighting_test():
    """Test with extreme lighting contrast to make shadows very visible"""
    print("🌟 Extreme Lighting Test - Very Obvious Shadows")
    print("This will make shadows extremely dark and bright areas very bright")
    print("Hold up ArUco marker to see dramatic lighting effects")
    print("Press 'q' to quit")
    
    # Load model
    obj_path = "C:/edu/HfG/IG4/invention/assets/models/breadboard.obj"
    texture_path = "C:/edu/HfG/IG4/invention/assets/images/uv.jpeg"
    
    model_data = load_obj_with_textures(obj_path)
    if model_data[0] is None:
        print("Failed to load model!")
        return
    
    vertices, faces, texture_data = model_data
    texture_image = load_texture_image(texture_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Camera parameters
    camera_matrix = np.array([[800, 0, 320],
                             [0, 800, 240],
                             [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1))
    
    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    
    # Extreme lighting function
    def extreme_lighting_intensity(normal, light_direction):
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        light_direction = light_direction / (np.linalg.norm(light_direction) + 1e-6)
        
        dot_product = np.dot(normal, light_direction)
        intensity = max(0.0, dot_product)
        
        # EXTREME contrast - almost no ambient light
        ambient = 0.05  # Very dark shadows
        diffuse = 0.95  # Very bright highlights
        
        final_intensity = ambient + diffuse * intensity
        
        # Apply gamma correction for even more dramatic effect
        final_intensity = final_intensity ** 0.7  # Make brights brighter, darks darker
        
        return final_intensity
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None:
            marker_size = 0.05
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            
            if rvec is not None and tvec is not None:
                rvec = rvec[0]
                tvec = tvec[0]
                
                # Project vertices
                projected_vertices = project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs)
                
                if projected_vertices is not None:
                    # Light direction - strong directional light
                    light_direction = np.array([1.0, -1.0, -0.5])
                    light_direction = light_direction / np.linalg.norm(light_direction)
                    
                    # Draw model with extreme lighting
                    result_frame = frame.copy()
                    
                    # Sort faces by depth
                    face_depths = []
                    for i, face in enumerate(faces):
                        if len(face) >= 3:
                            z_sum = sum(vertices[vi][2] for vi in face if vi < len(vertices))
                            face_depths.append((z_sum / len(face), i))
                    
                    face_depths.sort(key=lambda x: x[0], reverse=True)
                    
                    for depth, face_idx in face_depths[:1000]:  # Limit for performance
                        face = faces[face_idx]
                        if len(face) >= 3:
                            # Get face points
                            face_points = []
                            valid_face = True
                            
                            for vertex_idx in face:
                                if vertex_idx < len(projected_vertices):
                                    point = projected_vertices[vertex_idx]
                                    if -1000 <= point[0] <= frame.shape[1] + 1000 and -1000 <= point[1] <= frame.shape[0] + 1000:
                                        face_points.append(point)
                                    else:
                                        valid_face = False
                                        break
                                else:
                                    valid_face = False
                                    break
                            
                            if valid_face and len(face_points) >= 3:
                                face_points = np.array(face_points, dtype=np.int32)
                                
                                # Calculate extreme lighting
                                face_normal = calculate_face_normal(vertices, face)
                                lighting_intensity = extreme_lighting_intensity(face_normal, light_direction)
                                
                                # Use bright base color to show effect
                                base_color = (50, 200, 100)  # Bright green
                                
                                # Apply extreme lighting
                                lit_color = tuple(int(c * lighting_intensity) for c in base_color)
                                lit_color = tuple(max(0, min(255, c)) for c in lit_color)
                                
                                # Draw face
                                cv2.fillPoly(result_frame, [face_points], lit_color)
                                
                                # Show intensity as color brightness indicator
                                if face_idx < 10:
                                    center = np.mean(face_points, axis=0).astype(int)
                                    cv2.putText(result_frame, f"{lighting_intensity:.2f}", 
                                               tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    
                    frame = result_frame
        
        # Add instructions
        cv2.putText(frame, "EXTREME LIGHTING TEST - Look for very dark and bright areas", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow('Extreme Lighting Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    extreme_lighting_test()
