"""
Modern AR Application with Beautiful UI using PIL/Pillow for enhanced typography
"""
import cv2
import numpy as np
import math
import os
from PIL import Image, ImageDraw, ImageFont
import io
from camera_utils import get_camera_super_fast

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def create_clean_text_overlay(width, height, text, position=(100, 50)):
    """Create clean black text overlay without background"""
    # Create transparent overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        # Try to use a modern system font
        font_size = 28
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
    
    # Calculate position
    x, y = position
    x = x - text_width // 2  # Center horizontally
    
    # Draw clean black text only - no background
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))  # Pure black text
    
    return overlay

def blend_overlay_with_frame(frame, overlay):
    """Blend PIL overlay with OpenCV frame"""
    # Convert frame to PIL
    frame_pil = cv2_to_pil(frame)
    
    # Composite overlay onto frame
    composite = Image.alpha_composite(frame_pil.convert('RGBA'), overlay)
    
    # Convert back to OpenCV
    return pil_to_cv2(composite.convert('RGB'))

def draw_clean_text_3d(frame, text, position_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """Draw clean black 3D text using PIL for better typography"""
    try:
        # Project text position to 2D
        text_points, _ = cv2.projectPoints(
            np.array([position_3d], dtype=np.float32),
            rvec, tvec, camera_matrix, dist_coeffs
        )
        
        text_2d = tuple(text_points[0][0].astype(int))
        
        # Create clean text overlay
        height, width = frame.shape[:2]
        overlay = create_clean_text_overlay(width, height, text, text_2d)
        
        # Blend with frame
        frame = blend_overlay_with_frame(frame, overlay)
        
        return frame
        
    except Exception as e:
        print(f"Error drawing clean text: {e}")
        return frame

# Copy other functions from ar_test.py for completeness
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

def ar_main_modern():
    print("Clean AR Visualizer - Black Text with Custom Fonts")
    print("Hold up ArUco markers in front of the camera")
    print("Press 'q' to quit")
    
    # Load 3D model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "assets", "models", "breadboard.obj")
    model_path = os.path.normpath(model_path)
    vertices, faces = load_obj_model(model_path)
    
    if vertices is None:
        print("Failed to load 3D model. Using text-only visualization.")
        use_3d_model = False
    else:
        use_3d_model = True    # Initialize camera
    cap = get_camera_super_fast()
    if cap is None:
        print("Error: Could not initialize any camera")
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
                          # Draw clean black text above the model
                        text_position_3d = np.array([0.0, 0.0, 0.04])  # 4cm above marker
                        frame = draw_clean_text_3d(frame, "BREADBOARD", text_position_3d, 
                                                   rvec, tvec, camera_matrix, dist_coeffs)
              # Add clean instruction text
            model_status = "Clean UI + 3D Model" if use_3d_model else "Clean UI Only"
            
            # Create instruction overlay with clean black text
            instruction_overlay = create_clean_text_overlay(
                frame_width, frame_height, 
                f"{model_status} - Press 'q' to quit",
                (frame_width // 2, frame_height - 60)
            )
            frame = blend_overlay_with_frame(frame, instruction_overlay)
            
            cv2.imshow('Clean AR Visualizer - Black Text', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Clean AR Visualizer closed")

if __name__ == "__main__":
    ar_main_modern()
