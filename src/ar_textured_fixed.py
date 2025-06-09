"""
AR Application with UV Texture Mapping for Realistic Breadboard Rendering
This version applies the uv.jpeg texture to replace the flat green color with proper lighting!
"""
import cv2
import numpy as np
import math
import os
from PIL import Image, ImageDraw, ImageFont
import io

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

def load_obj_with_textures(obj_path):
    """Load 3D model from OBJ file with UV texture coordinates"""
    try:
        if not os.path.exists(obj_path):
            print(f"Model file not found: {obj_path}")
            return None, None, None
        
        print(f"Loading 3D model with textures: {obj_path}")
        
        # Parse OBJ file
        vertices = []
        texture_coords = []
        faces = []
        face_texture_indices = []
        
        with open(obj_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('v '):  # Vertex
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vt '):  # Texture coordinate
                    parts = line.split()
                    if len(parts) >= 3:
                        # UV coordinates (0-1 range)
                        texture_coords.append([float(parts[1]), float(parts[2])])
                elif line.startswith('f '):  # Face
                    parts = line.split()
                    face_vertices = []
                    face_textures = []
                    
                    for part in parts[1:]:
                        # Handle face format: vertex/texture/normal
                        indices = part.split('/')
                        vertex_index = int(indices[0]) - 1  # OBJ indices start at 1
                        face_vertices.append(vertex_index)
                        
                        # Get texture coordinate index if available
                        if len(indices) >= 2 and indices[1]:
                            texture_index = int(indices[1]) - 1
                            face_textures.append(texture_index)
                        else:
                            face_textures.append(-1)  # No texture coordinate
                    
                    if len(face_vertices) >= 3:
                        faces.append(face_vertices)
                        face_texture_indices.append(face_textures)
        
        vertices = np.array(vertices, dtype=np.float32)
        texture_coords = np.array(texture_coords, dtype=np.float32)
        
        # Convert faces to handle variable face sizes (triangulate if needed)
        triangulated_faces = []
        triangulated_texture_indices = []
        
        for i, face in enumerate(faces):
            face_tex = face_texture_indices[i]
            
            if len(face) == 3:
                triangulated_faces.append(face)
                triangulated_texture_indices.append(face_tex)
            elif len(face) == 4:  # Quad - convert to two triangles
                triangulated_faces.append([face[0], face[1], face[2]])
                triangulated_faces.append([face[0], face[2], face[3]])
                triangulated_texture_indices.append([face_tex[0], face_tex[1], face_tex[2]])
                triangulated_texture_indices.append([face_tex[0], face_tex[2], face_tex[3]])
            elif len(face) > 4:  # Polygon - fan triangulation
                for j in range(1, len(face) - 1):
                    triangulated_faces.append([face[0], face[j], face[j + 1]])
                    triangulated_texture_indices.append([face_tex[0], face_tex[j], face_tex[j + 1]])
        
        faces = np.array(triangulated_faces, dtype=np.int32)
        face_texture_indices = np.array(triangulated_texture_indices, dtype=np.int32)
        
        print(f"Model loaded: {len(vertices)} vertices, {len(texture_coords)} texture coords, {len(faces)} faces")
        
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
        
        return vertices, faces, (texture_coords, face_texture_indices)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def load_texture_image(texture_path):
    """Load texture image for UV mapping"""
    try:
        if not os.path.exists(texture_path):
            print(f"Texture file not found: {texture_path}")
            return None
        
        # Load texture using OpenCV
        texture = cv2.imread(texture_path)
        if texture is None:
            print(f"Failed to load texture: {texture_path}")
            return None
        
        print(f"Texture loaded: {texture.shape[1]}x{texture.shape[0]} pixels")
        return texture
        
    except Exception as e:
        print(f"Error loading texture: {e}")
        return None

def sample_texture(texture, u, v):
    """Sample texture color at UV coordinates (u, v)"""
    if texture is None:
        return (20, 100, 40)  # Fallback green color
    
    # Clamp UV coordinates to [0, 1] range
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    
    # Convert UV to pixel coordinates
    # Note: V coordinate is flipped (1-v) because image Y axis is inverted
    height, width = texture.shape[:2]
    x = int(u * (width - 1))
    y = int((1.0 - v) * (height - 1))  # Flip V coordinate
    
    # Sample pixel color (BGR format)
    color = texture[y, x]
    return tuple(int(c) for c in color)

def project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if vertices is None or len(vertices) == 0:
        return None
    
    # Project 3D points to 2D
    projected_points, _ = cv2.projectPoints(
        vertices, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(int)

def calculate_face_normal(vertices, face):
    """Calculate the normal vector of a face for lighting calculations"""
    if len(face) < 3:
        return np.array([0, 0, 1])  # Default normal pointing up
    
    # Get three vertices of the face
    v1 = vertices[face[0]]
    v2 = vertices[face[1]]
    v3 = vertices[face[2]]
    
    # Calculate two edge vectors
    edge1 = v2 - v1
    edge2 = v3 - v1
    
    # Calculate normal using cross product
    normal = np.cross(edge1, edge2)
    
    # Normalize the normal vector
    norm_length = np.linalg.norm(normal)
    if norm_length > 0:
        normal = normal / norm_length
    else:
        normal = np.array([0, 0, 1])  # Default normal
    
    return normal

def calculate_lighting_intensity(normal, light_direction):
    """Calculate lighting intensity based on surface normal and light direction"""
    # Ensure vectors are normalized
    normal = normal / (np.linalg.norm(normal) + 1e-6)
    light_direction = light_direction / (np.linalg.norm(light_direction) + 1e-6)
    
    # Calculate dot product (cosine of angle between normal and light)
    dot_product = np.dot(normal, light_direction)
    
    # Dramatic lighting calculation for visible shadows
    if dot_product <= 0.0:
        # Faces facing away from light - dark shadows
        return 0.2  # 20% brightness for shadows
    else:
        # Faces facing the light - use power function for dramatic contrast
        intensity = dot_product ** 1.2  # Power function for contrast
        return 0.2 + 0.8 * intensity  # Range from 20% to 100% brightness

def draw_textured_3d_model(frame, vertices, faces, projected_vertices, texture_data, texture_image):
    """Draw 3D model faces with UV texture mapping and dramatic lighting"""
    if projected_vertices is None or faces is None:
        return frame
    
    texture_coords, face_texture_indices = texture_data
    
    # STRONG directional light for DRAMATIC shadows
    light_direction = np.array([1.0, -1.5, -1.0])  # From top-right-front
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    try:
        # Create a copy for drawing
        result_frame = frame.copy()
        
        # Sort faces by average Z-depth for proper rendering order
        face_depths = []
        
        for i, face in enumerate(faces):
            if len(face) >= 3:
                # Calculate average Z-depth of face vertices
                z_sum = 0
                valid_vertices = 0
                
                for vertex_idx in face:
                    if vertex_idx < len(vertices):
                        z_sum += vertices[vertex_idx][2]
                        valid_vertices += 1
                
                if valid_vertices > 0:
                    avg_depth = z_sum / valid_vertices
                    face_depths.append((avg_depth, i))
        
        # Sort faces by depth (back to front)
        face_depths.sort(key=lambda x: x[0], reverse=True)
        
        # Draw faces in depth order
        for depth, face_idx in face_depths:
            face = faces[face_idx]
            
            if len(face) >= 3:
                # Get 2D points for this face
                face_points = []
                valid_face = True
                
                for vertex_idx in face:
                    if vertex_idx < len(projected_vertices):
                        point = projected_vertices[vertex_idx]
                        # Check if point is within reasonable bounds
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
                    
                    # Calculate lighting for this face FIRST
                    face_normal = calculate_face_normal(vertices, face)
                    lighting_intensity = calculate_lighting_intensity(face_normal, light_direction)
                    
                    # Check if we have texture coordinates for this face
                    face_tex_indices = face_texture_indices[face_idx] if face_idx < len(face_texture_indices) else None
                    
                    if (face_tex_indices is not None and 
                        len(face_tex_indices) >= 3 and 
                        all(idx >= 0 and idx < len(texture_coords) for idx in face_tex_indices) and
                        texture_image is not None):
                        
                        # Get UV coordinates for this face
                        uv_coords = []
                        for tex_idx in face_tex_indices:
                            if tex_idx >= 0:
                                uv_coords.append(texture_coords[tex_idx])
                            else:
                                uv_coords.append([0.5, 0.5])  # Default UV
                        
                        # Sample texture at triangle center
                        if len(uv_coords) >= 3:
                            # Calculate average UV coordinate
                            avg_u = sum(uv[0] for uv in uv_coords[:3]) / 3
                            avg_v = sum(uv[1] for uv in uv_coords[:3]) / 3
                            
                            # Sample texture color
                            base_color = sample_texture(texture_image, avg_u, avg_v)
                        else:
                            base_color = (60, 120, 80)  # Fallback breadboard color
                        
                        # Apply DRAMATIC lighting to texture color
                        texture_color = tuple(int(c * lighting_intensity) for c in base_color)
                        
                        # Ensure minimum visibility and clamp to valid range
                        texture_color = tuple(max(10, min(255, c)) for c in texture_color)
                        
                        # Draw textured face with lighting
                        cv2.fillPoly(result_frame, [face_points], texture_color)
                        
                        # Draw subtle wireframe (also lit)
                        edge_color = tuple(max(5, int(c * 0.7)) for c in texture_color)
                        cv2.polylines(result_frame, [face_points], True, edge_color, 1)
                        
                    else:
                        # No texture - use default color with lighting
                        base_color = (60, 120, 80)  # Breadboard green
                        
                        # Apply lighting
                        lit_color = tuple(int(c * lighting_intensity) for c in base_color)
                        lit_color = tuple(max(10, min(255, c)) for c in lit_color)
                        
                        cv2.fillPoly(result_frame, [face_points], lit_color)
                        
                        # Wireframe
                        edge_color = tuple(max(5, int(c * 0.7)) for c in lit_color)
                        cv2.polylines(result_frame, [face_points], True, edge_color, 1)
        
        return result_frame
        
    except Exception as e:
        print(f"Error drawing textured 3D model: {e}")
        return frame

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

def ar_main_textured():
    print("🔥 TEXTURED AR with DRAMATIC LIGHTING! 🔥")
    print("Hold up ArUco markers in front of the camera")
    print("You should see realistic shadows and highlights!")
    print("Press 'q' to quit")
    
    # Load 3D model with texture coordinates
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "assets", "models", "breadboard.obj")
    model_path = os.path.normpath(model_path)
    vertices, faces, texture_data = load_obj_with_textures(model_path)
    
    # Load texture image
    texture_path = os.path.join(script_dir, "..", "assets", "images", "uv.jpeg")
    texture_path = os.path.normpath(texture_path)
    texture_image = load_texture_image(texture_path)
    
    if vertices is None:
        print("Failed to load 3D model. Exiting.")
        return
    
    if texture_image is None:
        print("Failed to load texture. Using default colors.")
        use_texture = False
    else:
        use_texture = True
        print("✅ Texture mapping AND lighting enabled!")
    
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
                        # Project 3D model vertices to 2D
                        projected_vertices = project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs)
                        
                        # Draw textured 3D breadboard model WITH DRAMATIC LIGHTING
                        frame = draw_textured_3d_model(frame, vertices, faces, projected_vertices, 
                                                     texture_data, texture_image)
                        
                        # Draw clean black text above the model
                        text_position_3d = np.array([0.0, 0.0, 0.04])  # 4cm above marker
                        frame = draw_clean_text_3d(frame, "SHADOWED BREADBOARD", text_position_3d, 
                                                 rvec, tvec, camera_matrix, dist_coeffs)
            
            # Add instruction text
            instruction_overlay = create_clean_text_overlay(
                frame_width, frame_height, 
                "🌞 DRAMATIC LIGHTING + UV TEXTURE - Press 'q' to quit",
                (frame_width // 2, frame_height - 60)
            )
            frame = blend_overlay_with_frame(frame, instruction_overlay)
            
            cv2.imshow('🔥 TEXTURED AR with DRAMATIC SHADOWS! 🔥', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Textured AR Visualizer closed")

if __name__ == "__main__":
    ar_main_textured()
