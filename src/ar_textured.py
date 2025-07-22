"""
AR Application with UV Texture Mapping - OPTIMIZED for Multiple Models
Reduced lag with efficient shadow calculations and multi-model support
"""
import cv2
import numpy as np
import math
import os
import time
from PIL import Image, ImageDraw, ImageFont
import io

# PERFORMANCE OPTIMIZATION: Global caches
_lighting_cache = {}
_face_normal_cache = {}

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def create_clean_text_overlay(width, height, text, position=(100, 50)):
    """Create clean black text overlay without background"""
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    try:
        font_size = 28
        fonts_to_try = ["arial.ttf", "calibri.ttf", "segoeui.ttf", "helvetica.ttf"]
        
        font = None
        for font_name in fonts_to_try:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except:
                continue
        
        if font is None:
            font = ImageFont.load_default()
            
    except Exception as e:
        print(f"Font loading error: {e}")
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x, y = position
    x = x - text_width // 2
    
    draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))
    
    return overlay

def blend_overlay_with_frame(frame, overlay):
    """Blend PIL overlay with OpenCV frame"""
    frame_pil = cv2_to_pil(frame)
    composite = Image.alpha_composite(frame_pil.convert('RGBA'), overlay)
    return pil_to_cv2(composite.convert('RGB'))

class Model3D:
    """Optimized 3D Model class for multiple model support"""
    
    def __init__(self, obj_path, texture_path=None, scale=1.0, position_offset=(0, 0, 0)):
        self.obj_path = obj_path
        self.texture_path = texture_path
        self.scale = scale
        self.position_offset = np.array(position_offset)
        
        # Load model data
        self.vertices, self.faces, self.texture_data = self.load_obj_with_textures()
        self.texture_image = self.load_texture_image() if texture_path else None
        
        # Pre-compute face normals for better performance
        self.face_normals = self.precompute_face_normals()
        
        # Performance tracking
        self.last_frame_time = 0
        self.skip_frames = 0  # For dynamic LOD
        
    def load_obj_with_textures(self):
        """Load 3D model from OBJ file with UV texture coordinates"""
        try:
            if not os.path.exists(self.obj_path):
                print(f"Model file not found: {self.obj_path}")
                return None, None, None
            
            print(f"Loading optimized 3D model: {self.obj_path}")
            
            vertices = []
            texture_coords = []
            faces = []
            face_texture_indices = []
            
            with open(self.obj_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('vt '):
                        parts = line.split()
                        if len(parts) >= 3:
                            texture_coords.append([float(parts[1]), float(parts[2])])
                    elif line.startswith('f '):
                        parts = line.split()
                        face_vertices = []
                        face_textures = []
                        
                        for part in parts[1:]:
                            indices = part.split('/')
                            vertex_index = int(indices[0]) - 1
                            face_vertices.append(vertex_index)
                            
                            if len(indices) >= 2 and indices[1]:
                                texture_index = int(indices[1]) - 1
                                face_textures.append(texture_index)
                            else:
                                face_textures.append(-1)
                        
                        if len(face_vertices) >= 3:
                            faces.append(face_vertices)
                            face_texture_indices.append(face_textures)
            
            vertices = np.array(vertices, dtype=np.float32)
            texture_coords = np.array(texture_coords, dtype=np.float32)
            
            # Triangulate faces
            triangulated_faces = []
            triangulated_texture_indices = []
            
            for i, face in enumerate(faces):
                face_tex = face_texture_indices[i]
                
                if len(face) == 3:
                    triangulated_faces.append(face)
                    triangulated_texture_indices.append(face_tex)
                elif len(face) == 4:
                    triangulated_faces.append([face[0], face[1], face[2]])
                    triangulated_faces.append([face[0], face[2], face[3]])
                    triangulated_texture_indices.append([face_tex[0], face_tex[1], face_tex[2]])
                    triangulated_texture_indices.append([face_tex[0], face_tex[2], face_tex[3]])
                elif len(face) > 4:
                    for j in range(1, len(face) - 1):
                        triangulated_faces.append([face[0], face[j], face[j + 1]])
                        triangulated_texture_indices.append([face_tex[0], face_tex[j], face_tex[j + 1]])
            
            faces = np.array(triangulated_faces, dtype=np.int32)
            face_texture_indices = np.array(triangulated_texture_indices, dtype=np.int32)
            
            # Scale and position model
            self.apply_transformations(vertices)
            
            print(f"✅ Model loaded: {len(vertices)} vertices, {len(faces)} faces")
            return vertices, faces, (texture_coords, face_texture_indices)
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None, None
    
    def apply_transformations(self, vertices):
        """Apply scaling, rotation and positioning to model"""
        # Get bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        size = max_coords - min_coords
        max_size = np.max(size)
        
        # Scale to fit marker size
        marker_size_meters = 0.0635
        scale_factor = (marker_size_meters * self.scale) / max_size
        
        # Center and scale
        center = (min_coords + max_coords) / 2
        vertices[:] = (vertices - center) * scale_factor
        
        # Apply Blender coordinate system correction
        rotation_90_x = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        vertices[:] = vertices @ rotation_90_x.T
        
        # Apply position offset and lift above marker
        vertices[:, 2] += 0.005 + self.position_offset[2]
        vertices[:, 0] += self.position_offset[0]
        vertices[:, 1] += self.position_offset[1]
    
    def load_texture_image(self):
        """Load texture image for UV mapping"""
        try:
            if not self.texture_path or not os.path.exists(self.texture_path):
                print(f"Texture file not found: {self.texture_path}")
                return None
            
            texture = cv2.imread(self.texture_path)
            if texture is None:
                print(f"Failed to load texture: {self.texture_path}")
                return None
            
            print(f"✅ Texture loaded: {texture.shape[1]}x{texture.shape[0]} pixels")
            return texture
            
        except Exception as e:
            print(f"Error loading texture: {e}")
            return None
    
    def precompute_face_normals(self):
        """Pre-compute face normals for better performance"""
        if self.vertices is None or self.faces is None:
            return None
        
        normals = []
        for face in self.faces:
            if len(face) >= 3:
                normal = self.calculate_face_normal_fast(face)
                normals.append(normal)
            else:
                normals.append(np.array([0, 0, 1]))
        
        return np.array(normals)
    
    def calculate_face_normal_fast(self, face):
        """Fast face normal calculation"""
        if len(face) < 3:
            return np.array([0, 0, 1])
        
        v1 = self.vertices[face[0]]
        v2 = self.vertices[face[1]]
        v3 = self.vertices[face[2]]
        
        edge1 = v2 - v1
        edge2 = v3 - v1
        
        normal = np.cross(edge1, edge2)
        norm_length = np.linalg.norm(normal)
        
        if norm_length > 0:
            return normal / norm_length
        else:
            return np.array([0, 0, 1])

def project_vertices(vertices, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D vertices to 2D image coordinates"""
    if vertices is None or len(vertices) == 0:
        return None
    
    projected_points, _ = cv2.projectPoints(
        vertices, rvec, tvec, camera_matrix, dist_coeffs
    )
    
    return projected_points.reshape(-1, 2).astype(int)

def calculate_lighting_intensity_fast(normal, light_direction):
    """OPTIMIZED lighting calculation with caching"""
    # Create cache key
    normal_key = tuple(np.round(normal, 3))
    light_key = tuple(np.round(light_direction, 3))
    cache_key = (normal_key, light_key)
    
    # Check cache
    if cache_key in _lighting_cache:
        return _lighting_cache[cache_key]
    
    # Calculate lighting
    dot_product = np.dot(normal, light_direction)
    
    if dot_product <= 0.0:
        intensity = 0.2  # 20% for shadows
    else:
        intensity = 0.2 + 0.8 * (dot_product ** 1.2)  # 20%-100% range
    
    # Cache result
    _lighting_cache[cache_key] = intensity
    return intensity

def sample_texture_fast(texture, u, v):
    """Fast texture sampling"""
    if texture is None:
        return (60, 120, 80)  # Default breadboard color
    
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    
    height, width = texture.shape[:2]
    x = int(u * (width - 1))
    y = int((1.0 - v) * (height - 1))
    
    color = texture[y, x]
    return tuple(int(c) for c in color)

def draw_model_optimized(frame, model, projected_vertices, light_direction, frame_count=0):
    """OPTIMIZED model drawing with dynamic LOD"""
    if projected_vertices is None or model.faces is None:
        return frame
      # BALANCED optimization - prioritize showing the model
    current_time = time.time()
    frame_time = current_time - model.last_frame_time
    model.last_frame_time = current_time
    
    # Only skip if REALLY lagging (much more lenient)
    if frame_time > 0.1:  # Only if frame time > 100ms (very slow)
        model.skip_frames = min(model.skip_frames + 1, 1)  # Max skip only 1 frame
    else:
        model.skip_frames = 0  # Always reset to show model
    
    # Rarely skip rendering - only in extreme lag
    if model.skip_frames > 0 and frame_count % 10 == 0:  # Only skip every 10th frame max
        return frame
    
    try:
        result_frame = frame.copy()
        texture_coords, face_texture_indices = model.texture_data
          # OPTIMIZATION: Only sort faces every few frames (less aggressive)
        if frame_count % 5 == 0 or not hasattr(model, '_sorted_faces'):
            face_depths = []
            for i, face in enumerate(model.faces):
                if len(face) >= 3:
                    # Fast average Z calculation
                    valid_indices = face[face < len(model.vertices)]
                    if len(valid_indices) > 0:
                        avg_depth = np.mean(model.vertices[valid_indices, 2])
                        face_depths.append((avg_depth, i))
            
            face_depths.sort(key=lambda x: x[0], reverse=True)
            model._sorted_faces = face_depths
        
        # Draw faces with optimizations
        light_direction_norm = light_direction / np.linalg.norm(light_direction)
        
        for depth, face_idx in model._sorted_faces:
            face = model.faces[face_idx]
            
            if len(face) >= 3:
                # Fast bounds check
                face_points = []
                valid_face = True
                
                for vertex_idx in face:
                    if vertex_idx < len(projected_vertices):
                        point = projected_vertices[vertex_idx]
                        if -500 <= point[0] <= frame.shape[1] + 500 and -500 <= point[1] <= frame.shape[0] + 500:
                            face_points.append(point)
                        else:
                            valid_face = False
                            break
                    else:
                        valid_face = False
                        break
                
                if valid_face and len(face_points) >= 3:
                    face_points = np.array(face_points, dtype=np.int32)
                    
                    # Use pre-computed normals for lighting
                    if model.face_normals is not None and face_idx < len(model.face_normals):
                        face_normal = model.face_normals[face_idx]
                    else:
                        face_normal = model.calculate_face_normal_fast(face)
                    
                    lighting_intensity = calculate_lighting_intensity_fast(face_normal, light_direction_norm)
                    
                    # Texture sampling
                    face_tex_indices = face_texture_indices[face_idx] if face_idx < len(face_texture_indices) else None
                    
                    if (face_tex_indices is not None and 
                        len(face_tex_indices) >= 3 and 
                        all(idx >= 0 and idx < len(texture_coords) for idx in face_tex_indices) and
                        model.texture_image is not None):
                        
                        # Fast UV sampling (center only)
                        uv_coords = texture_coords[face_tex_indices[:3]]
                        avg_u = np.mean(uv_coords[:, 0])
                        avg_v = np.mean(uv_coords[:, 1])
                        
                        base_color = sample_texture_fast(model.texture_image, avg_u, avg_v)
                        texture_color = tuple(int(c * lighting_intensity) for c in base_color)
                        texture_color = tuple(max(10, min(255, c)) for c in texture_color)
                        
                        cv2.fillPoly(result_frame, [face_points], texture_color)
                        
                        # Simplified wireframe
                        edge_color = tuple(max(5, int(c * 0.7)) for c in texture_color)
                        cv2.polylines(result_frame, [face_points], True, edge_color, 1)
                    else:
                        # Default color with lighting
                        base_color = (60, 120, 80)
                        lit_color = tuple(int(c * lighting_intensity) for c in base_color)
                        lit_color = tuple(max(10, min(255, c)) for c in lit_color)
                        
                        cv2.fillPoly(result_frame, [face_points], lit_color)
                        cv2.polylines(result_frame, [face_points], True, tuple(max(5, int(c * 0.7)) for c in lit_color), 1)
        
        return result_frame
        
    except Exception as e:
        print(f"Error drawing optimized model: {e}")
        return frame

def draw_model_simple(frame, model, projected_vertices, light_direction):
    """SIMPLE model drawing - always shows the model with basic optimizations"""
    if projected_vertices is None or model.faces is None:
        return frame
    
    try:
        result_frame = frame.copy()
        texture_coords, face_texture_indices = model.texture_data
        
        # Simple depth sorting (only when needed)
        face_depths = []
        for i, face in enumerate(model.faces):
            if len(face) >= 3:
                # Fast average Z calculation
                valid_indices = face[face < len(model.vertices)]
                if len(valid_indices) > 0:
                    avg_depth = np.mean(model.vertices[valid_indices, 2])
                    face_depths.append((avg_depth, i))
        
        face_depths.sort(key=lambda x: x[0], reverse=True)
        
        # Draw faces with basic lighting
        light_direction_norm = light_direction / np.linalg.norm(light_direction)
        
        for depth, face_idx in face_depths:
            face = model.faces[face_idx]
            
            if len(face) >= 3:
                # Get face points
                face_points = []
                valid_face = True
                
                for vertex_idx in face:
                    if vertex_idx < len(projected_vertices):
                        point = projected_vertices[vertex_idx]
                        # More lenient bounds check
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
                    
                    # Use pre-computed normals for lighting
                    if model.face_normals is not None and face_idx < len(model.face_normals):
                        face_normal = model.face_normals[face_idx]
                    else:
                        face_normal = model.calculate_face_normal_fast(face)
                    
                    lighting_intensity = calculate_lighting_intensity_fast(face_normal, light_direction_norm)
                    
                    # Texture sampling with lighting
                    face_tex_indices = face_texture_indices[face_idx] if face_idx < len(face_texture_indices) else None
                    
                    if (face_tex_indices is not None and 
                        len(face_tex_indices) >= 3 and 
                        all(idx >= 0 and idx < len(texture_coords) for idx in face_tex_indices) and
                        model.texture_image is not None):
                        
                        # UV sampling
                        uv_coords = texture_coords[face_tex_indices[:3]]
                        avg_u = np.mean(uv_coords[:, 0])
                        avg_v = np.mean(uv_coords[:, 1])
                        
                        base_color = sample_texture_fast(model.texture_image, avg_u, avg_v)
                        texture_color = tuple(int(c * lighting_intensity) for c in base_color)
                        texture_color = tuple(max(10, min(255, c)) for c in texture_color)
                        
                        cv2.fillPoly(result_frame, [face_points], texture_color)
                        
                        # Wireframe
                        edge_color = tuple(max(5, int(c * 0.8)) for c in texture_color)
                        cv2.polylines(result_frame, [face_points], True, edge_color, 1)
                    else:
                        # Default color with lighting
                        base_color = (60, 120, 80)
                        lit_color = tuple(int(c * lighting_intensity) for c in base_color)
                        lit_color = tuple(max(10, min(255, c)) for c in lit_color)
                        
                        cv2.fillPoly(result_frame, [face_points], lit_color)
                        cv2.polylines(result_frame, [face_points], True, tuple(max(5, int(c * 0.8)) for c in lit_color), 1)
        
        return result_frame
        
    except Exception as e:
        print(f"Error drawing simple model: {e}")
        return frame

def ar_main_textured_multi():
    """BALANCED AR with support for multiple 3D models - prioritizes model visibility!"""
    print("🎯 BALANCED AR - MODEL ALWAYS VISIBLE! 🎯")
    print("Hold up ArUco markers in front of the camera")
    print("Optimized to SHOW the model, not hide it!")
    print("Press 'q' to quit")
    
    # Initialize models
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define multiple models with different positions and scales
    models = []
    
    # Model 1: Main breadboard
    breadboard_path = os.path.join(script_dir, "..", "assets", "models", "breadboard.obj")
    texture_path = os.path.join(script_dir, "..", "assets", "images", "uv.jpeg")
    breadboard_path = os.path.normpath(breadboard_path)
    texture_path = os.path.normpath(texture_path)
    
    model1 = Model3D(breadboard_path, texture_path, scale=1.2, position_offset=(0, 0, 0))
    if model1.vertices is not None:
        models.append(model1)
        print("✅ Breadboard model loaded")
    
    # Model 2: Second breadboard (if you want multiple)
    # model2 = Model3D(breadboard_path, texture_path, scale=0.8, position_offset=(0.05, 0, 0.01))
    # if model2.vertices is not None:
    #     models.append(model2)
    #     print("✅ Second model loaded")
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return
    
    # Camera setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Camera calibration
    camera_matrix = np.array([[800.0, 0.0, frame_width/2],
                             [0.0, 800.0, frame_height/2],
                             [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Marker setup
    marker_size = 0.0635
    marker_3d_points = np.array([
        [0.0, 0.0, 0.0],
        [marker_size, 0.0, 0.0],
        [marker_size, marker_size, 0.0],
        [0.0, marker_size, 0.0]
    ], dtype=np.float32)
    marker_3d_points -= np.array([marker_size/2, marker_size/2, 0])
    
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Performance tracking
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    
    # OPTIMIZED light direction
    light_direction = np.array([1.0, -1.5, -1.0])
    light_direction = light_direction / np.linalg.norm(light_direction)
    
    try:
        while True:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    
                    success, rvec, tvec = cv2.solvePnP(
                        marker_3d_points, corner[0], camera_matrix, dist_coeffs
                    )
                    
                    if success:                        # Draw all models for this marker
                        for model in models:
                            projected_vertices = project_vertices(
                                model.vertices, rvec, tvec, camera_matrix, dist_coeffs
                            )
                            
                            # Use simple drawing for single model, optimized for multiple
                            if len(models) == 1:
                                frame = draw_model_simple(
                                    frame, model, projected_vertices, light_direction
                                )
                            else:
                                frame = draw_model_optimized(
                                    frame, model, projected_vertices, light_direction, frame_count
                                )
            
            # Performance info
            frame_time = time.time() - frame_start
            fps_counter += 1
            
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                elapsed = time.time() - fps_start_time
                fps = fps_counter / elapsed
                print(f"🚀 Performance: {fps:.1f} FPS, Frame time: {frame_time*1000:.1f}ms")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Performance indicator on screen
            perf_color = (0, 255, 0) if frame_time < 0.03 else (0, 255, 255) if frame_time < 0.05 else (0, 0, 255)
            cv2.putText(frame, f"Frame: {frame_time*1000:.1f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, perf_color, 2)
              # Instruction text
            instruction_overlay = create_clean_text_overlay(
                frame_width, frame_height, 
                "🎯 BALANCED AR - Model Always Visible! Press 'q' to quit",
                (frame_width // 2, frame_height - 60)
            )
            frame = blend_overlay_with_frame(frame, instruction_overlay)
            
            cv2.imshow('🎯 BALANCED AR - Always Shows Model!', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Optimized AR closed")

# Maintain compatibility with original function name
def ar_main_textured():
    ar_main_textured_multi()

if __name__ == "__main__":
    ar_main_textured()
