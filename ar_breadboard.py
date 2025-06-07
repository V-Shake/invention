import cv2
import numpy as np
import pygame
import os
import trimesh
import math
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class ARBreadboard:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.model = None
        self.model_loaded = False
        
    def setup_camera_calibration(self):
        """Setup camera calibration parameters"""
        # Simple camera calibration (you might want to calibrate your specific camera)
        # These are approximate values - for better results, calibrate your camera
        self.camera_matrix = np.array([[800.0, 0.0, 320.0],
                                     [0.0, 800.0, 240.0],
                                     [0.0, 0.0, 1.0]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
    
    def load_3d_model(self, model_path):
        """Load a 3D model from GLB file"""
        try:
            if os.path.exists(model_path):
                print(f"Loading 3D model: {model_path}")
                self.model = trimesh.load(model_path)
                self.model_loaded = True
                print(f"Model loaded successfully!")
                
                # Scale down the model if it's too large
                if hasattr(self.model, 'vertices'):
                    bounds = self.model.bounds
                    size = np.max(bounds[1] - bounds[0])
                    if size > 2.0:  # If model is larger than 2 units
                        scale = 2.0 / size
                        self.model.apply_scale(scale)
                        print(f"Model scaled by factor: {scale}")
                elif hasattr(self.model, 'geometry'):
                    # Handle scene with multiple geometries
                    for name, geometry in self.model.geometry.items():
                        bounds = geometry.bounds
                        size = np.max(bounds[1] - bounds[0])
                        if size > 2.0:
                            scale = 2.0 / size
                            geometry.apply_scale(scale)
                            print(f"Geometry {name} scaled by factor: {scale}")
                
                return True
            else:
                print(f"Model file not found: {model_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def draw_breadboard(self):
        """Draw a 3D breadboard model"""
        if self.model_loaded and self.model is not None:
            self.draw_3d_model()
        else:
            self.draw_fallback_breadboard()
    
    def draw_3d_model(self):
        """Draw the loaded 3D model"""
        try:
            # Handle different model types (Scene vs Mesh)
            if hasattr(self.model, 'geometry') and len(self.model.geometry) > 0:
                # It's a Scene with multiple geometries
                for name, geometry in self.model.geometry.items():
                    self.render_mesh(geometry)
            elif hasattr(self.model, 'vertices') and hasattr(self.model, 'faces'):
                # It's a single Mesh
                self.render_mesh(self.model)
            else:
                print("Unknown model format, using fallback")
                self.draw_fallback_breadboard()
        except Exception as e:
            print(f"Error rendering model: {e}")
            self.draw_fallback_breadboard()
    
    def render_mesh(self, mesh):
        """Render a single mesh"""
        if not hasattr(mesh, 'vertices') or not hasattr(mesh, 'faces'):
            return
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Set material properties for breadboard
        glColor3f(0.0, 0.7, 0.0)  # Green color for breadboard
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Calculate normals if available
        if hasattr(mesh, 'vertex_normals'):
            normals = mesh.vertex_normals
        else:
            normals = None
        
        # Render triangles
        glBegin(GL_TRIANGLES)
        
        for face in faces:
            for vertex_idx in face:
                if normals is not None and vertex_idx < len(normals):
                    glNormal3fv(normals[vertex_idx])
                
                if vertex_idx < len(vertices):
                    glVertex3fv(vertices[vertex_idx])
        
        glEnd()
    
    def draw_fallback_breadboard(self):
        """Draw a simple 3D breadboard as fallback"""
        # Breadboard dimensions (scaled for your 2.5 inch marker)
        width, height, depth = 1.5, 0.8, 0.08
        
        # Main breadboard body (green)
        glColor3f(0.2, 0.8, 0.2)
        
        # Draw the main rectangular body
        glBegin(GL_QUADS)
        
        # Top face
        glNormal3f(0, 0, 1)
        glVertex3f(-width/2, height/2, depth/2)
        glVertex3f(width/2, height/2, depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        glVertex3f(-width/2, -height/2, depth/2)
        
        # Bottom face
        glNormal3f(0, 0, -1)
        glVertex3f(-width/2, height/2, -depth/2)
        glVertex3f(-width/2, -height/2, -depth/2)
        glVertex3f(width/2, -height/2, -depth/2)
        glVertex3f(width/2, height/2, -depth/2)
        
        # Front face
        glNormal3f(0, -1, 0)
        glVertex3f(-width/2, -height/2, -depth/2)
        glVertex3f(-width/2, -height/2, depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        glVertex3f(width/2, -height/2, -depth/2)
        
        # Back face
        glNormal3f(0, 1, 0)
        glVertex3f(-width/2, height/2, -depth/2)
        glVertex3f(width/2, height/2, -depth/2)
        glVertex3f(width/2, height/2, depth/2)
        glVertex3f(-width/2, height/2, depth/2)
        
        # Left face
        glNormal3f(-1, 0, 0)
        glVertex3f(-width/2, height/2, -depth/2)
        glVertex3f(-width/2, height/2, depth/2)
        glVertex3f(-width/2, -height/2, depth/2)
        glVertex3f(-width/2, -height/2, -depth/2)
        
        # Right face
        glNormal3f(1, 0, 0)
        glVertex3f(width/2, height/2, -depth/2)
        glVertex3f(width/2, -height/2, -depth/2)
        glVertex3f(width/2, -height/2, depth/2)
        glVertex3f(width/2, height/2, depth/2)
        
        glEnd()
        
        # Draw holes (dark green circles)
        glColor3f(0.1, 0.3, 0.1)
        rows, cols = 6, 10
        hole_spacing_x = width * 0.8 / cols
        hole_spacing_y = height * 0.6 / rows
        
        for i in range(rows):
            for j in range(cols):
                x = -width * 0.4 + j * hole_spacing_x
                y = -height * 0.3 + i * hole_spacing_y
                
                glPushMatrix()
                glTranslatef(x, y, depth/2 + 0.005)
                
                # Draw small cylinder for holes
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, 0, 0)
                for angle in range(0, 361, 30):
                    rad = math.radians(angle)
                    glVertex3f(0.02 * math.cos(rad), 0.02 * math.sin(rad), 0)
                glEnd()
                
                glPopMatrix()

    def draw_text_3d(self, text, x=0, y=0, z=0):
        """Draw 3D text using simple line segments"""
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 0.0)  # Yellow text
        glLineWidth(3.0)
        
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(0.08, 0.08, 0.08)
        
        # Simple letter rendering
        char_width = 1.2
        x_offset = 0
        
        for char in text.upper():
            glPushMatrix()
            glTranslatef(x_offset, 0, 0)
            
            glBegin(GL_LINE_STRIP)
            
            if char == 'B':
                # Letter B
                glVertex3f(0, 0, 0)
                glVertex3f(0, 2, 0)
                glVertex3f(1.2, 2, 0)
                glVertex3f(1.2, 1, 0)
                glVertex3f(0, 1, 0)
                glEnd()
                glBegin(GL_LINE_STRIP)
                glVertex3f(0, 1, 0)
                glVertex3f(1.2, 1, 0)
                glVertex3f(1.2, 0, 0)
                glVertex3f(0, 0, 0)
            elif char == 'R':
                # Letter R
                glVertex3f(0, 0, 0)
                glVertex3f(0, 2, 0)
                glVertex3f(1.2, 2, 0)
                glVertex3f(1.2, 1, 0)
                glVertex3f(0, 1, 0)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(0, 1, 0)
                glVertex3f(1.2, 0, 0)
            elif char == 'E':
                # Letter E
                glVertex3f(1.2, 0, 0)
                glVertex3f(0, 0, 0)
                glVertex3f(0, 2, 0)
                glVertex3f(1.2, 2, 0)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(0, 1, 0)
                glVertex3f(0.8, 1, 0)
            elif char == 'A':
                # Letter A
                glVertex3f(0, 0, 0)
                glVertex3f(0.6, 2, 0)
                glVertex3f(1.2, 0, 0)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(0.3, 1, 0)
                glVertex3f(0.9, 1, 0)
            elif char == 'D':
                # Letter D
                glVertex3f(0, 0, 0)
                glVertex3f(0, 2, 0)
                glVertex3f(0.8, 2, 0)
                glVertex3f(1.2, 1.5, 0)
                glVertex3f(1.2, 0.5, 0)
                glVertex3f(0.8, 0, 0)
                glVertex3f(0, 0, 0)
            elif char == 'O':
                # Letter O (circle)
                glEnd()
                glBegin(GL_LINE_LOOP)
                for angle in range(0, 361, 30):
                    rad = math.radians(angle)
                    glVertex3f(0.6 + 0.4 * math.cos(rad), 1 + 0.8 * math.sin(rad), 0)
                glEnd()
                glBegin(GL_LINE_STRIP)
            elif char == '1':
                # Number 1
                glVertex3f(0.6, 0, 0)
                glVertex3f(0.6, 2, 0)
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(0.3, 1.7, 0)
                glVertex3f(0.6, 2, 0)
                glVertex3f(0.2, 0, 0)
                glVertex3f(1.0, 0, 0)
                glEnd()
                glBegin(GL_LINE_STRIP)
            elif char == 'X':
                # Letter X
                glEnd()
                glBegin(GL_LINES)
                glVertex3f(0, 0, 0)
                glVertex3f(1.2, 2, 0)
                glVertex3f(0, 2, 0)
                glVertex3f(1.2, 0, 0)
                glEnd()
                glBegin(GL_LINE_STRIP)
            elif char == ' ':
                # Space - do nothing
                pass
            
            glEnd()
            glBegin(GL_LINE_STRIP)
            
            x_offset += char_width
            glPopMatrix()
        
        glEnd()
        glPopMatrix()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)

def main():
    print("ArUco Marker Augmented Reality - 3D Breadboard Overlay")
    print("Hold up ArUco markers (ID 0-10) in front of the camera")
    print("Press 'q' or ESC to quit")
    
    # Initialize OpenCV camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get camera frame size
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}")
    
    # Initialize Pygame and OpenGL
    pygame.init()
    display = (frame_width, frame_height)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AR Breadboard - Press Q to Quit")
    
    # Setup OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, [2, 2, 2, 0])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
    
    # Setup projection matrix
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    
    # ArUco setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # AR Breadboard instance
    ar_breadboard = ARBreadboard()
    ar_breadboard.setup_camera_calibration()
    
    # Load the 3D model
    model_path = "arduino_breadboard_-_low_poly.glb"
    if not ar_breadboard.load_3d_model(model_path):
        print("Using fallback breadboard model")
    
    # Marker size in meters (adjusted for 2.5 inch marker ≈ 0.0635 meters)
    marker_size = 0.0635
    
    clock = pygame.time.Clock()
    
    try:
        while True:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == QUIT:
                    raise KeyboardInterrupt
                elif event.type == KEYDOWN:
                    if event.key == K_q or event.key == K_ESCAPE:
                        raise KeyboardInterrupt
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect markers
            corners, ids, _ = detector.detectMarkers(gray)
            
            # Clear OpenGL buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Convert OpenCV frame to OpenGL texture
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = np.flipud(frame_rgb)
            
            # Draw background (camera feed)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, frame_width, 0, frame_height, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glRasterPos2f(0, 0)
            glDrawPixels(frame_width, frame_height, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
            # If markers detected, overlay 3D model
            if ids is not None:
                print(f"Detected {len(ids)} marker(s): {ids.flatten()}")
                
                for i, corner in enumerate(corners):
                    # Estimate pose using solvePnP
                    object_points = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)
                    
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, 
                        corner[0], 
                        ar_breadboard.camera_matrix, 
                        ar_breadboard.dist_coeffs
                    )
                    
                    if success:
                        # Apply 3D transformation
                        glPushMatrix()
                        
                        # Convert pose to OpenGL coordinates
                        x, y, z = tvec.flatten()
                        
                        # Scale and position adjustments for proper visibility
                        scale_factor = 15.0  # Adjust this to make model larger/smaller
                        x = x * scale_factor
                        y = -y * scale_factor  # Flip Y axis
                        z = -z * scale_factor - 8.0  # Move closer to camera
                        
                        glTranslatef(x, y, z)
                        
                        # Apply rotation from pose estimation
                        angle = np.linalg.norm(rvec) * 180.0 / np.pi
                        if angle > 0:
                            axis = rvec.flatten() / np.linalg.norm(rvec)
                            glRotatef(angle, axis[0], -axis[1], -axis[2])
                        
                        # Scale the model to appropriate size
                        model_scale = 0.8
                        glScalef(model_scale, model_scale, model_scale)
                        
                        # Lift the model slightly above the marker
                        glTranslatef(0, 0, 0.15)
                        
                        # Draw the 3D breadboard
                        ar_breadboard.draw_breadboard()
                        
                        # Draw text label above the breadboard
                        glPushMatrix()
                        glTranslatef(0, 0, 1.2)
                        ar_breadboard.draw_text_3d("BREADBOARD 1X", 0, 0, 0)
                        glPopMatrix()
                        
                        glPopMatrix()
            
            # Update display
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cap.release()
        pygame.quit()
        print("Application closed")

if __name__ == "__main__":
    main()