"""
Camera utility functions for automatic camera detection and selection
"""
import cv2

def detect_best_camera():
    """Detect and return the best available camera"""
    print("Detecting available cameras...")
    
    # List to store working cameras
    working_cameras = []
    
    # Test cameras from 0 to 4 (usually enough for most systems)
    for camera_index in range(5):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Try to read a frame to confirm camera is working
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                camera_info = {
                    'index': camera_index,
                    'width': width,
                    'height': height,
                    'fps': fps
                }
                working_cameras.append(camera_info)
                print(f"Camera {camera_index}: {width}x{height} @ {fps}fps")
            cap.release()
    
    if not working_cameras:
        print("No working cameras found!")
        return None
    
    # Priority logic: prefer external cameras (usually higher indices) over built-in (usually index 0)
    # Also prefer cameras with higher resolution
    best_camera = None
    
    # First, look for external cameras (typically index 1 or higher)
    external_cameras = [cam for cam in working_cameras if cam['index'] > 0]
    if external_cameras:
        # Among external cameras, pick the one with highest resolution
        best_camera = max(external_cameras, key=lambda x: x['width'] * x['height'])
        print(f"Selected external camera {best_camera['index']} (Priority: External camera)")
    else:
        # Fall back to built-in camera (usually index 0)
        best_camera = working_cameras[0]
        print(f"Selected built-in camera {best_camera['index']} (No external camera found)")
    
    print(f"Using Camera {best_camera['index']}: {best_camera['width']}x{best_camera['height']} @ {best_camera['fps']}fps")
    return best_camera['index']

def get_camera_with_fallback():
    """Get the best camera with fallback to default"""
    camera_index = detect_best_camera()
    if camera_index is None:
        print("Warning: No cameras detected, trying default camera 0")
        camera_index = 0
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    return cap
