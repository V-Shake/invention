"""
Camera utility functions for automatic camera detection and selection
"""
import cv2
import time

def detect_best_camera_fast():
    """Fast camera detection - prioritizes external cameras without extensive testing"""
    print("Quick camera detection...")
    
    # First, try external cameras (index 1-3) which are usually USB cameras
    external_cameras = [1, 2, 3]
    for camera_index in external_cameras:
        print(f"Testing camera {camera_index}...", end=" ")
        cap = cv2.VideoCapture(camera_index)
        
        # Set a very short timeout for faster detection
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # Quick test - just check if we can get basic properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if width > 0 and height > 0:  # Basic sanity check
                cap.release()
                print(f"✓ Found external camera {camera_index}: {width}x{height}")
                return camera_index
            
        cap.release()
        print("✗")
    
    # If no external camera found, use built-in camera (index 0)
    print("Testing built-in camera 0...", end=" ")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        if width > 0 and height > 0:
            print(f"✓ Using built-in camera: {width}x{height}")
            return 0
    
    cap.release()
    print("✗")
    print("No working cameras found!")
    return None

def detect_best_camera():
    """Detect and return the best available camera"""
    print("Detecting available cameras...")
    
    # List to store working cameras
    working_cameras = []
    
    # Test cameras from 0 to 4 (usually enough for most systems)
    for camera_index in range(5):
        print(f"Testing camera {camera_index}...", end=" ")
        cap = cv2.VideoCapture(camera_index)
        
        # Set shorter timeout for faster detection
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # Quick property check instead of frame reading
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            if width > 0 and height > 0:  # Basic sanity check
                camera_info = {
                    'index': camera_index,
                    'width': width,
                    'height': height,
                    'fps': fps
                }
                working_cameras.append(camera_info)
                print(f"✓ {width}x{height} @ {fps}fps")
            else:
                print("✗ Invalid properties")
        else:
            print("✗ Cannot open")
        
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
    """Get the best camera with fallback to default - FAST VERSION"""
    camera_index = detect_best_camera_fast()
    if camera_index is None:
        print("Warning: No cameras detected, trying default camera 0")
        camera_index = 0
    
    print(f"Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    # Optimize camera settings for faster startup
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
    cap.set(cv2.CAP_PROP_FPS, 30)        # Set desired FPS
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    # Warm up the camera with a single frame read
    print("Initializing camera...", end=" ")
    ret, frame = cap.read()
    if ret:
        print("✓ Ready!")
    else:
        print("⚠ Warning: Camera may need more time to initialize")
    
    return cap

def get_camera_super_fast():
    """Super fast camera initialization - tries external camera first, then built-in"""
    print("Super fast camera detection...")
    
    # Try external camera first (most likely your Logitech C922)
    for camera_index in [1, 2]:  # Most USB cameras are on index 1 or 2
        print(f"Trying camera {camera_index}...", end=" ")
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            # Quick validation
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            if width > 0:
                print(f"✓ Using external camera {camera_index}")
                return cap
        
        cap.release()
        print("✗")
    
    # Fallback to built-in camera
    print("Trying built-in camera 0...", end=" ")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap.isOpened():
        print("✓ Using built-in camera")
        return cap
    
    cap.release()
    print("✗ No cameras available")
    return None
