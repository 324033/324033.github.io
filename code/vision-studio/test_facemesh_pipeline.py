import os
import io
import time
import cv2
import numpy as np
from app import app

def create_synthetic_face_image():
    img = np.ones((600, 600, 3), dtype=np.uint8) * 40
    cx, cy = 300, 300
    # Head
    cv2.ellipse(img, (cx, cy), (120, 160), 0, 0, 360, (180, 210, 240), -1)
    # Eyes
    cv2.ellipse(img, (cx - 50, cy - 30), (25, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (cx + 50, cy - 30), (25, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(img, (cx - 50, cy - 30), 8, (120, 60, 20), -1)
    cv2.circle(img, (cx + 50, cy - 30), 8, (120, 60, 20), -1)
    # Eyebrows
    cv2.line(img, (cx - 75, cy - 55), (cx - 25, cy - 50), (40, 30, 20), 4)
    cv2.line(img, (cx + 25, cy - 50), (cx + 75, cy - 55), (40, 30, 20), 4)
    # Nose
    cv2.line(img, (cx, cy - 20), (cx, cy + 25), (140, 160, 190), 3)
    cv2.line(img, (cx, cy + 25), (cx + 12, cy + 25), (140, 160, 190), 3)
    # Lips
    cv2.ellipse(img, (cx, cy + 70), (40, 15), 0, 0, 360, (80, 50, 200), -1)
    
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_facemesh_api():
    print("Testing Face Mesh API endpoint...")
    client = app.test_client()

    img_bytes = create_synthetic_face_image()

    # 1. Test Original Background with Dual-layer Mesh
    res = client.post('/api/facemesh/detect', data={
        'image': (io.BytesIO(img_bytes), 'face.jpg'),
        'background': 'original',
        'mesh_style': 'both',
        'color_theme': 'cyan',
        'line_thickness': '1',
        'draw_points': 'false'
    }, content_type='multipart/form-data')

    assert res.status_code == 200, f"Face mesh endpoint failed: {res.data}"
    data = res.get_json()
    assert data['success'] is True
    print(f"Face Mesh (Original BG): Inference: {data['inference_time_ms']}ms, Faces detected: {data['faces_detected']}")

    # 2. Test Solid Black Background with Cyberpunk Magenta Contours
    res_black = client.post('/api/facemesh/detect', data={
        'image': (io.BytesIO(img_bytes), 'face.jpg'),
        'background': 'black',
        'mesh_style': 'tesselation',
        'color_theme': 'magenta',
        'line_thickness': '2',
        'draw_points': 'true'
    }, content_type='multipart/form-data')

    assert res_black.status_code == 200, f"Face mesh black background failed: {res_black.data}"
    data_black = res_black.get_json()
    assert data_black['success'] is True
    assert data_black['annotated_image'].startswith('data:image/jpeg;base64,')
    print("Face Mesh (Solid Black BG + Tesselation): Passed successfully!")
    print("ALL FACE MESH TESTS PASSED!")

if __name__ == '__main__':
    test_facemesh_api()
