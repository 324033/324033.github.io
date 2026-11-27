import io
import cv2
import numpy as np
from app import app

def test_pipeline():
    print("Testing Flask app test client...")
    client = app.test_client()

    # 1. Test GET /
    res = client.get('/')
    assert res.status_code == 200, f"GET / returned {res.status_code}"
    print("GET / passed (200 OK)")

    # 2. Create test image: A colored rectangle and circle to simulate objects
    img = np.ones((400, 400, 3), dtype=np.uint8) * 200
    # Add some high contrast shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)
    cv2.circle(img, (300, 250), 60, (255, 0, 0), -1)
    _, buffer = cv2.imencode('.jpg', img)

    # 3. Test POST /api/detect
    data = {
        'image': (io.BytesIO(buffer.tobytes()), 'test.jpg'),
        'threshold': '0.1',
        'max_results': '10'
    }
    res = client.post('/api/detect', data=data, content_type='multipart/form-data')
    assert res.status_code == 200, f"POST /api/detect returned {res.status_code}: {res.data}"
    json_data = res.get_json()
    assert json_data['success'] is True
    print(f"POST /api/detect passed! Image dimensions: {json_data['image_width']}x{json_data['image_height']}, Inference time: {json_data['inference_time_ms']}ms, Objects detected: {json_data['total_detected']}")
    print("All backend tests completed successfully!")

if __name__ == '__main__':
    test_pipeline()
