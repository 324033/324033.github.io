import os
import sys
import time
import cv2
import numpy as np
from app import app, video_jobs

def create_synthetic_test_video():
    temp_dir = os.path.join(os.path.dirname(__file__), 'temp')
    test_video_path = os.path.join(temp_dir, 'synthetic_test.mp4')
    os.makedirs(temp_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (320, 240))
    for i in range(15):
        frame = np.ones((240, 320, 3), dtype=np.uint8) * 128
        # Moving box and face-like oval
        cv2.rectangle(frame, (20 + i*5, 30), (80 + i*5, 90), (0, 0, 255), -1)
        cv2.ellipse(frame, (160, 120), (35, 50), 0, 0, 360, (200, 220, 240), -1)
        out.write(frame)
    out.release()
    return test_video_path

def test_video_pipeline():
    print("Testing Video file upload pipeline...")
    client = app.test_client()

    video_path = create_synthetic_test_video()

    # 1. Test Object Detection on uploaded video file
    with open(video_path, 'rb') as f:
        data = {
            'source_type': 'upload',
            'video_task': 'object',
            'video': (f, 'test_upload.mp4'),
            'duration': '5',
            'sample_fps': '10',
            'threshold': '0.1'
        }
        res = client.post('/api/video/start', data=data, content_type='multipart/form-data')

    assert res.status_code == 200, f"Failed to start object video job: {res.data}"
    job_id = res.get_json()['job_id']
    print(f"Object Detection video job started: {job_id}")

    # Poll status
    start_t = time.time()
    while time.time() - start_t < 25:
        s = client.get(f'/api/video/status/{job_id}').get_json()
        if s['status'] == 'completed':
            print("Object video detection completed:", s['video_url'])
            break
        elif s['status'] == 'failed':
            raise RuntimeError(f"Job failed: {s.get('error')}")
        time.sleep(1)

    # 2. Test 3D Face Mesh on uploaded video file with black background
    with open(video_path, 'rb') as f:
        data_face = {
            'source_type': 'upload',
            'video_task': 'facemesh',
            'video': (f, 'test_face_upload.mp4'),
            'duration': '5',
            'sample_fps': '10',
            'background': 'black',
            'mesh_style': 'both',
            'color_theme': 'cyan'
        }
        res_face = client.post('/api/video/start', data=data_face, content_type='multipart/form-data')

    assert res_face.status_code == 200, f"Failed to start face mesh video job: {res_face.data}"
    face_job_id = res_face.get_json()['job_id']
    print(f"Face Mesh video job started: {face_job_id}")

    start_t = time.time()
    while time.time() - start_t < 25:
        s = client.get(f'/api/video/status/{face_job_id}').get_json()
        if s['status'] == 'completed':
            print("Face Mesh video detection completed:", s['video_url'])
            break
        elif s['status'] == 'failed':
            raise RuntimeError(f"Face mesh video job failed: {s.get('error')}")
        time.sleep(1)

    # Clean up synthetic input video
    if os.path.exists(video_path):
        os.remove(video_path)

    print("ALL VIDEO FILE TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_video_pipeline()
