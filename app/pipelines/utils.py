from collections import defaultdict

def merge_for_detection(by_camera):
    frames = []
    meta = []  # mapping back to camera & frame_id

    for cam_id, data in by_camera.items():
        for frame, frame_id in zip(data["frame"], data["frame_id"]):
            frames.append(frame)
            meta.append((cam_id, frame_id, frame))

    return frames, meta

def split_detection_results_columnar(detections, meta, detection_type: str):
    results = defaultdict(lambda: {
        "frame_id": [],
        "frame": [],
        "detections": {}
    })

    keys = detections.keys()
    temp = []

    for det, (cam_id, frame_id, frame) in zip(zip(detections.values()), meta):
        bucket = results[cam_id]
        bucket["frame_id"].append(frame_id)
        bucket["frame"].append(frame)
        temp.append(det)
        
    bucket["detections"][detection_type] = temp

    return results