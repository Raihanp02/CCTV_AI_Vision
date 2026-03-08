from collections import defaultdict

def merge_for_detection(by_camera):
    frames = []
    meta = []  # mapping back to camera & frame_id

    for cam_id, data in by_camera.items():
        services = data["services"]
        for frame, frame_id in zip(data["frame"], data["frame_id"]):
            frames.append(frame)
            meta.append((cam_id, frame_id, frame, services))

    return frames, meta

def split_detection_results_columnar(detections, meta, detection_type: str):
    results = defaultdict(lambda: {
        "frame_id": [],
        "frame": [],
        "services": None,
        "detections": defaultdict(lambda: defaultdict(list))
    })

    keys = list(detections.keys())
    for det, (cam_id, frame_id, frame, services) in zip(zip(*detections.values()), meta):
        bucket = results[cam_id]

        if bucket["services"] == None:
            bucket["services"] = services
        bucket["frame_id"].append(frame_id)
        bucket["frame"].append(frame)
        for k, key in enumerate(keys):
            bucket["detections"][detection_type][key].append(det[k])
    
    return results