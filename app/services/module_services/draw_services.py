import cv2

class DrawServices:
    def __init__(self, thickness = 1, size=0.75):
        self.thickness = thickness
        self.size = size

    def process(self, frame_info):
        for key, info in frame_info.items():
            for frame, results in zip(info["frame"], zip(*info["result"].values())):
                for result_per_frame in results:
                    for result in result_per_frame:
                        x1,y1,x2,y2 = result["bbox"]
                        label = ""
                        for result_label in result.get("detections").values():
                            label += f"{result_label.get('label','')}, "

                        self.draw_bbox(frame, {
                            "bbox": [x1, y1, x2, y2],
                            "label": label,
                        })
                
    def draw_bbox(self, frame, info):
        x1,y1,x2,y2 = info["bbox"]
        label = info.get("label", "")
        confidence = info.get("confidence", "")
        id = info.get("id", "")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), self.thickness)
        cv2.putText(frame, f"{label} {confidence}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, self.size, (0, 255, 0), self.thickness)