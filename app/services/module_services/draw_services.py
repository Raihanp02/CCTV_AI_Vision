import cv2

class DrawServices:
    def __init__(self, thickness = 1, size=0.75):
        self.thickness = thickness
        self.size = size

    def draw_bbox(self, frame, info_bbox):
        for info in info_bbox:
            x1,y1,x2,y2 = info["bbox"]
            label = info.get("label", "")
            confidence = info.get("confidence", "")
            id = info.get("id", "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), self.thickness)
            cv2.putText(frame, f"{label} ({confidence})", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, self.size, (0, 255, 0), self.thickness)