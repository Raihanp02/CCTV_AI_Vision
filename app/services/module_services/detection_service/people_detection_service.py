import onnxruntime as ort
import numpy as np
import logging
import cv2
from sort.tracker import SortTracker
from app.services.module_services.detection_service.base_detection import BaseDetection


logger = logging.getLogger(__name__)


class PeopleDetectionService(BaseDetection):
    def __init__(
        self,
        model_path: str = "assets/models/yolo11n.onnx",
        max_boxes: int = 30
    ):
        self.max_boxes = max_boxes
        
        # Initialize with CUDA → CPU fallback
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.model = ort.InferenceSession(model_path, providers=providers)
        
        # Log which provider was actually selected
        actual_provider = self.model.get_providers()[0]
        if "CUDA" in actual_provider:
            logger.info(f"People Counting: Using CUDA acceleration")
        else:
            logger.info(f"People Counting: Using CPU (CUDA not available)")
        self.boxes = np.empty((max_boxes, 6), dtype=np.float32)
        self.frame_count = 0

    def _filter_detections(self,results: np.ndarray, thresh: float = 0.25):
        if results.size == 0:
            return np.array([[]])
        if results.shape[1] == 5:
            mask = results[:, 4] > thresh
            return results[mask] if np.any(mask) else np.array([[]])
        # multi-class output
        class_ids = results[:, 4:].argmax(axis=1)
        confidences = results[:, 4:].max(axis=1)
        keep_mask = confidences > thresh
        if not np.any(keep_mask):
            return np.array([[]])
        filtered = np.column_stack((results[keep_mask, :4], class_ids[keep_mask], confidences[keep_mask]))
        return filtered

    def _NMS(self, boxes: np.ndarray, conf_scores: np.ndarray, iou_thresh: float = 0.55):
        # boxes: Nx4 (x1,y1,x2,y2)
        if len(boxes) == 0:
            return [], []
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = conf_scores.argsort()[::-1]
        keep_idx = []
        while order.size > 0:
            i = order[0]
            keep_idx.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)
            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]
        keep = boxes[keep_idx]
        keep_conf = conf_scores[keep_idx]
        return keep, keep_conf

    def _rescale_back(self, results, img_w: int, img_h: int):
        # results columns: cx, cy, w, h, class_id, confidence
        cx = results[:, 0] / 640.0 * img_w
        cy = results[:, 1] / 640.0 * img_h
        w = results[:, 2] / 640.0 * img_w
        h = results[:, 3] / 640.0 * img_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.column_stack((x1, y1, x2, y2, results[:, 4]))
        keep, keep_conf = self._NMS(boxes[:, :4], results[:, -1])
        return keep, keep_conf

    def detect(self, frame):
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1.0 / 255.0,
            size=(640, 640),
            swapRB=True,
            crop=False,
        )

        results = self.model.run(None, {"images": blob})

        output = results[0][0].T
        boxes = self._filter_detections(output)

        if boxes.size == 0:
            return np.empty((0, 6))

        rescaled_results, confidences = self._rescale_back(boxes, w, h)
        num_boxes = min(len(confidences), self.max_boxes)
        valid_count = 0

        for i in range(num_boxes):
            startX = rescaled_results[i][0]
            startY = rescaled_results[i][1]
            endX = rescaled_results[i][2]
            endY = rescaled_results[i][3]

            self.boxes[valid_count, 0] = startX
            self.boxes[valid_count, 1] = startY
            self.boxes[valid_count, 2] = endX
            self.boxes[valid_count, 3] = endY
            self.boxes[valid_count, 4] = confidences[i]
            self.boxes[valid_count, 5] = 0.0
            valid_count += 1

        return self.boxes[:valid_count]