import numpy as np
import onnxruntime as ort
from typing import Tuple
import cv2
import logging
from app.services.module_services.detection_service.base_detection import BaseDetection

logger = logging.getLogger(__name__)

class FaceDetectionService(BaseDetection):
    def __init__(self, tracker_service,max_boxes: int = 20):
        self.tracked_data_service = tracker_service
        self.face_detection = RetinaFaceDecoder(model_path="assets/models/det_10g.onnx")
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']

    def detect(self, frames: list[np.ndarray], min_area: float = 0.03) -> dict[str, list[np.ndarray]]:
        filtered_boxes = []
        filtered_lmks = []
        filtered_scores = []

        for frame in frames:
            boxes, landmarks, scores = self.face_detection.detect(
                frame, 
                conf_threshold=0.5,
                check_alignment=False,
                max_roll_angle=15.0,      # Maximum head tilt in degrees
                max_yaw_ratio=0.15,       # Maximum left-right rotation (0-1)
                max_pitch_ratio=0.2       # Maximum up-down rotation (0-1)
            )

            valid_indices = []

            for idx, (box, lmks, score) in enumerate(zip(boxes, landmarks, scores)):

                x1, y1, x2, y2 = box.astype(int)
                area = (x2 - x1) * (y2 - y1)

                if area / (frame.shape[0] * frame.shape[1]) > min_area:
                    valid_indices.append(idx)

            # Now slice ONLY ONCE
            filtered_boxes.append(boxes[valid_indices])
            filtered_lmks.append(landmarks[valid_indices])
            filtered_scores.append(scores[valid_indices])

        return {"boxes": filtered_boxes, "landmarks": filtered_lmks, "scores": filtered_scores}
    
    def face_check(self, face_embed):
        with self.shared_lock:
            items = list(self.shared_data.items())
        
        for i, data in items:
            if self.shared_data[i].get("embedding") is None:
                continue
            similarity = self._cosine_similarity(face_embed, self.shared_data[i]["embedding"])
            if similarity > 0.8:
                return i
        return None
    
    def get_face_embedding(self, frame):
        blob = cv2.dnn.blobFromImage(
            image=frame,
            scalefactor=1.0/255,
            size = (160,160),  # normalize 0-1
            mean=(131.0912/255, 103.8827/255, 91.4953/255),# target size
            swapRB=True,           # RGB <-> BGR, not needed for grayscale
            crop=False
        )
        results = self.face_embed_model.run(None, {"input":blob})

        return results[0][0]
    
    def _cosine_similarity(self,a,b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RetinaFaceDecoder:
    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize RetinaFace decoder
        
        Args:
            model_path: Path to det_10g.onnx
            input_size: Model input size (width, height)
        """
        self.session = ort.InferenceSession(model_path)
        self.input_size = input_size
        self.input_name = self.session.get_inputs()[0].name
        
        # Feature map strides
        self.strides = [8, 16, 32]
        
        # Generate anchors for each scale
        self.anchors = self._generate_anchors()
        
        # Debug: Print anchor counts
        
    def _generate_anchors(self):
        """Generate anchor points for all feature pyramid levels"""
        # Expected anchor counts from model architecture
        expected_counts = [12800, 3200, 800]
        anchors_list = []
        
        for stride, expected_count in zip(self.strides, expected_counts):
            h = self.input_size[1] // stride
            w = self.input_size[0] // stride
            
            # Calculate how many anchors per grid position
            grid_size = h * w
            num_anchors = expected_count // grid_size
                        
            # Create grid of anchor centers
            shift_x = np.arange(0, w) * stride
            shift_y = np.arange(0, h) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            
            # Flatten
            shift_x = shift_x.ravel()
            shift_y = shift_y.ravel()
            
            # Repeat each position for multiple anchors
            shift_x = np.repeat(shift_x, num_anchors)
            shift_y = np.repeat(shift_y, num_anchors)
            
            # Stack to create anchor centers
            anchors = np.stack([shift_x, shift_y], axis=1).astype(np.float32)
            
            # Add stride offset to center the anchor
            anchors += stride // 2
            
            anchors_list.append(anchors)
            
        return anchors_list
    
    def _decode_boxes(self, box_preds: np.ndarray, anchors: np.ndarray, stride: int):
        """
        Decode bounding boxes from predictions
        
        Args:
            box_preds: [N, 4] predicted box deltas
            anchors: [N, 2] anchor centers
            stride: Feature stride
        """
        # RetinaFace uses distance format: [left, top, right, bottom] distances from center
        # The predictions are already scaled, so we use them directly
        boxes = np.zeros_like(box_preds)
        
        # Get anchor center without the offset
        anchor_center_x = anchors[:, 0] - stride // 2
        anchor_center_y = anchors[:, 1] - stride // 2
        
        # Decode to [x1, y1, x2, y2] format
        boxes[:, 0] = anchor_center_x - box_preds[:, 0] * stride  # x1 = cx - left
        boxes[:, 1] = anchor_center_y - box_preds[:, 1] * stride  # y1 = cy - top
        boxes[:, 2] = anchor_center_x + box_preds[:, 2] * stride  # x2 = cx + right
        boxes[:, 3] = anchor_center_y + box_preds[:, 3] * stride  # y2 = cy + bottom
        
        return boxes
    
    def _decode_landmarks(self, landmark_preds: np.ndarray, anchors: np.ndarray, stride: int):
        """
        Decode 5 facial landmarks from predictions
        
        Args:
            landmark_preds: [N, 10] predicted landmark deltas (5 points × 2 coords)
            anchors: [N, 2] anchor centers
            stride: Feature stride
        
        Returns:
            landmarks: [N, 5, 2] decoded landmark coordinates
        """
        N = landmark_preds.shape[0]
        landmarks = np.zeros((N, 5, 2), dtype=np.float32)
        
        # Get anchor center without the offset
        anchor_center_x = anchors[:, 0] - stride // 2
        anchor_center_y = anchors[:, 1] - stride // 2
        
        for i in range(5):
            # Decode x coordinate for landmark i
            landmarks[:, i, 0] = anchor_center_x + landmark_preds[:, i*2] * stride
            # Decode y coordinate for landmark i
            landmarks[:, i, 1] = anchor_center_y + landmark_preds[:, i*2 + 1] * stride
        
        return landmarks
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4):
        """Non-maximum suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5, 
               nms_threshold: float = 0.4,
               check_alignment: bool = True,
               max_roll_angle: float = 15.0,
               max_yaw_ratio: float = 0.15,
               max_pitch_ratio: float = 0.2):
        """
        Detect faces and landmarks with alignment validation
        
        Args:
            frame: Input frame (H, W, 3) BGR format
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            check_alignment: Whether to filter by face alignment
            max_roll_angle: Maximum roll angle in degrees for alignment check
            max_yaw_ratio: Maximum yaw deviation ratio for alignment check
            max_pitch_ratio: Maximum pitch deviation ratio for alignment check
            
        Returns:
            boxes: [N, 4] bounding boxes (x1, y1, x2, y2)
            landmarks: [N, 5, 2] facial landmarks
            scores: [N] confidence scores
        """
        # Preprocess
        img_input = self._preprocess(frame)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_input})
        
        # Parse outputs - order based on your model
        # Scores: outputs[0], outputs[1], outputs[2]
        # Boxes: outputs[3], outputs[4], outputs[5]
        # Landmarks: outputs[6], outputs[7], outputs[8]
        
        all_boxes = []
        all_landmarks = []
        all_scores = []
        
        for i, stride in enumerate(self.strides):
            scores = outputs[i].squeeze()  # [N, 1] -> [N]
            boxes = outputs[i + 3]  # [N, 4]
            landmarks = outputs[i + 6]  # [N, 10]
            anchors = self.anchors[i]
            
            # Filter by confidence
            mask = scores > conf_threshold
            if not mask.any():
                continue
            
            scores = scores[mask]
            boxes = boxes[mask]
            landmarks = landmarks[mask]
            anchors = anchors[mask]
            
            # Decode boxes and landmarks
            decoded_boxes = self._decode_boxes(boxes, anchors, stride)
            decoded_landmarks = self._decode_landmarks(landmarks, anchors, stride)
            
            all_boxes.append(decoded_boxes)
            all_landmarks.append(decoded_landmarks)
            all_scores.append(scores)
        
        if not all_boxes:
            return np.array([]), np.array([]), np.array([])
        
        # Concatenate all scales
        boxes = np.vstack(all_boxes)
        landmarks = np.vstack(all_landmarks)
        scores = np.concatenate(all_scores)
        
        # Apply NMS
        keep = self._nms(boxes, scores, nms_threshold)
        
        # Scale back to original frame size
        scale_x = frame.shape[1] / self.input_size[0]
        scale_y = frame.shape[0] / self.input_size[1]
        
        boxes = boxes[keep]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        landmarks = landmarks[keep]
        landmarks[:, :, 0] *= scale_x
        landmarks[:, :, 1] *= scale_y
        
        scores = scores[keep]
        
        # Filter by alignment if enabled
        if check_alignment:
            aligned_indices = []
            for idx in range(len(boxes)):
                if _check_face_alignment(landmarks[idx], boxes[idx],
                                             max_roll_angle, max_yaw_ratio, max_pitch_ratio):
                    aligned_indices.append(idx)
            
            if len(aligned_indices) == 0:
                return np.array([]), np.array([]), np.array([])
            
            boxes = boxes[aligned_indices]
            landmarks = landmarks[aligned_indices]
            scores = scores[aligned_indices]
        
        return boxes, landmarks, scores
    
    def _preprocess(self, frame: np.ndarray):
        """Preprocess frame for model input"""
        # Resize
        img = cv2.resize(frame, self.input_size)
        
        # Normalize (adjust based on your model's training)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0  # Common normalization
        
        # HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, 0)
        
        return img
    
def _check_face_alignment(landmarks: np.ndarray, box: np.ndarray, 
                            max_roll_angle: float = 15.0,
                            max_yaw_ratio: float = 0.15,
                            max_pitch_ratio: float = 0.2) -> bool:
        """
        Check if face is properly aligned (frontal)
        
        Args:
            landmarks: [5, 2] facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
            box: [4] bounding box (x1, y1, x2, y2)
            max_roll_angle: Maximum roll angle in degrees
            max_yaw_ratio: Maximum yaw deviation ratio (0-1)
            max_pitch_ratio: Maximum pitch deviation ratio (0-1)
            
        Returns:
            True if face is properly aligned
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate face center
        face_center_x = (box[0] + box[2]) / 2
        face_width = box[2] - box[0]
        face_height = box[3] - box[1]
        
        # 1. Check Roll (head tilt) - eyes should be roughly horizontal
        eye_delta_x = right_eye[0] - left_eye[0]
        eye_delta_y = right_eye[1] - left_eye[1]
        
        if eye_delta_x == 0:
            return False
        
        roll_angle = np.degrees(np.arctan(eye_delta_y / eye_delta_x))
        if abs(roll_angle) > max_roll_angle:
            return False
        
        # 2. Check Yaw (left-right rotation) - nose should be centered
        nose_center_offset = abs(nose[0] - face_center_x)
        yaw_ratio = nose_center_offset / face_width
        
        if yaw_ratio > max_yaw_ratio:
            return False
        
        # 3. Check Pitch (up-down rotation) - vertical alignment of features
        # Calculate eye center
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Calculate mouth center
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        
        # Expected nose position (between eyes and mouth)
        expected_nose_y = (eye_center_y + mouth_center_y) / 2
        nose_offset = abs(nose[1] - expected_nose_y)
        pitch_ratio = nose_offset / face_height
        
        if pitch_ratio > max_pitch_ratio:
            return False
        
        # 4. Check eye symmetry - both eyes should be roughly same distance from nose
        left_eye_nose_dist = np.linalg.norm(left_eye - nose)
        right_eye_nose_dist = np.linalg.norm(right_eye - nose)
        
        if left_eye_nose_dist == 0 or right_eye_nose_dist == 0:
            return False
        
        eye_symmetry_ratio = min(left_eye_nose_dist, right_eye_nose_dist) / max(left_eye_nose_dist, right_eye_nose_dist)
        
        if eye_symmetry_ratio < 0.7:  # Eyes should be relatively symmetric
            return False
        
        # 5. Check mouth symmetry - mouth corners should be roughly equidistant from nose
        left_mouth_nose_dist = np.linalg.norm(left_mouth - nose)
        right_mouth_nose_dist = np.linalg.norm(right_mouth - nose)
        
        if left_mouth_nose_dist == 0 or right_mouth_nose_dist == 0:
            return False
        
        mouth_symmetry_ratio = min(left_mouth_nose_dist, right_mouth_nose_dist) / max(left_mouth_nose_dist, right_mouth_nose_dist)
        
        if mouth_symmetry_ratio < 0.7:  # Mouth should be relatively symmetric
            return False
        
        return True