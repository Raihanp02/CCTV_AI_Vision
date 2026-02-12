"""
DeepSORT Tracker with Face Embedding Support
Combines motion-based tracking with appearance-based re-identification.

DeepSORT improves tracking by:
1. Using face embeddings for appearance matching
2. Kalman filter for motion prediction
3. Hungarian algorithm for optimal association
4. Track management with confirmed/tentative states
5. Better handling of occlusions and re-identification
"""

import numpy as np
import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from .tracker_backend import TrackerBackend, TrackState

logger = logging.getLogger(__name__)


@dataclass
class TrackCache:
    """Cache for analytics results per track."""
    # Face Recognition cache
    face_encoding: Optional[np.ndarray] = None
    identity: Optional[str] = None
    identity_confidence: float = 0.0
    
    # Gender Detection cache
    gender: Optional[str] = None
    gender_confidence: float = 0.0
    
    # Facial Expression (NOT cached - runs every frame)
    expression: Optional[str] = None
    expression_confidence: float = 0.0
    
    # People Counting cache
    counted: bool = False
    
    # Cache metadata
    last_updated: float = 0.0
    needs_reidentify: bool = True
    
    def clear(self):
        """Clear all cached data."""
        self.face_encoding = None
        self.identity = None
        self.identity_confidence = 0.0
        self.gender = None
        self.gender_confidence = 0.0
        self.expression = None
        self.expression_confidence = 0.0
        self.counted = False
        self.last_updated = 0.0
        self.needs_reidentify = True


class KalmanFilter:
    """
    Simple Kalman filter for bbox tracking.
    State: [x, y, a, h, vx, vy, va, vh] where (x,y) is center, a is aspect ratio, h is height.
    """
    
    def __init__(self):
        self.dim_x = 8  # State dimension
        self.dim_z = 4  # Measurement dimension
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        self.F[0, 4] = 1  # x += vx
        self.F[1, 5] = 1  # y += vy
        self.F[2, 6] = 1  # a += va
        self.F[3, 7] = 1  # h += vh
        
        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float32)
        
        # Process noise covariance
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise covariance
        self.R = np.eye(4, dtype=np.float32) * 10.0
        
        # State covariance
        self.P = np.eye(8, dtype=np.float32) * 1000.0
        
        # State vector
        self.x = np.zeros((8, 1), dtype=np.float32)
        
    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()
    
    def update(self, z: np.ndarray):
        """Update state with measurement."""
        z = z.reshape(-1, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


def bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to [cx, cy, aspect_ratio, height].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2
    cy = bbox[1] + h / 2
    a = w / (h + 1e-6)
    return np.array([cx, cy, a, h], dtype=np.float32)


def z_to_bbox(z: np.ndarray) -> np.ndarray:
    """
    Convert [cx, cy, aspect_ratio, height] to [x1, y1, x2, y2].
    """
    cx, cy, a, h = z[:4]
    w = a * h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bboxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)


def cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine distance between embeddings."""
    if emb1 is None or emb2 is None:
        return 1.0
    
    # Normalize embeddings
    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-6)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-6)
    
    # Cosine similarity
    similarity = np.dot(emb1, emb2)
    
    # Convert to distance
    distance = 1.0 - similarity
    return distance


class Track:
    """Single track with Kalman filter, appearance features, and analytics cache."""
    
    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
        class_id: int = 0
    ):
        """
        Initialize track.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            embedding: Face embedding for appearance matching
            class_id: Object class
        """
        self.track_id = str(uuid.uuid4())
        
        self.confidence = confidence
        self.class_id = class_id
        
        # Kalman filter for motion
        self.kf = KalmanFilter()
        z = bbox_to_z(bbox)
        self.kf.x[:4] = z.reshape(-1, 1)
        
        # Appearance features
        self.embeddings: List[np.ndarray] = []
        if embedding is not None:
            self.embeddings.append(embedding)
        
        # Track state
        self.state = TrackState.Tentative
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        # Thresholds
        self.n_init = 3  # Frames needed to confirm track
        self.max_age = 30  # Max frames without update before deletion
        
        # Analytics cache
        self.cache = TrackCache()
        
    def predict(self):
        """Predict next position."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        
    def update(self, bbox: np.ndarray, confidence: float, embedding: Optional[np.ndarray] = None):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            confidence: Detection confidence
            embedding: Face embedding
        """
        z = bbox_to_z(bbox)
        self.kf.update(z)
        
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # Update appearance features
        if embedding is not None:
            self.embeddings.append(embedding)
            # Keep only recent embeddings (e.g., last 100)
            if len(self.embeddings) > 100:
                self.embeddings = self.embeddings[-100:]
        
        # Confirm track after n_init hits
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Tracked
    
    def mark_missed(self):
        """Mark as missed detection."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Removed
        elif self.time_since_update > self.max_age:
            self.state = TrackState.Removed
            self.cache.clear()
    
    def is_confirmed(self) -> bool:
        """Check if track is confirmed."""
        return self.state == TrackState.Tracked
    
    def is_tentative(self) -> bool:
        """Check if track is tentative."""
        return self.state == TrackState.Tentative
    
    def is_deleted(self) -> bool:
        """Check if track should be deleted."""
        return self.state == TrackState.Removed
    
    @property
    def bbox(self) -> np.ndarray:
        """Get current bounding box."""
        return z_to_bbox(self.kf.x[:4, 0])
    
    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        """Get mean of all embeddings."""
        if not self.embeddings:
            return None
        return np.mean(self.embeddings, axis=0)


class DeepSORTTracker(TrackerBackend):
    """
    DeepSORT tracker combining motion and appearance.
    
    Uses Kalman filter for motion prediction and face embeddings
    for appearance-based re-identification.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_iou_distance: float = 0.7,
        max_cosine_distance: float = 0.3,
        nn_budget: Optional[int] = 100,
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Max frames to keep track without update
            n_init: Number of consecutive detections to confirm track
            max_iou_distance: Max IoU distance for matching
            max_cosine_distance: Max cosine distance for appearance matching
            nn_budget: Max embeddings to store per track
            use_cuda: Force CUDA usage (None=auto-detect)
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        
        # Track lists
        self.tracks: List[Track] = []
        self.frame_count = 0
        
        logger.info(
            f"DeepSORTTracker initialized: max_age={max_age}, n_init={n_init}, "
            f"max_iou_distance={max_iou_distance}, max_cosine_distance={max_cosine_distance}"
        )
    
    def update(
        self,
        detections: np.ndarray,
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with detections and embeddings.
        
        Args:
            detections: Array (N, 6) with [x1, y1, x2, y2, confidence, class_id]
            embeddings: Array (N, D) with face embeddings (optional but recommended)
        
        Returns:
            Array of dicts with keys: 'bbox' (ndarray), 'track_id' (str UUID), 
            'class_id' (int), 'confidence' (float)
        """
        self.frame_count += 1
        
        # Predict all tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to tracks
        matches, unmatched_detections, unmatched_tracks = self._match(
            detections, embeddings
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            bbox = detections[det_idx, :4]
            confidence = detections[det_idx, 4]
            embedding = embeddings[det_idx] if embeddings is not None else None
            self.tracks[track_idx].update(bbox, confidence, embedding)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox = detections[det_idx, :4]
            confidence = detections[det_idx, 4]
            class_id = int(detections[det_idx, 5]) if detections.shape[1] > 5 else 0
            embedding = embeddings[det_idx] if embeddings is not None else None
            
            track = Track(bbox, confidence, embedding, class_id)
            track.n_init = self.n_init
            track.max_age = self.max_age
            self.tracks.append(track)
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Get confirmed tracks for output
        # Return structured array to accommodate UUID strings
        outputs = []
        for track in self.tracks:
            if track.is_confirmed():
                bbox = track.bbox
                outputs.append({
                    'bbox': bbox,
                    'track_id': track.track_id,
                    'class_id': track.class_id,
                    'confidence': track.confidence
                })
        
        return np.array(outputs, dtype=object) if outputs else np.array([], dtype=object)
    
    def _match(
        self,
        detections: np.ndarray,
        embeddings: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using motion and appearance.
        
        Returns:
            matches: List of (track_idx, detection_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Get confirmed and tentative tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_tracks = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        
        # First: Match confirmed tracks with cascade matching
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._matching_cascade(confirmed_tracks, detections, embeddings)
        
        # Second: Match remaining tentative tracks with IOU
        iou_track_candidates = tentative_tracks + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._iou_matching(
                iou_track_candidates, detections, unmatched_detections, embeddings
            )
        
        matches = matches_a + matches_b
        unmatched_tracks = [
            k for k in unmatched_tracks_a if k not in [m[0] for m in matches_b]
        ] + [k for k in unmatched_tracks_b]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _matching_cascade(
        self,
        track_indices: List[int],
        detections: np.ndarray,
        embeddings: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Cascade matching by age."""
        matches = []
        unmatched_detections = list(range(len(detections)))
        
        # Match by age (prefer recent tracks)
        for age in range(self.max_age):
            if len(unmatched_detections) == 0:
                break
            
            track_indices_age = [
                k for k in track_indices
                if self.tracks[k].time_since_update == 1 + age
            ]
            
            if len(track_indices_age) == 0:
                continue
            
            matches_age, _, unmatched_detections = self._gate_cost_matching(
                track_indices_age, detections, unmatched_detections, embeddings
            )
            
            matches += matches_age
        
        unmatched_tracks = [k for k in track_indices if k not in [m[0] for m in matches]]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _gate_cost_matching(
        self,
        track_indices: List[int],
        detections: np.ndarray,
        detection_indices: List[int],
        embeddings: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Cost-based matching with gating."""
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices
        
        # Build cost matrix
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track = self.tracks[track_idx]
            track_bbox = track.bbox
            track_emb = track.mean_embedding
            
            for j, det_idx in enumerate(detection_indices):
                det_bbox = detections[det_idx, :4]
                det_emb = embeddings[det_idx] if embeddings is not None else None
                
                # Compute IoU distance
                iou_dist = 1.0 - iou(track_bbox, det_bbox)
                
                # Compute appearance distance
                if track_emb is not None and det_emb is not None:
                    app_dist = cosine_distance(track_emb, det_emb)
                    # Weighted combination
                    cost = 0.3 * iou_dist + 0.7 * app_dist
                else:
                    cost = iou_dist
                
                cost_matrix[i, j] = cost
        
        # Gate by max distance
        cost_matrix[cost_matrix > self.max_iou_distance] = self.max_iou_distance + 1e-5
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(track_indices)))
        unmatched_detections = list(range(len(detection_indices)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_iou_distance:
                track_idx = track_indices[row]
                det_idx = detection_indices[col]
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        unmatched_tracks = [track_indices[i] for i in unmatched_tracks]
        unmatched_detections = [detection_indices[i] for i in unmatched_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _iou_matching(
        self,
        track_indices: List[int],
        detections: np.ndarray,
        detection_indices: List[int],
        embeddings: Optional[np.ndarray]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """IoU-based matching for tentative tracks."""
        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices
        
        # Build IoU cost matrix
        cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track_bbox = self.tracks[track_idx].bbox
            for j, det_idx in enumerate(detection_indices):
                det_bbox = detections[det_idx, :4]
                cost_matrix[i, j] = 1.0 - iou(track_bbox, det_bbox)
        
        # Gate and match
        cost_matrix[cost_matrix > self.max_iou_distance] = self.max_iou_distance + 1e-5
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(track_indices)))
        unmatched_detections = list(range(len(detection_indices)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_iou_distance:
                track_idx = track_indices[row]
                det_idx = detection_indices[col]
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        unmatched_tracks = [track_indices[i] for i in unmatched_tracks]
        unmatched_detections = [detection_indices[i] for i in unmatched_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1
        logger.debug("DeepSORT tracker reset")
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get track by ID with cached analytics."""
        for track in self.tracks:
            if track.track_id == track_id and track.is_confirmed():
                return {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "confidence": track.confidence,
                    "state": track.state,
                    "cache": track.cache
                }
        return None
    
    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """Get all active tracks with cached analytics."""
        return [
            {
                "track_id": track.track_id,
                "bbox": track.bbox,
                "confidence": track.confidence,
                "state": track.state,
                "cache": track.cache
            }
            for track in self.tracks
            if track.is_confirmed()
        ]
