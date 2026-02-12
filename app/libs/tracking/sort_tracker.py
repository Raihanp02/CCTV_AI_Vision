"""Lightweight SORT tracker wrapper.

This module implements the core ideas from the Simple Online and Realtime Tracking
paper (Bewley et al.) and mirrors the public `sort-track` package so the
application can keep face/body ID continuity without pulling in that package's
strict SciPy pin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


ArrayLike = np.ndarray


def _linear_assignment(cost_matrix: ArrayLike) -> ArrayLike:
    """Return best matching pairs using the Hungarian algorithm."""

    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    rows, cols = linear_sum_assignment(cost_matrix)
    return np.asarray(list(zip(rows, cols)), dtype=int)


def _iou_batch(candidates: ArrayLike, targets: ArrayLike) -> ArrayLike:
    """Compute pairwise IoU for two [N, 4] and [M, 4] bbox tensors."""

    if len(candidates) == 0 or len(targets) == 0:
        return np.zeros((len(candidates), len(targets)), dtype=float)

    candidates = np.expand_dims(candidates, axis=1)
    targets = np.expand_dims(targets, axis=0)

    inter_x1 = np.maximum(candidates[..., 0], targets[..., 0])
    inter_y1 = np.maximum(candidates[..., 1], targets[..., 1])
    inter_x2 = np.minimum(candidates[..., 2], targets[..., 2])
    inter_y2 = np.minimum(candidates[..., 3], targets[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter_area = inter_w * inter_h

    cand_area = (candidates[..., 2] - candidates[..., 0]) * (
        candidates[..., 3] - candidates[..., 1]
    )
    tgt_area = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])

    union = cand_area + tgt_area - inter_area
    union = np.clip(union, a_min=1e-6, a_max=None)
    return inter_area / union


def _convert_bbox_to_z(bbox: ArrayLike) -> ArrayLike:
    """Convert [x1, y1, x2, y2] box into [cx, cy, s, r] state vector."""

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    scale = w * h
    ratio = w / max(h, 1e-6)
    return np.array([cx, cy, scale, ratio]).reshape(4, 1)


def _convert_state_to_bbox(
    state: ArrayLike, score: Optional[float] = None
) -> ArrayLike:
    """Convert [cx, cy, s, r] state vector back into corner box."""

    w = np.sqrt(max(state[2], 1e-6) * state[3])
    h = max(state[2], 1e-6) / max(w, 1e-6)
    x1 = state[0] - w / 2.0
    y1 = state[1] - h / 2.0
    x2 = state[0] + w / 2.0
    y2 = state[1] + h / 2.0
    if score is None:
        return np.array([x1, y1, x2, y2]).reshape(1, 4)
    return np.array([x1, y1, x2, y2, score]).reshape(1, 5)


@dataclass
class _TrackMeta:
    conf: float
    cls: int


class _KalmanBox:
    """Simple constant-velocity Kalman tracker for a single bbox."""

    _next_id = 0

    def __init__(self, bbox: ArrayLike, confidence: float, cls: int):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.eye(4, 7)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = _convert_bbox_to_z(bbox)
        self.meta = _TrackMeta(conf=confidence, cls=cls)
        self.id = _KalmanBox._next_id
        _KalmanBox._next_id += 1

        self.time_since_update = 0
        self.hit_streak = 0
        self.hits = 0
        self.age = 0
        self._history: List[ArrayLike] = []

    def predict(self) -> ArrayLike:
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self._history.append(_convert_state_to_bbox(self.kf.x))
        return self._history[-1]

    def update(self, bbox: ArrayLike) -> None:
        self.time_since_update = 0
        self._history.clear()
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(_convert_bbox_to_z(bbox))

    def get_state(self) -> ArrayLike:
        return _convert_state_to_bbox(self.kf.x)


def _associate(
    detections: ArrayLike, trackers: ArrayLike, iou_threshold: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    if len(detections) == 0 and trackers.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype=int),
        )

    if len(detections) == 0:
        unmatched_trackers = (
            np.arange(len(trackers)) if trackers.size else np.empty((0,), dtype=int)
        )
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            unmatched_trackers,
        )

    if trackers.size == 0:
        unmatched_dets = np.arange(len(detections))
        return np.empty((0, 2), dtype=int), unmatched_dets, np.empty((0,), dtype=int)

    iou_matrix = _iou_batch(detections[:, :4], trackers[:, :4])
    if min(iou_matrix.shape) > 0:
        tentative = (iou_matrix > iou_threshold).astype(np.int32)
        if (
            tentative.size
            and tentative.sum(1).max() == 1
            and tentative.sum(0).max() == 1
        ):
            matched_indices = np.stack(np.where(tentative), axis=1)
        else:
            matched_indices = _linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty((0, 2), dtype=int)

    if len(matched_indices):
        unmatched_dets = [
            d for d in range(len(detections)) if d not in matched_indices[:, 0]
        ]
        unmatched_trks = [
            t for t in range(len(trackers)) if t not in matched_indices[:, 1]
        ]
    else:
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

    matches: List[ArrayLike] = []
    for det_idx, trk_idx in matched_indices:
        if iou_matrix[det_idx, trk_idx] < iou_threshold:
            unmatched_dets.append(det_idx)
            unmatched_trks.append(trk_idx)
        else:
            matches.append(np.array([[det_idx, trk_idx]], dtype=int))

    if matches:
        matched = np.concatenate(matches, axis=0)
    else:
        matched = np.empty((0, 2), dtype=int)

    return (
        matched,
        np.array(unmatched_dets, dtype=int),
        np.array(unmatched_trks, dtype=int),
    )


class SortTracker:
    """Drop-in replacement for the `sort-track` package tracker."""

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._trackers: List[_KalmanBox] = []
        self._frame_count = 0

    def update(
        self, detections: ArrayLike, _embeddings: Optional[ArrayLike]=None
    ) -> ArrayLike:
        """Update tracker state with `[x1,y1,x2,y2,score,class_id]` detections."""

        self._frame_count += 1
        predicted = np.zeros((len(self._trackers), 5))
        delete_indices: List[int] = []
        for idx, trk in enumerate(self._trackers):
            state = trk.predict()[0]
            predicted[idx, :4] = state
            if np.isnan(state).any():
                delete_indices.append(idx)
        predicted = np.ma.compress_rows(np.ma.masked_invalid(predicted))
        for idx in reversed(delete_indices):
            self._trackers.pop(idx)

        matches, unmatched_det_idx, unmatched_trk_idx = _associate(
            detections, predicted, self.iou_threshold
        )

        for det_idx, trk_idx in matches:
            self._trackers[trk_idx].update(detections[det_idx, :4])

        for det_idx in unmatched_det_idx:
            det = detections[det_idx]
            self._trackers.append(
                _KalmanBox(
                    det[:4], float(det[4]), int(det[5]) if det.shape[0] > 5 else 0
                )
            )

        outputs: List[ArrayLike] = []
        for trk in list(self._trackers):
            bbox = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self._frame_count <= self.min_hits
            ):
                outputs.append(
                    np.concatenate(
                        (bbox, [trk.id + 1], [trk.meta.cls], [trk.meta.conf])
                    ).reshape(1, -1)
                )
            if trk.time_since_update > self.max_age:
                self._trackers.remove(trk)

        if outputs:
            return np.concatenate(outputs, axis=0)
        return np.empty((0, 7))
