"""
Abstract Tracker Backend Interface
Provides a common interface for different tracking algorithms (SORT, ByteTrack, DeepSORT, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import numpy as np


class TrackerBackend(ABC):
    """
    Abstract base class for object tracking backends.
    All trackers must implement this interface for drop-in compatibility.
    """

    @abstractmethod
    def update(
        self, 
        detections: np.ndarray, 
        embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Update tracker with new detections.

        Args:
            detections: Array of shape (N, 6) with format [x1, y1, x2, y2, confidence, class_id]
                - x1, y1: Top-left corner coordinates
                - x2, y2: Bottom-right corner coordinates
                - confidence: Detection confidence score (0.0 to 1.0)
                - class_id: Object class identifier

            embeddings: Optional array of shape (N, D) with appearance embeddings
                - Used by DeepSORT and other appearance-based trackers
                - Can be None for motion-only trackers

        Returns:
            Array of shape (M, 7) with format [x1, y1, x2, y2, track_id, class_id, confidence]
                - x1, y1, x2, y2: Bounding box coordinates
                - track_id: Unique stable track identifier
                - class_id: Object class identifier
                - confidence: Tracking confidence score

        Notes:
            - M <= N (some detections may not generate tracks)
            - track_id is persistent across frames for the same object
            - Returned tracks are only confirmed/active tracks
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset tracker state.
        
        This clears all active tracks and resets internal counters.
        Useful when:
        - Switching between video sources
        - Starting a new tracking session
        - Recovering from errors
        """
        pass

    def get_track_by_id(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Get track object by ID (optional, for advanced usage).

        Args:
            track_id: Track identifier

        Returns:
            Dict with track information including cached analytics, or None if not found
        """
        return None

    def get_active_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all active track objects (optional, for advanced usage).

        Returns:
            List of track information dicts with cached analytics
        """
        return []


class TrackState:
    """
    Track state enumeration for advanced trackers.
    
    States:
        New: Track just created, not yet confirmed
        Tracked: Actively tracked with recent detections
        Lost: Temporarily lost, still being predicted
        Removed: Permanently removed from tracker
    """
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3
    Tentative = 4
