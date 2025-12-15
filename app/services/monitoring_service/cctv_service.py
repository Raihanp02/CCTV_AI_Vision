import asyncio
import logging
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from services.module_services.face_extract_service import FaceExtractkerService, _check_face_alignment
from services.module_services.facial_expression_service import FacialExpressionService
from services.tracker_service.tracked_data_service import TrackedDataService
import numpy as np
from core.config import settings

camera_monitoring_service = settings.CAMERA_MONITORING_SOURCES
camera_width = settings.CAMERA_MONITORING_FRAME_WIDTH
camera_height = settings.CAMERA_MONITORING_FRAME_HEIGHT

class CameraMonitoringService:
    def __init__(self, camera_url=0):
        self.camera_url = camera_url
        self.cap = None
        self.module = {
            "face_extract_service": FaceExtractkerService(),
            "facial_expression_service": FacialExpressionService()
        }

    async def start(self):
        self.cap = cv2.VideoCapture(self.camera_url)
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera stream: {self.camera_url}")
            return

    async def stop(self):
        try:
            self._async_event.clear()
            self._thread_pool.shutdown(wait=False)
        except Exception:
            logging.exception("Failed to stop camera monitoring")

    async def _capture_frames(self, camera: dict):
        success, frame = self.cap.read()

        if success:
            return frame

camera_monitoring_service = CameraMonitoringService()