import asyncio
import logging
import time
import cv2
from concurrent.futures import ThreadPoolExecutor
from app.services.module_services.face_detection_service import FaceDetectionService, _check_face_alignment
from services.module_services.facial_expression_service import FacialExpressionService
import numpy as np
from core.config import settings

import threading

class CCTVService:
    def __init__(self, camera_url):
        self.camera_url = camera_url
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self.latest_frame = None
        self.frame_count = 0

    def start(self):
        with self.lock:
            if self.running:
                return

            self.cap = cv2.VideoCapture(self.camera_url)
            if not self.cap.isOpened():
                logging.error(f"Failed to open camera: {self.camera_url}")
                return

            self.running = True
            self.thread = threading.Thread(
                target=self._capture_loop,
                daemon=True
            )
            self.thread.start()

    def stop(self):
        with self.lock:
            if not self.running:
                return

            self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.latest_frame = None
        logging.info(f"CCTV stopped: {self.camera_url}")

    def read(self):
        """
        Non-blocking read.
        Returns latest frame or None.
        """
        return self.latest_frame

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame read failed")
                break

            self.latest_frame = frame
            self.frame_count += 1

        self.running = False
