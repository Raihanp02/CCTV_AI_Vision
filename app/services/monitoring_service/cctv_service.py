import logging
import time
import cv2
from core.config import settings

import threading
from queue import Queue, Empty

import cv2
import logging

class CCTVService:
    def __init__(self, camera_url, camera_id, buffer = Queue):
        self.buffer = buffer

        self.camera_url = camera_url
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.thread = None

        self.frame_id = 0

    def start(self):
        if self.running:
            return

        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            logging.error(f"Failed to open camera: {self.camera_url}")
            return

        self.cap = cap
        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.cap:
            self.cap.release()
            self.cap = None

        logging.info(f"CCTV stopped: {self.camera_url}")

    def read(self):
        """
        Non-blocking read.
        Returns (camera_id, camera_url, frame_id, frame) or None.
        """
        try:
            return list(CCTVService.global_buffer.queue)
        except Empty:
            return None

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logging.warning("Frame read failed")
                break

            self.frame_id += 1
            CCTVService.global_buffer.put({
                "camera_id":self.camera_id, 
                "camera_url": self.camera_url, 
                "frame_id": self.frame_id, 
                "frame":frame
            })

        self.running = False

