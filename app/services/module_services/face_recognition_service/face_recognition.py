from .face_embedding_service import FaceEmbeddingInception, FaceEmbeddingArcFace
from .crud_service import PGCrud
from core.db import SessionDep

import numpy as np 
import cv2
from pathlib import Path
import uuid as uuidlib

class FaceRecognitionService:
    BASE_DIR = Path(__file__).resolve().parents[5]
    def __init__(self, face_embedding_service = FaceEmbeddingArcFace(), 
                 embedding_db = PGCrud(), 
                 employee_db = PGCrud()):
        self.face_embedding_service = face_embedding_service
        self.embedding_db = embedding_db
        self.employee_db = employee_db

    def recognize_faces(self, session: SessionDep, face:np.ndarray, image: np.ndarray, bbox: list, uuid = None):

        employee_id = None
        if uuid:
            employee = self.employee_db.get_employee_by_uuid(session, uuid)
            employee_id = employee.id

        embedding = self.face_embedding_service.get_face_embedding(face)
        result = self.embedding_db.get_employee_from_embedding(session, embedding, employee_id = employee_id)

        return result
    
    def add_employee_face(self, session: SessionDep, face: np.ndarray, image: np.ndarray, bbox: list, uuid, filename: str, created_by_id: int = None, save_image = True):
        employee_id = None
        if uuid:
            employee = self.employee_db.get_employee_by_uuid(session, uuid)
            employee_id = employee.id
        
        x1, y1, x2, y2 = bbox
        embedding = self.face_embedding_service.get_face_embedding(face)

        # Save the face image to disk
        w, h = image.shape[1], image.shape[0]
        img_absolute_path = self.BASE_DIR / "media" / "employee_face_images" / str(employee_id) / filename
        img_relative_path = img_absolute_path.relative_to(self.BASE_DIR)
        
        if save_image:
            img_absolute_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_image_to_disk(image, (w, h), bbox, scale=1.5, path=img_absolute_path)

        # Save the embedding and image path to the database
        result = self.embedding_db.add_embedding(session, embedding, img_relative_path.as_posix(), employee_id, created_by_id=created_by_id)
        return result
    
    def save_image_to_disk(self, image: np.ndarray, image_shape: tuple, bbox: list, scale: float, path):
        x1, y1, x2, y2 = bbox
        w, h = image_shape[0], image_shape[1]
        new_bbox = self._expand_bbox(bbox, scale=scale, img_shape=(w, h))
        x1, y1, x2, y2 = new_bbox
        
        cv2.imwrite(str(path), image[y1:y2, x1:x2])
    
    def _expand_bbox(self, bbox, scale, img_shape=None):
        x1, y1, x2, y2 = bbox

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w / 2
        cy = y1 + h / 2

        new_w = w * scale
        new_h = h * scale

        new_x1 = int(cx - new_w / 2)
        new_y1 = int(cy - new_h / 2)
        new_x2 = int(cx + new_w / 2)
        new_y2 = int(cy + new_h / 2)

        if img_shape is not None:
            img_w, img_h = img_shape
            new_x1 = max(0, new_x1)
            new_y1 = max(0, new_y1)
            new_x2 = min(img_w, new_x2)
            new_y2 = min(img_h, new_y2)

        return new_x1, new_y1, new_x2, new_y2