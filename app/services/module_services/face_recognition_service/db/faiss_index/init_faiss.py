from .faiss_db import FAISSDB
from ...face_embedding_service import FaceEmbeddingService
from ....detection_service.face_detection_service import FaceDetectionService
from pathlib import Path
import cv2

class FaceRetinaExtractor:
    def __init__(self, face_detection_service = FaceDetectionService()):
        self.face_detection_service = face_detection_service

    def get_face(self, frame):
        frame = [frame]

        result = self.face_detection_service.detect(frame)
        try:
            bbox = result['boxes'][0][0] # Assuming one face per image
        except:
            print(f"No face detected in image:")
            return None
        
        face = frame[0][bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face.size == 0:
            print(f"Invalid face crop in image:")
            return None

        return face
    
class UltraLightExtractor:
    from ultralight import UltraLightDetector

    def __init__(self, face_detection_service = UltraLightDetector()):
        self.face_detection_service = face_detection_service

    def get_face(self, frame):
        result_boxes, scores = self.face_detection_service.detect_one(frame)

        if result_boxes is None or len(result_boxes) == 0:
            print(f"No face detected in image:")
            return None
        
        bbox = result_boxes[0]
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        if face.size == 0:
            print(f"Invalid face crop in image:")
            return None
        
        return face

class InitFaiss:
    def __init__(self, 
                 faiss_db = FAISSDB(), 
                 face_embedding = FaceEmbeddingService(), 
                 folder_path = str(FAISSDB.BASE_DIR / "assets" / "media"),
                 face_extractor = FaceRetinaExtractor()):
        self.faiss_db = faiss_db
        self.face_embedding = face_embedding
        self.folder_path = folder_path
        self.face_extractor = face_extractor
        self.image_loader()
        self.faiss_db.create_index()
    
    def image_loader(self):
        self.image = []
        self.label = []

        for file in Path(self.folder_path).rglob("*"):
            if file.is_file():
                self.image.append(str(file))
                self.label.append(str(file.parent.name))

    def init_faiss(self, detect_face = True, force = True):
        for image_path, label in zip(self.image, self.label):
            print("-----------------------------")
            print(f"Processing image: {image_path} for label: {label}")
            frame = cv2.imread(image_path)

            if detect_face:
                face = self.face_extractor.get_face(frame)

                if face is None:
                    if force:
                        face = frame
                        print(f"Using original image for label {label} as no face detected.")
                    else:
                        print(f"Skipping {image_path} as no face detected.")
                        continue

            embedding = self.face_embedding.get_face_embedding(face)
            self.faiss_db.add_embeddings(embedding, label)
            print(f"Added {label} to FAISS index.")

        self.faiss_db.save_index()
        print("FAISS index initialized and saved.")