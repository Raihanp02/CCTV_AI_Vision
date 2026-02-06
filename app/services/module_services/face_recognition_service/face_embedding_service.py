import cv2
import onnxruntime as ort
from pathlib import Path

class FaceEmbeddingService:
    BASE_DIR = Path(__file__).parent.resolve().parents[3]
    def __init__(self, embed_model_path = str(BASE_DIR / "assets" / "models" / "inception_resnet_v1.onnx")):
        self.face_embed_model = ort.InferenceSession(embed_model_path)

    def get_face_embedding(self, frame):
        blob = self.preprocess_face(frame)
        results = self.face_embed_model.run(None, {"input":blob})

        return results[0][0]
    
    def preprocess_face(self, face):
        blob = cv2.dnn.blobFromImage(
            image=face,
            scalefactor=1.0/255,
            size = (300,300),  # normalize 0-1
            mean=(131.0912/255, 103.8827/255, 91.4953/255),# target size
            swapRB=True,           # RGB <-> BGR, not needed for grayscale
            crop=False
        )
        return blob