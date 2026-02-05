import cv2
import onnxruntime as ort

class FaceEmbeddingService:
    def __init__(self, embed_model_path):
        self.face_embed_model = ort.InferenceSession(embed_model_path)

    def get_face_embedding(self, frame):
        blob = self.preprocess_face(frame)
        results = self.face_embed_model.run(None, {"input":blob})

        return results[0][0]
    
    def preprocess_face(self, face):
        blob = cv2.dnn.blobFromImage(
            image=face,
            scalefactor=1.0/255,
            size = (160,160),  # normalize 0-1
            mean=(131.0912/255, 103.8827/255, 91.4953/255),# target size
            swapRB=True,           # RGB <-> BGR, not needed for grayscale
            crop=False
        )
        return blob