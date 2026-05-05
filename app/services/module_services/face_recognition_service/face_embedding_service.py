import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import requests
import gdown

class FaceEmbeddingInception:
    BASE_DIR = Path(__file__).resolve().parents[5]
    def __init__(self, embed_model_path = str(BASE_DIR / ".models" / "inception_resnet_v1.onnx")):
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
    
class FaceEmbeddingArcFace:
    BASE_DIR = Path(__file__).resolve().parents[5]
    def __init__(self, embed_model_path = str(BASE_DIR / ".models" / "mobilefacenet.onnx")):
        try:
            self.face_embed_model = ort.InferenceSession(embed_model_path)

        except:
            if Path(embed_model_path).name == "w600k_r50.onnx":
                download_from_gdrive("1z6vNMivXh-rciF8ufyeXIZPllEcG5saT", embed_model_path)
                self.face_embed_model = ort.InferenceSession(embed_model_path)

            if Path(embed_model_path).name == "mobilefacenet.onnx":
                download_from_gdrive("1VPtElVGbEQrSgox_HAeXHUbqOsLMXtJ6", embed_model_path)
                self.face_embed_model = ort.InferenceSession(embed_model_path)
            else:
                raise Exception("Embedding model not found")
        
        self.input_name = self.face_embed_model.get_inputs()[0].name

    def get_face_embedding(self, frame):
        blob = self.preprocess_face(frame)
        results = self.face_embed_model.run(None, {self.input_name:blob})

        embedding = results[0][0]

        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)
    
    def preprocess_face(self, face):
        blob = cv2.dnn.blobFromImage(
            image=face,
            scalefactor=1.0 / 127.5,   # scale to [-1,1]
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,                # BGR -> RGB
            crop=False
        )
        return blob.astype("float32")
    
def download_from_gdrive(file_id, output_path):
    """download from google drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)