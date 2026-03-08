import faiss
import numpy as np
import threading
from pathlib import Path

class FAISSDB:
    BASE_DIR = Path(__file__).parent.resolve().parents[3]

    def __init__(self, 
                 index_path: str = str(BASE_DIR / "app" / "services" / "module_services" / "face_recognition_service" / "faiss_index" / "index.faiss")):
        self.lock = threading.RLock() 
        self.index_path = index_path

        try:
            self.index = faiss.read_index(index_path)
        except:
            self.create_index()

    def create_index(self, dimension: int = 512, index_type: str = 'Flat'):
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        if index_type == 'Flat':
            base_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap(base_index)
        elif index_type == 'IVF':
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_L2)
            self.index.train(np.empty((0, dimension), dtype='float32'))
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def add_embeddings(self, embeddings, ids: int):
        
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings_np = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_np)
        self.index.add_with_ids(embeddings_np, np.array([ids]).astype('int64'))

    def search(self, query_embeddings, top_k=5):

        if isinstance(query_embeddings, list):
            query_embeddings = np.array(query_embeddings)

        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)

        query_np = np.array(query_embeddings).astype('float32')
        faiss.normalize_L2(query_np)
        distances, indices = self.index.search(query_np, top_k)
        return distances, indices

    def save_index(self):
        with self.lock:
            faiss.write_index(self.index, self.index_path)