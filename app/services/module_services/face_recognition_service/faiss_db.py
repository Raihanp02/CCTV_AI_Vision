import faiss
import numpy as np
import threading

class FAISSDB:
    def __init__(self, index_path: str):
        self.lock = threading.RLock() 
        self.index_path = index_path
        try:
            self.index = faiss.read_index(index_path)
        except:
            self.create_index()

    def create_index(self, dimension: int = 512, index_type: str = 'Flat'):
        if index_type == 'Flat':
            self.index = faiss.IndexFlatIP(dimension)
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
        self.save_index()

    def search(self, query_embeddings, top_k=5):

        query_np = np.array(query_embeddings).astype('float32')
        faiss.normalize_L2(query_np)
        distances, indices = self.index.search(query_np, top_k)
        return distances, indices

    def save_index(self):
        with self.lock:
            faiss.write_index(self.index, self.index_path)