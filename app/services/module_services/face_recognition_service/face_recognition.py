
class FaceRecognitionService:
    def __init__(self, face_embed_model, db, employee_search):
        self.face_embedding_service = face_embed_model
        self.db = db
        self.employee_search = employee_search

    def recognize_faces(self, face):
        embedding = self.face_embedding_service.get_face_embedding(face)
        distances, ids = self.db.search(embedding)
        
        employee = self.employee_search.get_employee(ids.item())

        return employee, distances.item()
    
    def add_employee_face(self, face, employee_id):
        embedding = self.face_embedding_service.get_face_embedding(face)
        self.db.add_embeddings(embedding, employee_id)