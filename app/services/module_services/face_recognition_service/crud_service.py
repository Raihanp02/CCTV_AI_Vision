from core.models.person_face import PersonFace
from core.models.person import Person

import numpy as np
from sqlalchemy.exc import IntegrityError
from sqlalchemy import delete, select
from typing import Type

from aqi_attendance.src.core.database import session

class PGCrud:
    # ---------- CREATE ----------
    def add_embedding(self, session, embedding: list[float] | np.ndarray, image_path: str, person_id: int, created_by_id: int = None) -> PersonFace:
        # person_id: int, image_url: str, embedding: list[float]
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(float).tolist()

            face = PersonFace(
                image_path=image_path,
                embedding=embedding,
                person_id=person_id,
            )

            session.add(face)
            session.commit()
            session.refresh(face)
            return face
        
        except IntegrityError:
            session.rollback()
            raise Exception("Duplicate embedding")
        
        except Exception as e:
            session.rollback()
            raise Exception(f"Error creating face embedding record: {e}")


    # ---------- READ ----------
    def get_person_by_person_id(self, session, person_id: int) -> Person:
        obj = session.execute(select(Person).where(Person.id == person_id)).first()
        if obj is None:
            raise Exception("Person not found")
        return obj

    def get_face_by_id(self, session, id: int) -> PersonFace:
        obj = session.execute(select(PersonFace).where(PersonFace.id == id)).first()
        if obj is None:
            raise Exception("Person face not found")
        return obj

    def get_person_from_embedding(self, session, embedding: list[float] | np.ndarray, top_k: int = 1, person_id: int = None):
        try:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.astype(float).tolist()

            similarity_expr = 1 - PersonFace.embedding.cosine_distance(embedding)

            stmt = select(
                similarity_expr.label("distance"),
                PersonFace
            )

            if person_id is not None:
                stmt = stmt.where(PersonFace.person_id == person_id)

            stmt = (
                stmt.order_by(similarity_expr.desc())
                .limit(top_k)
            )

            result = session.execute(stmt).first()
            print(result)

            if result:
                return result
            
            else:
                raise Exception("face database is empty")
        except Exception as e:
            raise Exception(f"error in getting embedding {e}")
        
    # ---------- DELETE ----------
    def delete_face_with_id(self, session, id: int):
        try:
            statement = select(PersonFace).where(PersonFace.id == id)
            face = session.execute(statement).scalars().first()

            if face:
                session.delete(face)
                session.commit()

            return face
        
        except Exception as e:
            raise ValueError(e)
        
    def delete_all_rows_personface(self, session):
        session.execute(delete(PersonFace))
        session.commit()
