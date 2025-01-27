from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field
from typing import Annotated, Optional
import psycopg2
from psycopg2.pool import SimpleConnectionPool
import numpy as np
import tempfile
import os
import logging
import time
import json

from preprocessor import FaceProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

class PersonData(BaseModel):
    doc_id: str
    name: str
    address: str  
    age: int
    
    def __str__(self):
        return f"PersonData(id={self.doc_id}, name={self.name}, age={self.age})"
    
class RegisterBody(BaseModel):
    token: str
    person_data: PersonData

class DBManager:
    def __init__(self):
        self.conn_params = {
            'dbname': 'face_search',
            'user': 'grvsh02', 
            'password': 'postgress',
            'host': 'localhost',
            'port': '5432'
        }
        self.pool = SimpleConnectionPool(minconn=1, maxconn=10, **self.conn_params)
        logger.info("Initialized connection pool for PostgreSQL")

    def get_connection(self):
        try:
            conn = self.pool.getconn()
            logger.debug("Database connection acquired from pool")
            return conn
        except Exception as e:
            logger.error(f"Error acquiring connection from pool: {e}")
            raise

    def release_connection(self, conn):
        self.pool.putconn(conn)
        logger.debug("Database connection returned to pool")

    def save_face_embedding(self, face_embedding: np.ndarray):
        
        embedding = self.normalize_embedding(face_embedding.squeeze())

        uuid = str(uuid4())
        query = """
            INSERT INTO face_records (id, face_embedding)
            VALUES (%s, %s::vector);
            """
        
        start_time = time.time()
        conn = self.get_connection()
        try:

            with conn.cursor() as cur:
                cur.execute(query, (uuid, embedding.tolist()))
                conn.commit()
        finally:
            self.release_connection(conn)

        store_time = time.time() - start_time

        logger.info(f"Face embedding stored successfully in {store_time:.2f}s")
        return uuid
        

    def find_similar_face(self, embedding: np.ndarray, similarity_threshold: float = 0.2) -> Optional[dict]:
        logger.info(f"Searching for similar face with threshold {similarity_threshold}")
        
        embedding = self.normalize_embedding(embedding.squeeze())

        query = """
        SELECT id, name, address, age, 1 - (face_embedding <=> %s::vector) as similarity
        FROM face_records 
        WHERE 1 - (face_embedding <=> %s::vector) > %s
        ORDER BY face_embedding <=> %s::vector ASC
        LIMIT 1;
        """

        start_time = time.time()
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SET enable_seqscan = OFF;")
                params = (embedding.tolist(), embedding.tolist(), similarity_threshold, embedding.tolist())
                cur.execute(query, params)
                result = cur.fetchone()
        finally:
            self.release_connection(conn)

        query_time = time.time() - start_time
        logger.info(f"Face search completed in {query_time:.2f}s")

        if result:
            similar_face = {
                'id': result[0],
                'name': result[1],
                'address': result[2],
                'age': result[3],
                'similarity': float(result[4])
            }
            logger.info(f"Found similar face: {similar_face['name']} with similarity {similar_face['similarity']:.2f}")
            return similar_face

        logger.info("No similar face found")
        return None
    
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        logger.info(f"Fetching embedding for token: {token}")
        query = """
        SELECT face_embedding
        FROM face_records
        WHERE id = %s;
        """
        start_time = time.time()
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (token,))
                result = cur.fetchone()
        finally:
            self.release_connection(conn)

        query_time = time.time() - start_time
        logger.info(f"Embedding fetched in {query_time:.2f}s")

        if result:
            return np.array(result[0])
        return None

    def store_record(self, person_data: PersonData, embedding: np.ndarray):
        logger.info(f"Storing new record for {person_data}")

        # embedding = self.normalize_embedding(embedding.squeeze())

        query = """
        INSERT INTO face_records (id, name, address, age, face_embedding)
        VALUES (%s, %s, %s, %s, %s::vector);
        """

        start_time = time.time()
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (
                    int(person_data.doc_id),
                    person_data.name,
                    person_data.address,
                    person_data.age,
                    embedding.tolist()
                ))
                conn.commit()
        finally:
            self.release_connection(conn)

        store_time = time.time() - start_time
        logger.info(f"Record stored successfully in {store_time:.2f}s")

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm


class FaceAPI:
    def __init__(self):
        logger.info("Initializing FaceAPI")
        self.face_processor = FaceProcessor()
        self.db_manager = DBManager()

    async def register_face(self, token: str, person_data: PersonData) -> dict:
        logger.info(f"Registering face with token: {token}, Person data: {person_data}")
        embedding = self.db_manager.get_embedding(token)
        if not embedding:
            logger.warning("Invalid token provided")
            raise HTTPException(status_code=400, detail="Invalid token provided")
        self.db_manager.store_record(person_data, embedding)
        return {
            "status": "stored",
            "message": "New face record stored in database",
            "data": person_data.dict()
        }

    async def process_face(self, image: UploadFile, person_data: Optional[PersonData] = None) -> dict:
        logger.info(f"Processing face request. Image: {image.filename}, Person data provided: {person_data is not None}")
        start_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            content = await image.read()
            temp_file.write(content)
            temp_path = temp_file.name
            logger.debug(f"Saved uploaded image to temporary file: {temp_path}")

        try:
            logger.info("Starting face detection and embedding generation")
            results = self.face_processor.process_image(temp_path)
            
            if not results:
                logger.warning("No faces detected in image")
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            if len(results) > 1:
                logger.warning(f"Multiple faces ({len(results)}) detected in image")
                raise HTTPException(status_code=400, detail="Multiple faces detected in image")
                
            face_data, embedding = results[0]
            logger.info(f"Face detected with confidence: {face_data['confidence']:.2f}")
            
            similar_face = self.db_manager.find_similar_face(embedding)
            
            if similar_face:
                process_time = time.time() - start_time
                logger.info(f"Request completed in {process_time:.2f}s: Found similar face")
                return {
                    "status": "found",
                    "message": "Similar face found in database",
                    "data": similar_face
                }
            
            save_embedding = self.db_manager.save_face_embedding(embedding)
            
            # if person_data:
            #     self.db_manager.store_record(person_data, embedding)
            #     process_time = time.time() - start_time
            #     logger.info(f"Request completed in {process_time:.2f}s: Stored new face")
            #     return {
            #         "status": "stored",
            #         "message": "New face record stored in database",
            #         "data": person_data.dict()
            #     }
            
            process_time = time.time() - start_time
            logger.info(f"Request completed in {process_time:.2f}s: No similar face found")
            return {
                "status": "not_found",
                "message": "No similar face found in database please register",
                "data": {
                    "token": save_embedding,
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise
        finally:
            os.unlink(temp_path)
            logger.debug("Cleaned up temporary file")

face_api = FaceAPI()

@app.post("/face-search")
async def process_face_endpoint(
    image: UploadFile = File(...),
    person_data: str = Form(None)
):
    if person_data:
        person_data = PersonData(**json.loads(person_data))
    return await face_api.process_face(image, person_data)

@app.post("/register")
async def register_face_endpoint(
    body: RegisterBody
):
    return await face_api.register_face(body.token, body.person_data)
