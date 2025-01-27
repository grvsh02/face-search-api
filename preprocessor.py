import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict
import tritonclient.http as httpclient
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FaceProcessor:
    def __init__(self, triton_url: str = "localhost:8000", confidence_threshold: float = 0.5):
        self.triton_url = triton_url
        self.confidence_threshold = confidence_threshold
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=confidence_threshold)
        self.client = None
        self._connect_client()
    def _connect_client(self) -> None:
        try:
            self.client = httpclient.InferenceServerClient(url=self.triton_url)
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    def detect_faces(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(image_rgb)
        faces = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                faces.append({
                    'box': [x, y, width, height],
                    'confidence': detection.score[0]
                })
        faces = [face for face in faces if face['confidence'] > self.confidence_threshold]
        if not faces:
            logger.warning("No faces detected in image")
            return [], image
        logger.info(f"Detected {len(faces)} faces with confidence > {self.confidence_threshold}")
        return faces, image
    def extract_and_align_face(self, image: np.ndarray, face_data: Dict) -> np.ndarray:
        x, y, width, height = face_data['box']
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (112, 112))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.transpose(2, 0, 1).astype(np.float32)
        face = (face - 127.5) / 128.0
        return np.expand_dims(face, axis=0)
    def generate_embedding(self, preprocessed_face: np.ndarray) -> np.ndarray:
        try:
            inputs = httpclient.InferInput("input", preprocessed_face.shape, "FP32")
            inputs.set_data_from_numpy(preprocessed_face)
            outputs = httpclient.InferRequestedOutput("output")
            response = self.client.infer(
                model_name="arcface",
                inputs=[inputs],
                outputs=[outputs]
            )
            embedding = response.as_numpy("output")
            return embedding.squeeze()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    def process_image(self, image_path: str) -> List[Tuple[Dict, np.ndarray]]:
        faces, image = self.detect_faces(image_path)
        if not faces:
            return []
        results = []
        for face_data in faces:
            try:
                aligned_face = self.extract_and_align_face(image, face_data)
                embedding = self.generate_embedding(aligned_face)
                results.append((face_data, embedding))
            except Exception as e:
                logger.error(f"Failed to process face: {e}")
                continue
        return results