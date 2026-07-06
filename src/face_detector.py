"""
Face Detection using Haar Cascade Classifiers
"""
import cv2
import numpy as np
import os
import logging
from typing import Tuple, List, Optional
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Face detector using Haar Cascade classifier
    """
    
    def __init__(self, cascade_path: str = None, scale_factor: float = 1.1,
                 min_neighbors: int = 5, min_size: Tuple[int, int] = (30, 30)):
        """
        Initialize face detector
        
        Args:
            cascade_path: Path to Haar cascade XML file
            scale_factor: Parameter specifying how much image size is reduced
            min_neighbors: Minimum number of neighbors for detection
            min_size: Minimum face size
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        
        # Load cascade classifier
        if cascade_path is None or not os.path.exists(cascade_path):
            cascade_path = self._download_cascade()
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise ValueError("Failed to load Haar cascade classifier")
        
        logger.info(f"Face detector initialized with cascade: {cascade_path}")
    
    def _download_cascade(self) -> str:
        """Download Haar cascade file if not available"""
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        local_path = "haarcascade_frontalface_default.xml"
        
        if not os.path.exists(local_path):
            logger.info("Downloading Haar cascade classifier...")
            response = requests.get(url)
            with open(local_path, 'wb') as f:
                f.write(response.content)
            logger.info("Download complete")
        
        return local_path
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            List of face rectangles (x, y, w, h)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )
        
        return list(faces)
    
    def get_face_regions(self, image: np.ndarray, 
                         padding: float = 0.2) -> List[np.ndarray]:
        """
        Extract face regions from image
        
        Args:
            image: Input image
            padding: Padding around face (percentage)
            
        Returns:
            List of face region images
        """
        faces = self.detect_faces(image)
        face_images = []
        
        h, w = image.shape[:2]
        
        for (x, y, fw, fh) in faces:
            # Add padding
            pad_x = int(fw * padding)
            pad_y = int(fh * padding)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)
            
            face_img = image[y1:y2, x1:x2]
            face_images.append(face_img)
        
        return face_images
    
    def draw_faces(self, image: np.ndarray, 
                   faces: List[Tuple[int, int, int, int]] = None,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
        """
        Draw rectangles around detected faces
        
        Args:
            image: Input image
            faces: List of face rectangles (if None, detect faces)
            color: Rectangle color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with face rectangles drawn
        """
        img_copy = image.copy()
        
        if faces is None:
            faces = self.detect_faces(image)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, thickness)
        
        return img_copy
    
    def get_face_landmarks(self, image: np.ndarray) -> List[dict]:
        """
        Get face landmarks (using Haar cascade - simplified version)
        Note: For full landmarks, use dlib or mediapipe
        
        Args:
            image: Input image
            
        Returns:
            List of landmark dictionaries
        """
        # Simplified - just returns face bounding box
        faces = self.detect_faces(image)
        landmarks = []
        
        for (x, y, w, h) in faces:
            landmarks.append({
                'bbox': (x, y, w, h),
                'center': (x + w//2, y + h//2)
            })
        
        return landmarks
