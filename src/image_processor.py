"""
Image preprocessing techniques for facial expression recognition
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging
from scipy import ndimage
from skimage import exposure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image preprocessing pipeline for face expression recognition
    """
    
    def __init__(self, target_size: Tuple[int, int] = (48, 48),
                 use_gaussian: bool = True,
                 gaussian_kernel: Tuple[int, int] = (5, 5),
                 use_edge_detection: bool = False):
        """
        Initialize image processor
        
        Args:
            target_size: Target image size for CNN input
            use_gaussian: Whether to apply Gaussian blur
            gaussian_kernel: Kernel size for Gaussian blur
            use_edge_detection: Whether to apply edge detection
        """
        self.target_size = target_size
        self.use_gaussian = use_gaussian
        self.gaussian_kernel = gaussian_kernel
        self.use_edge_detection = use_edge_detection
    
    def preprocess(self, image: np.ndarray, 
                   normalize: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input image (face region)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale if color
        if len(image.shape) == 3:
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            processed = image.copy()
        
        # Apply Gaussian blur
        if self.use_gaussian:
            processed = self.apply_gaussian_blur(processed)
        
        # Apply edge detection
        if self.use_edge_detection:
            processed = self.apply_edge_detection(processed)
            # Convert to 3-channel for visualization
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        
        # Resize to target size
        processed = self.resize_image(processed, self.target_size)
        
        # Normalize
        if normalize:
            processed = self.normalize_image(processed)
        
        # Add channel dimension for CNN (if needed)
        if len(processed.shape) == 2:
            processed = np.expand_dims(processed, axis=-1)
        
        return processed
    
    def apply_gaussian_blur(self, image: np.ndarray, 
                            kernel_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Apply Gaussian blur to reduce noise
        
        Args:
            image: Input image
            kernel_size: Kernel size (default: from config)
            
        Returns:
            Blurred image
        """
        if kernel_size is None:
            kernel_size = self.gaussian_kernel
        
        return cv2.GaussianBlur(image, kernel_size, 0)
    
    def apply_edge_detection(self, image: np.ndarray, 
                             low_threshold: int = 50,
                             high_threshold: int = 150) -> np.ndarray:
        """
        Apply Canny edge detection
        
        Args:
            image: Input image (grayscale)
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            
        Returns:
            Edge-detected image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    def resize_image(self, image: np.ndarray, 
                     target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Input image
            target_size: Target size (default: from config)
            
        Returns:
            Resized image
        """
        if target_size is None:
            target_size = self.target_size
        
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray, 
                        method: str = 'minmax') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            method: 'minmax' or 'standard'
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == 'minmax':
            # Scale to [0, 1]
            if image.max() > 1:
                image = image / 255.0
        elif method == 'standard':
            # Standardization (zero mean, unit variance)
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
        
        return image
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations for training
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random rotation
        angle = np.random.uniform(-15, 15)
        rows, cols = image.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        augmented = cv2.warpAffine(image, M, (cols, rows))
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            augmented = cv2.flip(augmented, 1)
        
        # Random brightness adjustment
        brightness = np.random.uniform(0.8, 1.2)
        augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
        
        return augmented
    
    def extract_face_features(self, face_image: np.ndarray) -> dict:
        """
        Extract facial features for analysis
        
        Args:
            face_image: Face region image
            
        Returns:
            Dictionary of extracted features
        """
        gray = face_image if len(face_image.shape) == 2 else cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        features = {
            'mean_intensity': float(gray.mean()),
            'std_intensity': float(gray.std()),
            'min_intensity': float(gray.min()),
            'max_intensity': float(gray.max()),
            'contrast': float(gray.max() - gray.min()),
        }
        
        # Edge density
        edges = self.apply_edge_detection(gray)
        features['edge_density'] = float((edges > 0).sum() / edges.size)
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist = hist / hist.sum()
        
        features['histogram'] = hist.tolist()
        
        return features
