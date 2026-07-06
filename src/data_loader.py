"""
Data loading and preprocessing for facial expression datasets
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Tuple, Dict, List, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader for facial expression recognition
    """
    
    def __init__(self, dataset_path: str = 'data/',
                 image_size: Tuple[int, int] = (48, 48),
                 test_size: float = 0.2,
                 validation_size: float = 0.2,
                 random_seed: int = 42):
        """
        Initialize data loader
        
        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size
            test_size: Proportion of test data
            validation_size: Proportion of validation data
            random_seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.random_seed = random_seed
        
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 
                           'neutral', 'sad', 'surprise']
        self.num_classes = len(self.class_names)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)
    
    def load_from_directory(self, directory: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from directory structure
        
        Expected structure:
            directory/
                class1/
                    image1.jpg
                    image2.jpg
                class2/
                    image3.jpg
                    ...
        
        Args:
            directory: Dataset directory (default: self.dataset_path)
            
        Returns:
            (images, labels)
        """
        if directory is None:
            directory = self.dataset_path
        
        images = []
        labels = []
        
        logger.info(f"Loading data from: {directory}")
        
        for class_name in self.class_names:
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        
                        # Resize
                        img = cv2.resize(img, self.image_size)
                        
                        # Normalize
                        img = img.astype(np.float32) / 255.0
                        
                        images.append(img)
                        labels.append(class_name)
                        
                    except Exception as e:
                        logger.error(f"Error loading {img_path}: {e}")
                        continue
        
        if len(images) == 0:
            raise ValueError(f"No images found in {directory}")
        
        X = np.array(images)
        X = X.reshape(-1, self.image_size[0], self.image_size[1], 1)
        
        # Encode labels
        y = self.label_encoder.transform(labels)
        
        logger.info(f"Loaded {len(X)} images from {len(np.unique(labels))} classes")
        
        return X, y
    
    def load_fer2013(self, filepath: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load FER2013 dataset (CSV format)
        
        Args:
            filepath: Path to fer2013.csv
            
        Returns:
            (images, labels)
        """
        if filepath is None:
            filepath = os.path.join(self.dataset_path, 'fer2013.csv')
        
        if not os.path.exists(filepath):
            logger.warning("FER2013 dataset not found. Please download from Kaggle.")
            return None, None
        
        df = pd.read_csv(filepath)
        
        # Parse pixels
        pixels = df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float32))
        images = np.array(pixels.tolist())
        images = images.reshape(-1, 48, 48, 1)
        
        # Normalize
        images = images / 255.0
        
        # Labels
        labels = df['emotion'].values
        
        logger.info(f"Loaded FER2013: {len(images)} images")
        
        return images, labels
    
    def create_dataset(self, X: np.ndarray, y: np.ndarray,
                       batch_size: int = 32,
                       augment: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset with augmentation
        
        Args:
            X: Image data
            y: Labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            
        Returns:
            TensorFlow dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if augment:
            dataset = dataset.map(self._augment_function, 
                                 num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _augment_function(self, image: tf.Tensor, label: tf.Tensor):
        """
        Data augmentation function
        """
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        
        # Random rotation
        angle = tf.random.uniform((-0.2, 0.2))
        image = tf.image.rot90(image, tf.cast(angle * 4, tf.int32))
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.1)
        
        # Random contrast
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        return image, label
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Image data
            y: Labels (can be class names or encoded integers)
            
        Returns:
            Dictionary with train, val, test splits
        """
        # Convert labels to categorical if needed
        if len(y.shape) == 1:
            y_categorical = to_categorical(y, self.num_classes)
        else:
            y_categorical = y
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=np.argmax(y_categorical, axis=1) if len(y_categorical.shape) > 1 else y_categorical
        )
        
        # Second split: train vs val
        val_size = self.validation_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.random_seed,
            stratify=np.argmax(y_temp, axis=1)
        )
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def get_class_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """
        Get class distribution
        
        Args:
            labels: Class labels
            
        Returns:
            Dictionary with class distributions
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        distribution = {}
        for label, count in zip(unique, counts):
            if label in self.class_names:
                distribution[label] = int(count)
            elif isinstance(label, (int, np.integer)):
                distribution[self.class_names[label]] = int(count)
        
        return distribution
