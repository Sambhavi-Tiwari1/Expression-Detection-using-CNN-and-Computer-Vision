"""
CNN Model for Facial Expression Recognition
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import logging
import os
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpressionModel:
    """
    Convolutional Neural Network for facial expression recognition
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (48, 48, 1),
                 num_classes: int = 7,
                 learning_rate: float = 0.001):
        """
        Initialize expression recognition model
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of emotion classes
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 
                           'neutral', 'sad', 'surprise']
        
        self._build_model()
    
    def _build_model(self) -> None:
        """
        Build CNN architecture
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        logger.info(f"Model built with {self.model.count_params():,} parameters")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Number of classes: {self.num_classes}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              callbacks_list: Optional[list] = None,
              verbose: int = 1) -> dict:
        """
        Train the CNN model
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_val: Validation images
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size
            callbacks_list: List of Keras callbacks
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        logger.info(f"Starting training with {len(X_train)} samples")
        logger.info(f"Validation with {len(X_val)} samples")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict emotion from image
        
        Args:
            image: Preprocessed image (48x48x1)
            
        Returns:
            (predicted_class, probabilities)
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        probabilities = self.model.predict(image, verbose=0)[0]
        predicted_class = np.argmax(probabilities)
        
        return predicted_class, probabilities
    
    def predict_batch(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict emotions for batch of images
        
        Args:
            images: Batch of images
            
        Returns:
            (predicted_classes, probabilities)
        """
        probabilities = self.model.predict(images, verbose=0)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        return predicted_classes, probabilities
    
    def get_confident_predictions(self, image: np.ndarray, 
                                  confidence_threshold: float = 0.7) -> Tuple[int, np.ndarray, bool]:
        """
        Predict with confidence threshold
        
        Args:
            image: Preprocessed image
            confidence_threshold: Minimum confidence for valid prediction
            
        Returns:
            (predicted_class, probabilities, is_confident)
        """
        class_id, probabilities = self.predict(image)
        confidence = probabilities[class_id]
        is_confident = confidence >= confidence_threshold
        
        return class_id, probabilities, is_confident
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary of evaluation metrics
        """
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        predictions = self.model.predict(X_test, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        from sklearn.metrics import classification_report, confusion_matrix
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predicted_classes,
            'true_labels': true_classes,
            'classification_report': classification_report(true_classes, predicted_classes),
            'confusion_matrix': confusion_matrix(true_classes, predicted_classes)
        }
    
    def save_model(self, filepath: str = 'models/expression_model.h5'):
        """
        Save model weights
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/expression_model.h5'):
        """
        Load model weights
        
        Args:
            filepath: Path to saved model
        """
        if os.path.exists(filepath):
            self.model = models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        return self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
