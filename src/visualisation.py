"""
Visualization utilities for expression detection
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    """
    Visualization tools for expression recognition
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """Initialize visualizer with style"""
        plt.style.use(style)
        self.class_names = ['angry', 'disgust', 'fear', 'happy', 
                           'neutral', 'sad', 'surprise']
        self.emotion_colors = {
            'angry': '#FF4444',
            'disgust': '#44FF44',
            'fear': '#FF44FF',
            'happy': '#FFD700',
            'neutral': '#888888',
            'sad': '#4444FF',
            'surprise': '#44FFFF'
        }
    
    def plot_training_history(self, history: dict, 
                              save_path: Optional[str] = None):
        """
        Plot training and validation metrics
        
        Args:
            history: Training history dictionary
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax = axes[0]
        ax.plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        ax.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Loss
        ax = axes[1]
        ax.plot(history['loss'], label='Training Loss', linewidth=2)
        ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Model Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
