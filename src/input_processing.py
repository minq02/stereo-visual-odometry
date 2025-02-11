import os
import cv2
import numpy as np

class KITTIInputProcessor:
    def __init__(self, data_dir="data", sequence="00"):
        """
        Initialize the input processor
        :param data_dir: Path to the KITTI dataset directory
        :param sequence: Sequence number to load
        """
        self.data_dir = data_dir
        self.sequence = sequence
        self.left_images, self.right_images = self._load_images()
    
    def _load_images(self):
        """Load left and right stereo images from the dataset"""
        left_path = os.path.join(self.data_dir, "data_odometry_gray", "dataset", "sequences", self.sequence, "image_0")
        right_path = os.path.join(self.data_dir, "data_odometry_gray", "dataset","sequences", self.sequence, "image_1")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            raise FileNotFoundError("KITTI dataset not found. Check Path")
        
        left_images = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if f.endswith(".png")])
        right_images = sorted([os.path.join(right_path, f) for f in os.listdir(right_path) if f.endswith(".png")])

        return left_images, right_images
        
