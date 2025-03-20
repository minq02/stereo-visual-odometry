import os
import cv2
import numpy as np

class InputProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_images(self, sequence):
        left_path = os.path.join(self.data_dir, "data_odometry_gray", "dataset", "sequences", sequence, "image_0")
        right_path = os.path.join(self.data_dir, "data_odometry_gray", "dataset","sequences", sequence, "image_1")

        if not os.path.exists(left_path) or not os.path.exists(right_path):
            raise FileNotFoundError("KITTI dataset not found. Check Path")
        
        left_images = sorted([os.path.join(left_path, f) for f in os.listdir(left_path) if f.endswith(".png")])
        right_images = sorted([os.path.join(right_path, f) for f in os.listdir(right_path) if f.endswith(".png")])

        if left_images and right_images:
            left_img = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
            right_img = cv2.imread(right_images[0], cv2.IMREAD_GRAYSCALE)

            cv2.imshow("Left Image", left_img)
            cv2.imshow("Right Image", right_img)
            print(f"Showing Image Sequence {sequence}: \n{left_images[0]}\n{right_images[0]}")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No images found in the dataset folder.")