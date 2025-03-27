"""
Processing Data from KITTI Visual Odometry Dataset
"""

import os
import cv2
import pandas as pd
import numpy as np

class DataHandler:
    def __init__(self, sequence):
        # Set up directories
        self.sequence_path = os.path.join("data", "data_odometry_gray", "dataset", "sequences", sequence)
        self.pose_path = os.path.join("data", "data_odometry_pose", "dataset", "poses", f"{sequence}.txt")
        self.calibration_path = os.path.join("data", "data_odometry_cali", "dataset", "sequences", sequence)
        
        # Fetch image
        self.left_image_files = sorted(os.listdir(os.path.join(self.sequence_path, "image_0")), key = lambda x: int(x.split('.')[0]))
        self.right_image_files = sorted(os.listdir(os.path.join(self.sequence_path, "image_1")), key = lambda x: int(x.split('.')[0]))
        self.num_frames = len(self.left_image_files)

        # Fetch pose
        self.poses = pd.read_csv(self.pose_path, delimiter=" ", header=None)

        # Fetch calibration parameters
        full_calibration_path = os.path.join(self.calibration_path, "calib.txt")
        calibration_csv = pd.read_csv(full_calibration_path, delimiter=" ", header=None, index_col=0)
        self.P0 = np.array(calibration_csv.loc['P0:']).reshape((3, 4))
        self.P1 = np.array(calibration_csv.loc['P1:']).reshape((3, 4))
        self.P2 = np.array(calibration_csv.loc['P2:']).reshape((3, 4))
        self.P3 = np.array(calibration_csv.loc['P3:']).reshape((3, 4))

        # Fetch timestamp
        ts = np.array(pd.read_csv(os.path.join(self.calibration_path, "times.txt"), delimiter=" ", header=None))

        # Fetch ground truth poses
        self.gt = np.zeros((len(self.poses), 3, 4))
        for i in range(len(self.poses)):
            self.gt[i] = np.array(self.poses.iloc[i]).reshape((3, 4))

        self.reset_frames()

    def reset_frames(self):
        """
        Generator to reset first frame
        """
        self.left_image = (cv2.imread(os.path.join(self.sequence_path, "image_0", img), cv2.IMREAD_GRAYSCALE) for img in self.left_image_files)
        self.right_image = (cv2.imread(os.path.join(self.sequence_path, "image_0", img), cv2.IMREAD_GRAYSCALE) for img in self.right_image_files)
