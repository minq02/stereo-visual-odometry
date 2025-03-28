import cv2
import numpy as np

def compute_disparity(left_frame, right_frame):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*6,
        blockSize=7,
        P1=8*1*5**2,
        P2=32*1*5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disparity = stereo.compute(left_frame, right_frame).astype(np.float32) / 16.0

    return disparity