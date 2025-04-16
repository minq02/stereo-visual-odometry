import cv2
import numpy as np

def compute_disparity(left_frame, right_frame):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*10,     # Increased range
        blockSize=11,             # Better for textureless regions
        P1=8 * 3 * 11**2,
        P2=32 * 3 * 11**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,       # More selective
        speckleWindowSize=200,    # Remove small speckles
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(left_frame, right_frame).astype(np.float32) / 16.0

    return disparity

def compute_depth(disparity, k_left, t_left, t_right):
    f = k_left[0, 0]
    b = abs(t_right[0] - t_left[0])

    depth = np.zeros_like(disparity, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (f * b) / disparity[valid]

    return depth

def extract_features(frame):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(frame, None)
    return kp, des