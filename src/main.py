from data_handler import DataHandler
import frame_processing

import matplotlib.pyplot as plt
import numpy as np
import cv2

handler = DataHandler(sequence="00")

for i in range(handler.num_frames):
    left_frame = next(handler.left_image)
    right_frame = next(handler.right_image)

    disparity = frame_processing.compute_disparity(left_frame, right_frame)

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow("Disparity Video", disp_vis)

    key = cv2.waitKey(30)
    if key == 27:  # ESC key to break
        break

cv2.destroyAllWindows()


