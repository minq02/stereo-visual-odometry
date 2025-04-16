from data_handler import DataHandler
import frame_processing

import matplotlib.pyplot as plt
import numpy as np
import cv2

handler = DataHandler(sequence="00")

for i in range(handler.num_frames):
    frame = next(handler.left_image)  # Or any image source
    keypoints, descriptors = frame_processing.extract_features(frame)

    # Convert to color so keypoints show up well
    frame_color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Draw keypoints
    img_with_kp = cv2.drawKeypoints(
        frame_color, keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Show video with keypoints
    cv2.imshow("SIFT Keypoints", img_with_kp)

    key = cv2.waitKey(30)  # Wait 30 ms per frame
    if key == 27:  # ESC key to break
        break

cv2.destroyAllWindows()

handler = DataHandler(sequence="00")

for i in range(handler.num_frames):
    left_frame = next(handler.left_image)
    right_frame = next(handler.right_image)

    disparity = frame_processing.compute_disparity(left_frame, right_frame)
    depth = frame_processing.compute_depth(disparity, handler.k0, handler.t0, handler.t1)

    # Disparity visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored_disp = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # Depth visualization with yellow invalids
    max_depth = 30
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    valid_mask = depth > 0
    depth_vis[valid_mask] = np.uint8(255 * (1.0 - np.clip(depth[valid_mask], 0, max_depth) / max_depth))
    colored_depth = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)

    # Show in separate windows
    cv2.imshow("Disparity", colored_disp)
    cv2.imshow("Depth", colored_depth)

    if cv2.waitKey(30) == 27:  # ESC
        break

cv2.destroyAllWindows()
