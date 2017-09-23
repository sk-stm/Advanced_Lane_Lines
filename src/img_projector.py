import numpy as np
import cv2

def project_back(img, undist, warped, Minv: np.array, left_fitx: np.array, right_fitx:np.array, ploty:np.array):
    """
    Project warped and detected lanes back to original image
    :param img: original image
    :param undist: undistorted image
    :param warped: warped image to project back
    :param Minv: inverse projection matrix
    :param left_fitx: x coordinates of the left lane
    :param right_fitx: x coordinated of the right lane
    :param ploty: y coordinates of the lanes
    :return: back projected image fused the original one
    """
    # draw on pictures
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result