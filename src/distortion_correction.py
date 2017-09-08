import cv2
import glob
import numpy as np
import matplotlib.image as mpimg

CORNER_W = 9
CORNER_H = 6
CAL_IMG_PATH = '../camera_cal/calibration*.jpg'

def get_dist_coeff():
    """
    Calculates the calibration coefficients for chessboard images.s
    :return: the calibration coeffitients [ret, mtx, dist, rvecs, tvecs]
    """
    # read in all chessboard images
    images = glob.glob(CAL_IMG_PATH)

    obj_points = []  # object points of corners in the image (3D)
    image_points = []  # pixel points in the image

    objp = np.zeros((CORNER_W * CORNER_H, 3), np.float32)  # the object point in the current image
    objp[:, :2] = np.mgrid[0:CORNER_W, 0:CORNER_H].T.reshape(-1, 2)  # get the objects in the right order
    img_shape = mpimg.imread(images[0]).shape

    for img_name in images:
        img = mpimg.imread(img_name)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (CORNER_W, CORNER_H), None)

        if ret:
            image_points.append(corners)
            obj_points.append(objp)

    return cv2.calibrateCamera(obj_points, image_points, img_shape[:-1], None, None)
