import numpy as np
import cv2
import matplotlib.image as mpimg

def _abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel = 3):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sob = np.abs(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sca_sob = np.uint8(255 * abs_sob / np.max(abs_sob))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(sca_sob)
    sbinary[(sca_sob >= thresh[0]) & (sca_sob <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary


def _mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    abs_xy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sca = np.uint8(abs_xy * 255 / np.max(abs_xy))
    # 5) Create a binary mask where mag thresholds are met
    bi = np.zeros_like(sca)
    bi[(sca > mag_thresh[0]) & (sca < mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return bi


def _dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_x = np.abs(sobx)
    abs_y = np.abs(soby)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_y, abs_x)
    # 5) Create a binary mask where direction thresholds are met
    bi = np.zeros_like(grad_dir)
    bi[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return bi

def _color_thresh(img, thresh=(170, 255)):
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return s_binary


def get_binary_image(image: np.array):

    # Choose a Sobel kernel size
    ksize = 3 # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = _abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(50, 100))
    grady = _abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = _mag_thresh(image, sobel_kernel=5, mag_thresh=(30, 100))
    dir_binary = _dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    s_binary = _color_thresh(image, thresh=(170, 255))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    return combined
