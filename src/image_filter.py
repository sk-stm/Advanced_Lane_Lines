import numpy as np
import cv2
import matplotlib.image as mpimg

def _abs_sobel_thresh(img, orient='x', thresh=(0, 255), sobel_kernel = 3):
    """
    Calculates the gradient using the sobel gradient for the specified direction and filters the outcome
     with a threshold.
    :param img: the image to find the gradient in.
    :param orient: the direction to search for the gradient
    :param thresh: the threshold to filter the outcome
    :param sobel_kernel: the kernel size of the sobel matrix (must be an odd number)
    :return:
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the derivative or gradient
    abs_sob = np.abs(sobel)
    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sca_sob = np.uint8(255 * abs_sob / np.max(abs_sob))
    # Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(sca_sob)
    sbinary[(sca_sob >= thresh[0]) & (sca_sob <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return sbinary


def _mag_thresh(img, mag_thresh=(0, 255)):
    """
    Filters img by applying sobel operator in both directions and the filtering the magnitude of the gradient
    :param img: image to filter
    :param mag_thresh: threshold of allowed magnitude
    :return: filtered image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # Calculate the magnitude
    abs_xy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    sca = np.uint8(abs_xy * 255 / np.max(abs_xy))
    # Create a binary mask where mag thresholds are met
    bi = np.zeros_like(sca)
    bi[(sca > mag_thresh[0]) & (sca < mag_thresh[1])] = 1
    # Return this mask as your binary_output image
    return bi


def _dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    """
    Filters image for gradients with a certain direction.
    :param img: image to filter
    :param sobel_kernel: sobel kernel size
    :param thresh: angles that are allowed to pass
    :return: filtered image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    soby = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_x = np.abs(sobx)
    abs_y = np.abs(soby)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    grad_dir = np.arctan2(abs_y, abs_x)
    # Create a binary mask where direction thresholds are met
    bi = np.zeros_like(grad_dir)
    bi[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    # Return this mask as your binary_output image
    return bi

def _color_thresh(img, thresh=(170, 255)):
    """
    Filters the image by color in the S channel of HLS color scale.
    :param img: image to filter
    :param thresh: value range that is allowed in the image
    :return: filtered image
    """
    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1

    return s_binary


def get_binary_image(image: np.array):
    """
    Allpies all filters to the image to create a binary image with only the unfiltered pixels remaining.
    The filters are tuned for lane detection
    :param image: image to filter
    :return: filtered image
    """
    # apply the filters
    gradx = _abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 100))
    grady = _abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
    mag_binary = _mag_thresh(image, mag_thresh=(30, 100))
    dir_binary = _dir_threshold(image, sobel_kernel=15, thresh=(0.7, 1.3))
    s_binary = _color_thresh(image, thresh=(170, 255))

    # combine the filters
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    return combined
