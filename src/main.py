import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

import src.distortion_correction as dc
import src.image_filter as img_f

TEST_IMG_PATH = '../test_images/test5.jpg'
src_pt = np.float32([[563, 480],
                     [717, 480],
                     [1020, 720],
                     [260, 720]])

dst_pt = np.float32([[360, 465],
                     [920, 465],
                     [920, 750],
                     [360, 750]])
# TEST_IMG_PATH = '../camera_cal/calibration*.jpg'

def main():
    # initialize
    # camera calibration
    # get distortion coefficients
    ret, mtx, dist, rvecs, tvecs = dc.get_dist_coeff()
    # get test images
    test_imgs = glob.glob(TEST_IMG_PATH)

    # run on test images
    for img_path in test_imgs:
        # load image
        img = mpimg.imread(img_path)
        img_size = img.shape
        # undistort the image
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        # get a binary image using filters and sobel
        binary_output = img_f.get_binary_image(undist)
        test_print(undist, binary_output)

        # transforming the image to top down view
        M = cv2.getPerspectiveTransform(src_pt, dst_pt)
        warped = cv2.warpPerspective(binary_output, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)
        test_print(undist, warped)

        # search for inital lines


        # expand lines for whole frame using sliding window

        # sanity check the current line

        # use existing line to find line in the next frame

        # reset the lines and search again if lane are lost

        # smooth lanes over the last estimations

        # draw on pictures

def test_print(before_img, after_img):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(before_img)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(after_img, cmap='gray')
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

if __name__ == '__main__':
    main()
