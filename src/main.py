import cv2
import numpy as np
from moviepy.editor import VideoFileClip

import src.distortion_correction as dc
import src.image_filter as img_f
import src.sliding_window as sw
import src.existing_line_search as els
from src.line import Line
import src.img_projector as img_pr


import src.calc_curvature as cvr

# define Video File to use
VIDEO_PATH = '../project_video.mp4'
VIDEO_OUTPUT = '../output_images/output.mp4'

def process_image(img):
    """
    Detect lanes in the image.
    :param img: the image to detect lanes in
    :return: the image with detected lanes in
    """
    img_size = img.shape
    # undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # get a binary image using filters and sobel
    binary_output = img_f.get_binary_image(undist)

    # define undistortion points to use
    src_pt = np.float32([[img_size[1]*7/16, img_size[0]*2/3],
                         [img_size[1]*79/128, img_size[0]*2/3],
                         [img_size[1]*15/16, img_size[0]],
                         [img_size[1]*247/1280, img_size[0]]])

    dst_pt = np.float32([[img_size[1]*7/32, img_size[0]*5/9],
                         [img_size[1]*15/16, img_size[0]*5/9],
                         [img_size[1]*105/128, img_size[0]],
                         [img_size[1]*7/32, img_size[0]]])

    # transforming the image to top down view
    M = cv2.getPerspectiveTransform(src_pt, dst_pt)
    Minv = cv2.getPerspectiveTransform(dst_pt, src_pt)
    warped = cv2.warpPerspective(binary_output, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)

    # sanity check the current line
    if lline.sanity_check() and rline.sanity_check():
        # use existing line to find line in the next frame
        left_fit, right_fit, left_fitx, right_fitx, ploty, result = els.existing_line_search(warped,
                                                                                             lline.recent_fits[-1],
                                                                                             rline.recent_fits[-1],
                                                                                             margin=100)
    else:
        # search for inital lines or reset the lines and search again if lane are lost
        left_fit, right_fit, left_fitx, right_fitx, ploty, result = sw.sliding_window(warped)

    # update lanes
    lline.update_line(line_fit=left_fit, line_fitx=left_fitx, line_fity=ploty)
    rline.update_line(line_fit=right_fit, line_fitx=right_fitx, line_fity=ploty)

    result = img_pr.project_back(img, undist, warped, Minv, lline.bestx, rline.bestx, ploty)

    mean_rad = (abs(lline.radius_of_curvature) + abs(rline.radius_of_curvature))/2
    car_dist = cvr.cal_dist(left_fitx[-1], right_fitx[-1], img_size[1])

    # update lines distances
    lline.line_base_pos =  car_dist
    rline.line_base_pos = car_dist
    lline.radius_of_curvature = mean_rad
    rline.radius_of_curvature = mean_rad

    return result

def main():
    """
    Main method of the programm.
    :return:
    """
    # detect lanes in the video file
    white_output = VIDEO_OUTPUT
    clip1 = VideoFileClip(VIDEO_PATH)
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

if __name__ == '__main__':
    # initialize camera undistortion
    ret, mtx, dist, rvecs, tvecs = dc.get_dist_coeff()
    # initialize line
    lline = Line()
    rline = Line()

    # run
    main()

