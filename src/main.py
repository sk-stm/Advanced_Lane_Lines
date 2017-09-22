import cv2
import numpy as np
from moviepy.editor import VideoFileClip

import src.distortion_correction as dc
import src.image_filter as img_f
import src.sliding_window as sw
import src.existing_line_search as els
from src.line import Line
import src.img_projector as img_pr
import src.test_curvature as crv
from PIL import Image
from PIL import ImageDraw

# define Video File to use
VIDEO_PATH = '../project_video.mp4'
VIDEO_OUTPUT = '../output_images/output.mp4'

# define undistortion points to use
src_pt = np.float32([[560, 480],
                     [791, 480],
                     [1200, 720],
                     [247, 720]])

dst_pt = np.float32([[280, 400],
                     [1200, 400],
                     [1050, 750],
                     [280, 750]])


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
    warped = cv2.warpPerspective(binary_output, M, (img_size[1], img_size[0]), flags=cv2.INTER_LINEAR)


    # sanity check the current line
    if lline.sanity_check() and rline.sanity_check():
        # use existing line to find line in the next frame
        left_fit, right_fit, left_fitx, right_fitx, ploty, result, left_lane_pts, right_lane_pts = els.existing_line_search(warped,
                                                                                             lline.recent_fits[-1],
                                                                                             rline.recent_fits[-1],
                                                                                             margin=100)
    else:
        # search for inital lines or reset the lines and search again if lane are lost
        left_fit, right_fit, left_fitx, right_fitx, ploty, result, left_lane_pts, right_lane_pts = sw.sliding_window(warped)

    # update lanes
    lline.update_line(line_fit=left_fit, line_fitx=left_fitx, line_fity=ploty, lane_pts=left_lane_pts)
    rline.update_line(line_fit=right_fit, line_fitx=right_fitx, line_fity=ploty, lane_pts=right_lane_pts)

    result = img_pr.project_back(img, undist, warped, Minv, lline.bestx, rline.bestx, ploty)

    # calc radius
    left_rad, right_rad = crv.cal_rad(left_fit, right_fit, left_fitx, right_fitx, ploty)

    mean_rad = (abs(left_rad) + abs(right_rad))/2

    image_result = Image.fromarray(result)
    draw = ImageDraw.Draw(image_result)
    draw.text((0, 0), "Radius of Curvature = " + str(mean_rad), (255, 255, 255))
    image_result.save('sample-out.jpg')

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
    # transforming the image to top down view
    M = cv2.getPerspectiveTransform(src_pt, dst_pt)
    Minv = cv2.getPerspectiveTransform(dst_pt, src_pt)
    # initialize line
    lline = Line()
    rline = Line()

    # run
    main()

