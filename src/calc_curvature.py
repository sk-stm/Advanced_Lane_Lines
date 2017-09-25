import numpy as np


def cal_rad(fitx, ploty):
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

    return curverad

def cal_dist(lfitx, rfitx, img_width):
    # Define conversions in x and y from pixels space to meters
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    img_middlex = img_width/2
    car_posx = abs(rfitx - lfitx)/2 + lfitx
    dist = img_middlex - car_posx
    dist_m = dist * xm_per_pix
    return dist_m