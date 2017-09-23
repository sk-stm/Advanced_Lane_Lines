import numpy as np
import src.calc_curvature as crv

# history size in frames
N = 10
class Line():
    """
    Define a class to receive the characteristics of each line detection
    :ivar detected: was the line detected in the last iteration?
    :ivar recent_xfitted: x values of the last n fits of the line
    :ivar bestx: average x values of the fitted line over the last n iterations
    :ivar recent_fits: polynomial cefficients all together in a list
    :ivar fest:fit_cum: polynomial coefficients summed up over the last n iterations
    :ivar best_fit: polynomial coefficients averaged over the last n iterations
    :ivar current_fit: polynomial coefficients for the most recent fit
    :ivar radius_curvature: radius of curvature of the line in some units
    :ivar line_base_pos: distance in meters of vehicle center from the line
    :ivar diffs: difference in fit coefficients between last and new fits
    :ivar allx: x values for detected line pixels
    :ivar ally: y values for detected line pixels
    """
    def __init__(self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.recent_fits = []
        self.best_fit_cum = np.array([0, 0, 0], dtype='float')
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

    def update_line(self, line_fit: np.array, line_fitx: np.array, line_fity: np.array):
        """
        Updates the line.
        :param line_fit: polynom of the current line
        :param line_fitx: x values of the current line
        """
        # update lane fit
        if line_fit.any():
            self.detected = True
            self.recent_fits.append(line_fit)
            self.diffs = self.recent_fits[-1] - line_fit
            self.best_fit_cum += line_fit
            self.current_fit = line_fit
            self.best_fit = self.best_fit_cum/len(self.recent_fits)
        else:
            self.detected = False

        # update line pxs
        if line_fitx.any():
            self.recent_xfitted.append(line_fitx)
            if len(self.recent_xfitted) > N:
                self.recent_xfitted.pop(0)
            self.bestx = np.average(self.recent_xfitted, axis=0)

        # calc radius
        self.radius_of_curvature = crv.cal_rad(self.bestx, line_fity)
        self.line_base_pos = None

        self.allx = line_fitx
        self.ally = line_fity

    def sanity_check(self):
        """
        Checks if the new line is reasonable with respect to the last one
        :return:
        """
        res = True
        res = res and self.detected
        res = res and np.sum(self.diffs) < 1  # experimental value
        return res
