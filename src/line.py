import numpy as np

# history size
N = 10
class Line():
    """
    Define a class to receive the characteristics of each line detection
    """
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial cefficients all together in a list

        self.recent_fits = []
        # polynomial coefficients summed up over the last n iterations
        self.best_fit_cum = np.array([0, 0, 0], dtype='float')
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.lane_confidence = 1

        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

    def update_line(self, line_fit: np.array, line_fitx: np.array, line_fity: np.array, lane_pts: np.array):
        """
        Updates the line.
        :param line_fit: polynom of the current line
        :param line_fitx: x values of the current line
        :return:
        """
        if line_fit.any():
            self.detected = True
            self.recent_fits.append(line_fit)
            self.diffs = self.recent_fits[-1] - line_fit
            self.best_fit_cum += line_fit
            self.current_fit = line_fit
            self.best_fit = self.best_fit_cum/len(self.recent_fits)
            self.lane_confidence = len(lane_pts)/np.sum(self.diffs)
        else:
            self.detected = False
            self.lane_confidence = 0

        if line_fitx.any():
            self.recent_xfitted.append(line_fitx)
            if len(self.recent_xfitted) > N:
                self.recent_xfitted.pop(0)
            self.bestx = np.average(self.recent_xfitted, axis=0)

        # TODO
        self.radius_of_curvature = 0
        self.line_base_pos = None

        self.allx = line_fitx
        self.ally = line_fity

    def sanity_check(self):
        res = True
        res = res and self.detected
        res = res and np.sum(self.diffs) < 1
        return res