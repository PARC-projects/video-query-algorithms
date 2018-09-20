import numpy as np
import os
import logging

eps_threshold = float(os.environ["COMPUTE_EPS"])


class Hyperparameter:
    def __init__(self, default_weights, default_threshold=0.8, ballast=0.3, near_miss_default=0.5,
                 streams=('rgb', 'warped_optical_flow'), feature_name='global_pool', mu=.3, f_bootstrap=0.5):
        self.default_weights = default_weights  # e.g. {'rgb': 1.0, 'warped_optical_flow': 1.5}
        self.weights = {}
        self.default_threshold = default_threshold
        self.threshold = self.default_threshold
        self.near_miss_default = near_miss_default
        self.streams = streams
        self.feature_name = feature_name
        self.ballast = ballast
        self.weight_grid = np.arange(0.5, 2.5, 0.05)
        self.threshold_grid = np.arange(0.5, 1.1, 0.02)
        self.mu = mu
        self.f_bootstrap = f_bootstrap

    def optimize_weights(self, ticket):
        """
        Conditions:
        :param ticket: job ticket, instance of Ticket class
        :return: scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
                 new_weights: {<stream>: weight}  there should be an entry for every item in streams.
                 threshold_optimum: real value of computed threshold to use to separate matches from non-matches

         finds grid point with minimum loss, and locally fits a parabola to further minimize
         user_matches = {<video clip id>: <0 or 1 to indicate whether user says it is a match>}
         dims is number of grid dimensions:
           1st is for threshold, the cutoff score for match vs. not a match
           remainder are for streams 2, 3, ....
           weight for rgb stream is set equal to one, since it otherwise would get normalized out
        """
        match_status = self.get_info_about_matches_in_ticket(ticket)
        ith0, iw0, losses = self.compute_loss(match_status, ticket)
        threshold_optimum, weight_optimum = self.fit_losses(ith0, iw0, losses)
        self.add_buffer(threshold_optimum)
        self.weights = {
            self.streams[0]: 1.0,
            self.streams[1]: weight_optimum
        }

    @staticmethod
    def get_info_about_matches_in_ticket(ticket):
        # get info about matches in the ticket
        match_status = {}
        for match in ticket.matches:
            if match["user_match"] is not None:
                match_status[match['video_clip']] = match["user_match"]  # For user_match == True or False
            else:
                match_status[match['video_clip']] = match["is_match"]  # For clips the user did not evaluate
        return match_status

    def compute_loss(self, match_status, ticket):
        # compute loss function and find minimum.
        # Loss = 0 for correct scores
        # Loss = abs(score - th) for false positive
        # Loss = abs(score - th)*(1 + ballast) for false negative
        losses = 100 * np.ones([self.weight_grid.shape[0], self.threshold_grid.shape[0]])  # initialize loss matrix
        for iw, w in enumerate(self.weight_grid):
            ticket.compute_scores({self.streams[0]: 1.0, self.streams[1]: w})
            for ith, th in enumerate(self.threshold_grid):
                loss = 0
                for video_clip_id in match_status:
                    score = ticket.scores[video_clip_id]
                    loss += (np.heaviside(score - th, 1) - match_status[video_clip_id]) * (score - th) \
                            * (1 + match_status[video_clip_id] * self.ballast)
                losses[iw, ith] = loss / len(match_status)
        [iw0, ith0] = np.unravel_index(np.argmin(losses, axis=None), losses.shape)
        return ith0, iw0, losses

    def fit_losses(self, ith0, iw0, losses):
        # fit losses around minimum to a parabola and fine tune the minimum, unless minimum is on the border of the grid
        if iw0 == 0 or ith0 == 0 or iw0 == len(self.weight_grid) - 1 or ith0 == len(self.threshold_grid) - 1:
            weight_optimum = self.weight_grid[iw0]
            threshold_optimum = self.threshold_grid[ith0]
        else:
            weight_optimum, threshold_optimum = self.fine_tune(iw0, ith0, losses)
        return threshold_optimum, weight_optimum

    def add_buffer(self, threshold_optimum):
        self.threshold = threshold_optimum - eps_threshold  # add a small buffer to account for round-off errors

    def fine_tune(self, iw0, ith0, losses):
        xrange = [(self.weight_grid[iw0 - 1], self.weight_grid[iw0], self.weight_grid[iw0 + 1]),
                  (self.threshold_grid[ith0 - 1], self.threshold_grid[ith0], self.threshold_grid[ith0 + 1])]
        ydata = [losses[iw0 - 1, ith0], losses[iw0, ith0 - 1], losses[iw0, ith0], losses[iw0, ith0 + 1],
                 losses[iw0 + 1, ith0]]
        return self._quad_fit(xrange, ydata)

    @staticmethod
    def _quad_fit(x, y):
        # determine the parameters of a0 * (x[0] - w0) ** 2 + b0 * (x[1] - th0) ** 2 + c0 that fit the five ydata values
        w0 = (y[4]-y[0]) * x[0][1]**2 + (y[2]-y[4]) * x[0][0]**2 - (y[2]-y[0]) * x[0][2]**2
        w0 = 0.5 * w0 / ((y[4]-y[0]) * x[0][1] + (y[2]-y[4]) * x[0][0] - (y[2]-y[0]) * x[0][2])
        a0 = (y[2]-y[0]) / ((x[0][1] - w0)**2 - (x[0][0] - w0)**2)
        th0 = (y[3]-y[1]) * x[1][1]**2 + (y[2]-y[3]) * x[1][0]**2 - (y[2]-y[1]) * x[1][2]**2
        th0 = 0.5 * th0 / ((y[3]-y[1]) * x[1][1] + (y[2]-y[3]) * x[1][0] - (y[2]-y[1]) * x[1][2])
        b0 = (y[2]-y[1]) / ((x[1][1] - th0)**2 - (x[1][0] - th0)**2)
        c0 = y[2] - a0 * (x[0][1] - w0)**2 - b0 * (x[1][1] - th0)**2

        # for flat y values, round-off errors could move w0 or th0 out of the range of x, so we correct for that
        w0 = min(w0, x[0][2])
        w0 = max(w0, x[0][0])
        th0 = min(th0, x[1][2])
        th0 = max(th0, x[1][0])

        # make sure fit is good
        eps = 10**-6
        y0 = a0 * (x[0][0]-w0)**2 + b0 * (x[1][1]-th0)**2 + c0
        y1 = a0 * (x[0][1] - w0) ** 2 + b0 * (x[1][0] - th0) ** 2 + c0
        y2 = a0 * (x[0][1] - w0) ** 2 + b0 * (x[1][1] - th0) ** 2 + c0
        y3 = a0 * (x[0][1] - w0) ** 2 + b0 * (x[1][2] - th0) ** 2 + c0
        y4 = a0 * (x[0][2] - w0) ** 2 + b0 * (x[1][1] - th0) ** 2 + c0
        if (abs(y[0]-y0) + abs(y[1]-y1) + abs(y[2]-y2) + abs(y[3]-y3) + abs(y[4]-y4)) > eps:
            logging.warning("hyperparameter quadratic fine tuning failed - resort to selecting optimum on grid without "
                            "further interpolation")
            w0 = x[0][1]
            th0 = x[1][1]
        return w0, th0
