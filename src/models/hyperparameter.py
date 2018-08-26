import numpy as np


class Hyperparameter:
    def __init__(self, default_weights, default_threshold=0.8, ballast=0.3, near_miss_default=0.5,
                 streams=('rgb', 'warped_optical_flow'), feature_name='global_pool'):
        self.default_weights = default_weights  # e.g. {'rgb': 1.0, 'warped_optical_flow': 1.5}
        self.weights = {}
        self.default_threshold = default_threshold
        self.threshold = self.default_threshold
        self.near_miss_default = near_miss_default
        self.streams = streams
        self.feature_name = feature_name
        self.ballast = ballast

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
        match_status = {}
        for match in ticket.matches:
            if match["user_match"]:
                match_status[match['video_clip']] = match["user_match"]  # For user_match == True or False
            else:
                match_status[match['video_clip']] = match["is_match"]  # For clips the user did not evaluate

        # set up grid of weight & threshold
        weight_grid = np.arange(0.5, 2.5, 0.05)
        threshold_grid = np.arange(0.6, 1.1, 0.01)

        # compute loss function and find minimum.
        # Loss = 0 for correct scores
        # Loss = abs(score - th) for false positive
        # Loss = abs(score - th)*(1 + ballast) for false negative
        losses = 100 * np.ones([weight_grid.shape[0], threshold_grid.shape[0]])     # initialize loss matrix
        for iw, w in enumerate(weight_grid):
            ticket.compute_scores({self.streams[0]: 1.0, self.streams[1]: w})
            for ith, th in enumerate(threshold_grid):
                loss = 0
                for video_clip_id, score in ticket.scores.items():
                    if video_clip_id in match_status:
                        loss += (np.heaviside(score - th, 1) - match_status[video_clip_id]) * (score - th) \
                                * (1 + match_status[video_clip_id]*self.ballast)
                losses[iw, ith] = loss / len(match_status)
        [iw0, ith0] = np.unravel_index(np.argmin(losses, axis=None), losses.shape)

        '''
        # fit losses around minimum to a parabola and fine tune the minimum, unless minimum is on the border of the grid
        xrange = []
        ydata = []
        if iw0 == 0 or ith0 == 0 or iw0 == len(weight_grid)-1 or ith0 == len(threshold_grid)-1:
            weight_optimum = weight_grid[iw0]
            threshold_optimum = threshold_grid[ith0]
        else:
            xrange.append((weight_grid[iw0 - 1], weight_grid[iw0], weight_grid[iw0], weight_grid[iw0],
                           weight_grid[iw0 + 1]))
            xrange.append((threshold_grid[ith0], threshold_grid[ith0 - 1], threshold_grid[ith0], threshold_grid[ith0+1],
                           threshold_grid[ith0]))
            ydata.append(losses[iw0 - 1, ith0])
            ydata.append(losses[iw0, ith0 - 1])
            ydata.append(losses[iw0, ith0])
            ydata.append(losses[iw0, ith0 + 1])
            ydata.append(losses[iw0 + 1, ith0])
            try:
                popt, _ = curve_fit(_quad_fun, xrange, ydata)
                weight_optimum = popt[3]
                threshold_optimum = popt[4]
            except Exception as e:
                print(e)
                # TODO: add explicit Jacobian to curve_fit above so exceptions are fewer to none
                weight_optimum = weight_grid[iw0]
                threshold_optimum = threshold_grid[ith0]
        '''
        self.threshold = threshold_grid[ith0]
        self.weights = {self.streams[0]: 1.0, self.streams[1]: weight_grid[iw0]}

    '''
        def _quad_fun(self, x, a0, b0, c0, w0, th0):
            # function provided to scipy.optimize.curve_fit
            return a0 * (x[0] - w0) ** 2 + b0 * (x[1] - th0) ** 2 + c0
    '''
