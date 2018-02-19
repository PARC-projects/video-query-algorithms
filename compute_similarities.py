import psycopg2
import numpy as np
import csv
from scipy.optimize import curve_fit
"""
Input is the reference clip video and time
For now, assume clip duration of 10 sec.
Also assume warped optical flow and rgb are being used
"""


def compute_similarities(ref_video, ref_clip, search_set='all', streams=('rgb', 'warped_optical_flow'),
                         feature_name='global_pool', clip_duration=10):
    '''
    Conditions
    :param ref_video: str name of reference video
    :param ref_clip: int clip number for the reference video.
                    For a 10 sec clip duration, ref_clip = int( reference time in seconds / 10 ) + 1
    :param search_set: str name of the search set that the user wants to search over.  Currently not used.
                    For now, we search over all clips in the database
    :param streams: tuple of names of streams that we are using to assess similarity.
    :param feature_name:  str name of feature for which similarities are to be computed.
    :param clip_duration: int duration in seconds of clips of interest
    :return: avgd_similarities = {video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]}}
    '''

    # database connection
    conn_cs = psycopg2.connect("host=localhost dbname=video-query user=torres")
    cur_cs = conn_cs.cursor()
    # query for the video_clip_id corresponding to the reference clip
    cur_cs.execute(
        "SELECT id FROM video_clips "
        "WHERE video = %s AND clip = %s AND clip_duration = %s;",
        [ref_video, ref_clip, clip_duration]
    )
    ref_clip_id = cur_cs.fetchall()
    assert len(ref_clip_id) == 1, "Error in database table, mroe than one reference clip found!"
    # query db for the feature of the reference video clip, for all streams and splits
    cur_cs.execute(
        "SELECT dnn_stream, dnn_stream_split, feature FROM features "
        "WHERE video_clip_id = %s AND feature_name = %s;",
        [ref_clip_id[0], feature_name]
    )
    ref_features = cur_cs.fetchall()
    # set up reference feature dictionary with entries like { <stream type>: {<split #>:[<feature>], ...} }
    ref_dict = {}
    for stream_type in streams:
        ref_dict[stream_type] = {}
    for feature_list in ref_features:
        stream_type = feature_list[0]
        fsplit = feature_list[1]
        feature_vector = feature_list[2]
        if stream_type in streams:
            ref_dict[stream_type][fsplit] = feature_vector

    # compute emsemble averaged similarities for each stream type in streams
    # LATER:  need to add constraint for the target video being in the search set
    avgd_similarities = {}
    for stream_type, all_splits in ref_dict.items():
        # compute similarities for each split
        # load into similarities dictionary with key:value entries of
        # id:[<split 1 similarity>, <split 2 similarity>, <split 3 similarity>], not necessarily in order
        similarities = {}
        for split, feature_vector in all_splits.items():
            # query the features db for each entry with a matching dnn_stream, dnn_stream_split & feature name
            cur_cs.execute(
                "SELECT video_clip_id, feature FROM features "
                "WHERE dnn_stream = %s AND dnn_stream_split = %s AND feature_name = %s;",
                [stream_type, split, feature_name]
            )
            target_features = cur_cs.fetchall()

            # compute dot product similarities for matching dnn_stream: key:value is id:[<similarity
            for target in target_features:
                similarity = np.dot(feature_vector, target[1]) / np.linalg.norm(feature_vector)**2
                similarities[target[0]] = similarities.get(target[0], []) + [similarity]

        # ensemble average over the splits for each id
        # dictionary structure is { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
        for clip_id, sim_array in similarities.items():
            id_len = len(sim_array)
            avg_sim = sum(sim_array) / id_len
            avgd_similarities[clip_id] = avgd_similarities.get(clip_id, {})
            avgd_similarities[clip_id].update({stream_type: [avg_sim, id_len]})
    cur_cs.close()
    conn_cs.close()
    return avgd_similarities


def compute_score(similarities, weights):
    '''
    Conditions:
    :param similarities: { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
    :param weights: {<stream_type>:<weight>}
    :return: scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
    '''

    scores = {}
    for video_clip_id, vsim in similarities.items():
        ssum = 0
        denom = 0
        for stream_type, w in weights.items():
            ssum += (w * (1 - vsim[stream_type][0])) ** 2
            denom += w ** 2
        vscore = np.sqrt(ssum / denom)
        scores[video_clip_id] = 1 - vscore
    return scores


def optimize_weights(similarities, user_matches, streams=('rgb', 'warped_optical_flow')):
    '''
    Conditions:
    :param similarities: { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
    :param user_matches: {<video clip id>: <0 or 1 to indicate whether user says it is a match>}
    :param streams: tuple of names of streams that we are using to assess similarity.
    :return: scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
             new_weights: {<stream>: weight}  there should be an entry for every item in streams.
             threshold_optimum: real value of computed threshold to use to separate matches from non-matches
    '''
    # finds grid point with minimum loss, and locally fits a parabola to further minimize
    # user_matches = {<video clip id>: <0 or 1 to indicate whether user says it is a match>}
    # dims is number of grid dimensions:
    #   1st is for threshold, the cutoff score for match vs. not a match
    #   remainder are for streams 2, 3, ....
    #   weight for rgb stream is set equal to one, since it otherwise would get normalized out

    # set up grid of weight & threshold
    weight_grid = np.arange(0.5, 2.5, 0.1)
    threshold_grid = np.arange(0.6, 1.0, 0.05)
    # compute loss function (L2 loss) and find minimum
    losses = 10 * np.ones([weight_grid.shape[0], threshold_grid.shape[0]])     # initialize loss matrix
    for iw, w in enumerate(weight_grid):
        test = compute_score(similarities, {streams[0]: 1.0, streams[1]: w})
        for ith, th in enumerate(threshold_grid):
            loss = 0
            for video_clip_id, score in test.items():
                loss += ((np.heaviside((score - th), 1) - user_matches[video_clip_id]) * (score - th)) ** 2
            losses[iw, ith] = loss / len(test)
    [iw0, ith0] = np.unravel_index(np.argmin(losses, axis=None), losses.shape)
    print(iw0, ith0)
    # fit losses around minimum to a parabola and fine tune the minimum
    xrange = []
    ydata = []
    xrange.append((weight_grid[iw0 - 1], weight_grid[iw0], weight_grid[iw0], weight_grid[iw0], weight_grid[iw0 + 1]))
    xrange.append((threshold_grid[ith0], threshold_grid[ith0 - 1], threshold_grid[ith0], threshold_grid[ith0 + 1],
                   threshold_grid[ith0]))
    ydata.append(losses[iw0 - 1, ith0])
    ydata.append(losses[iw0, ith0 - 1])
    ydata.append(losses[iw0, ith0])
    ydata.append(losses[iw0, ith0 + 1])
    ydata.append(losses[iw0 + 1, ith0])
    popt, _ = curve_fit(quad_fun, xrange, ydata)
    weight_optimum = popt[3]
    threshold_optimum = popt[4]
    # compute score at optimal weight and return
    new_weights = {streams[0]: 1.0, streams[1]: weight_optimum}
    score_optimized = compute_score(similarities, new_weights)
    return score_optimized, new_weights, threshold_optimum


def quad_fun(x, a0, b0, c0, w0, th0):
    # function provided to scipy.optimize.curve_fit
    return a0 * (x[0] - w0) ** 2 + b0 * (x[1] - th0) ** 2 + c0


if __name__ == '__main__':
    result = compute_similarities('S06NDS_Sample_120406_1451_00186_Forward', 11, '')
    assert 1==0
    clip_scores = compute_score(result, {'rgb': 1.0, 'warped_optical_flow': 1.5})
    conn = psycopg2.connect("host=localhost dbname=video-query user=torres")
    cur = conn.cursor()
    with open('test_file.csv', 'w') as f:
        simwriter = csv.writer(f)
        for video_id, sims in result.items():
            for stream, simvalue in sims.items():
                cur.execute(
                    "SELECT video, clip FROM video_clips "
                    "WHERE id = %s;",
                    [video_id]
                )
                video_clip = cur.fetchone()
                if video_clip[0] == 'S06NDS_Sample_120406_1451_00186_Forward':
                    simwriter.writerow([video_clip[1], stream, simvalue[0], clip_scores[video_id]])
                    assert simvalue[1] == 3

    # checked with calculations I did manually, all similarities match with 1e-7
    # file with manual calculations is similarityPlots.xlsx
    test_matches = {}
    for iclip, _ in result.items():
        test_matches[iclip] = 0
    for iclip in (12463, 12464, 12465, 12466, 12469, 12470, 12488, 12490):
        test_matches[iclip] = 1
    new_scores, new_weight, new_threshold = optimize_weights(result, test_matches)
    # output to file for checking results
    with open('test_optimized_file.csv', 'w') as f:
        simwriter = csv.writer(f)
        for video_id, sims in result.items():
            for stream, simvalue in sims.items():
                cur.execute(
                    "SELECT video, clip FROM video_clips "
                    "WHERE id = %s;",
                    [video_id]
                )
                video_clip = cur.fetchone()
                if video_clip[0] == 'S06NDS_Sample_120406_1451_00186_Forward':
                    simwriter.writerow([video_clip[1], stream, simvalue[0], clip_scores[video_id]])
                    assert simvalue[1] == 3
    print("optimum weight = {}, tuned threshold = {}".format(new_weight, new_threshold))
    # close database session
    cur.close()
    conn.close()
