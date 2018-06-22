import numpy as np
from scipy.optimize import curve_fit
import coreapi
from api.authenticate import authenticate
import random
import os
"""
    Feature lists returned by the API are as follows:
    [
        {
            "id": 6,
            "video_clip_id": 1,
            "dnn_stream_id": "warped_optical_flow",
            "dnn_stream_split": 3,
            "name": "global_pool",
            "dnn_weights_uri": "dummy_weights_uri",
            "dnn_spec_uri": "dummy_spec_uri",
            "feature_vector": [
                1.1,
                2.3,
                0.98,
                0.5,
                0.2,
                1.05,
                ...
            ]
        },
        ...
    ]
"""


def compute_similarities(query, base_url, streams=('rgb', 'warped_optical_flow'), feature_name='global_pool'):
    """
    Public contract to compute dictionary of averaged similarities for the query.
    Dictionary structure is { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }

    Args:
        query:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "tuing_update": QueryResult values for search tuning parameters for most recent
                                analysis of the query, including the current round
                "matches": for "revise" updates, matches of previous round
            }
        base_url:   url of Video-Query-API
        streams:    DNN streams to include in similarity computations
        feature_name:   name of DNN feature used for computing similarities
    """
    # Initialize an API client & load the schema document
    client = coreapi.Client(auth=authenticate(base_url))
    schema = client.get(os.path.join(base_url, "docs"))

    # get reference clip feature dictionary with entries like
    # { <stream type>: {<split #>: [<ref feature>], ...} }
    ref_dict, splits = _get_ref_clip_features(query["ref_clip_id"], client, schema, streams, feature_name)

    # get target feature dictionary with entries like
    # { <stream type>: {<split #>: { clip#: [<target feature>], ...} } }
    target_dict = _get_target_features(query["search_set"], client, schema, streams, splits, feature_name)

    # compute similarities and average them
    # avgd_similarities structure is { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
    avgd_similarities = {}
    for stream_type, all_splits in ref_dict.items():
        similarities = {}
        # compute dot product similarities{} for each split, saved as key:value = clip:array of similarities
        for split, ref_feature in all_splits.items():
            for clip, target_feature in target_dict[stream_type][split].items():
                similarity = np.dot(ref_feature, target_feature) / np.linalg.norm(ref_feature)**2
                similarities[clip] = similarities.get(clip, []) + [similarity]

        # ensemble average over the splits for each id
        for clip_id, sim_array in similarities.items():
            id_len = len(sim_array)
            avg_sim = sum(sim_array) / id_len
            # create dictionary item for clip_id if it does not exist:
            avgd_similarities[clip_id] = avgd_similarities.get(clip_id, {})
            avgd_similarities[clip_id].update({stream_type: [avg_sim, id_len]})

    return avgd_similarities


def select_matches(similarities, weights, threshold=0.8, max_number_matches=20, near_miss=0.5):
    """
    Find matches and near matches for review,
    half being above threshold and half for 1-(1+near_miss)*(1-threshold) < score < threshold.
    For review, matches are chosen randomly from all scores within the specified interval,
    with the total of matches and near matches being equal to or less than max_number_matches.

    Conditions:
    :param similarities:  { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
    :param weights:  weights for similarities for each stream
    :param threshold: real threshold, either the initial default or a new threshold_optimum from optimize_weights
    :param max_number_matches:  max number of matches the user wants to review.
    :param near_miss:  range of scores for near misses relative to the range (1-threshold) for hits
    :return: match_indicator: {<video_clip_id>: <True or False>}
    """
    scores = compute_score(similarities, weights)

    lower_limit = 1 - (1 + near_miss)*(1 - threshold)
    match_candidates = {k: v for k, v in scores.items() if v >= threshold}
    near_match_candidates = {k: v for k, v in scores.items() if lower_limit <= v < threshold}

    mscores = min(int(max_number_matches/2), len(match_candidates))
    m_near_scores = min(max_number_matches - mscores, len(near_match_candidates))
    match_scores = random.sample(match_candidates.items(), mscores)
    near_match_scores = random.sample(near_match_candidates.items(), m_near_scores)

    return dict(match_scores + near_match_scores)


def compute_score(similarities, weights):
    """
    Conditions:
        similarities: { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
        weights: {<stream_type>: <weight>}, e.g. {'rgb': 1.0, 'warped_optical_flow': 1.5}
        return: scores: {<video_clip_id>: score}  where <video_clip_id> is the id primary key in the video_clips table
    """

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


def optimize_weights(similarities, updated_matches, streams=('rgb', 'warped_optical_flow'), ballast=0.3):
    """
    Conditions:
    :param similarities: { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
    :param updated_matches: {<video clip id>: <0 or 1 to indicate whether user says it is a match>}
    :param streams: tuple of names of streams that we are using to assess similarity.
    :param ballast: extra penalty given to a user match falling below threshold vs. a non-match being above
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

    # set up grid of weight & threshold
    weight_grid = np.arange(0.5, 2.5, 0.05)
    threshold_grid = np.arange(0.6, 1.1, 0.025)

    # compute loss function and find minimum.  Loss = 0 for correct scores, abs(score - th) for incorrect scores
    losses = 10 * np.ones([weight_grid.shape[0], threshold_grid.shape[0]])     # initialize loss matrix
    for iw, w in enumerate(weight_grid):
        test = compute_score(similarities, {streams[0]: 1.0, streams[1]: w})
        for ith, th in enumerate(threshold_grid):
            loss = 0
            for video_clip_id, score in test.items():
                if video_clip_id in updated_matches:
                    loss += (np.heaviside(score - th, 0.5) - updated_matches[video_clip_id]) * (score - th) \
                            * (1 + updated_matches[video_clip_id]*ballast)
            losses[iw, ith] = loss / len(updated_matches)
    [iw0, ith0] = np.unravel_index(np.argmin(losses, axis=None), losses.shape)

    # fit losses around minimum to a parabola and fine tune the minimum, unless minimum is on the border of the grid
    xrange = []
    ydata = []
    if iw0 == 0 or ith0 == 0 or iw0 == len(weight_grid)-1 or ith0 == len(threshold_grid)-1:
        weight_optimum = weight_grid[iw0]
        threshold_optimum = threshold_grid[ith0]
    else:
        xrange.append((weight_grid[iw0 - 1], weight_grid[iw0], weight_grid[iw0], weight_grid[iw0],
                       weight_grid[iw0 + 1]))
        xrange.append((threshold_grid[ith0], threshold_grid[ith0 - 1], threshold_grid[ith0], threshold_grid[ith0 + 1],
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
        except:
            # TODO: add explicit Jacobian to curve_fit above so exceptions are fewer to none
            weight_optimum = weight_grid[iw0]
            threshold_optimum = threshold_grid[ith0]

    # compute score at optimal weight and return
    new_weights = {streams[0]: 1.0, streams[1]: weight_optimum}
    score_optimized = compute_score(similarities, new_weights)
    return score_optimized, new_weights, threshold_optimum


def _quad_fun(x, a0, b0, c0, w0, th0):
    # function provided to scipy.optimize.curve_fit
    return a0 * (x[0] - w0) ** 2 + b0 * (x[1] - th0) ** 2 + c0


def _get_ref_clip_features(query_ref_clip_id, client, schema, streams, feature_name):
    # Interact with the API endpoint to get features for the ref video clip
    action = ["video-clips", "features"]
    params = {"id": query_ref_clip_id}
    ref_features = client.action(schema, action, params=params)

    # set up reference feature dictionary with entries like { <stream type>: {<split #>:[<feature>], ...} }
    ref_dict = {}
    splits = set()
    for stream_type in streams:
        ref_dict[stream_type] = {}
    for feature_object in ref_features:
        stream_type = feature_object["dnn_stream_id"]  # this is actually the stream name, not the id
        fsplit = feature_object["dnn_stream_split"]
        splits.add(fsplit)  # splits is a set, so fsplit is only added if it is not already in splits
        name = feature_object["name"]
        feature_vector = feature_object["feature_vector"]
        if stream_type in streams and name == feature_name:
            ref_dict[stream_type][fsplit] = feature_vector
    return ref_dict, splits


def _get_target_features(search_set_id, client, schema, streams, splits, feature_name):
    # Create target feature dictionary with entries like
    # { <stream type>: {<split #>: { clip#: [<target feature>], ...} } }

    # Interact with the API endpoint to get features for the query's search set
    action = ["search-sets", "features"]
    params = {"id": search_set_id}
    features = client.action(schema, action, params=params)

    # create dictionary
    target_dict = {}
    for stream in streams:
        target_dict[stream] = {}
        for split in splits:
            target_dict[stream][split] = {}

    for tf in features:
        stream_type = tf["dnn_stream_id"]
        fsplit = tf["dnn_stream_split"]
        feature_vector = tf["feature_vector"]
        name = tf["name"]
        nclip = tf["video_clip_id"]
        if stream_type in streams and name == feature_name and fsplit in splits:
            target_dict[stream_type][fsplit][nclip] = feature_vector
    return target_dict
