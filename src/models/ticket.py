"""Make requests for Queries based on processing state
"""
from api.authenticate import authenticate
from requests import ConnectionError
import coreapi
import os
import csv
from datetime import datetime, timedelta
import numpy as np
import random
from time import sleep
import logging
import json


class Ticket:   # base_url is the api url.  The default is the dev default.
    def __init__(self, update_object, api_url):
        """
        :param update_object:
        json object:
        {
            "query_id": query["id"],
            "video_id": query["video"],
            "ref_clip": reference clip number in the video (i.e. reference clip starts at ref_clip * clip duration)
            "ref_clip_id": pk for the reference video clip,
            "search_set": search set id
            "number_of_matches_to_review": number_of_matches
            "latest_query_result": for "revise" and "finalize" updates, QueryResult values for search tuning parameters
                             for most recent round of the query
            "matches": for "revise" updates, matches of most recent round of the query
            "user_matches": dictionary of {video_clip: user_match} entries from earlier rounds
            "dynamic_target_adjustment": dynamic_target_adjustment
        }
        :param api_url is the url for the Video Query API
        """
        self.client = coreapi.Client(auth=authenticate(api_url))
        self.schema = self.client.get(os.path.join(api_url, "docs"))
        self.query_id = update_object["query_id"]
        self.video_id = update_object["video_id"]
        self.ref_clip = update_object["ref_clip"]
        self.ref_clip_id = update_object["ref_clip_id"]
        self.search_set = update_object["search_set"]
        self.number_of_matches_to_review = update_object["number_of_matches_to_review"]
        self.dynamic_target_adjustment = update_object["dynamic_target_adjustment"]
        if "latest_query_result" in update_object:
            self.latest_query_result = update_object["latest_query_result"]
        else:
            self.latest_query_result = None
        if "matches" in update_object:
            self.matches = update_object["matches"]
        if "user_matches" in update_object:
            self.user_matches = update_object["user_matches"]
        else:
            self.user_matches = {}
        self.target = None
        self.similarities = {}
        self.scores = {}

    def add_matches_to_database(self, new_result_id):
        for video_clip, score in self.matches.items():
            user_match = self.user_matches.get(str(video_clip))
            self.create_match(new_result_id, score, user_match, video_clip)

    def add_note(self, note):
        # Get current notes by interacting with API
        action = ["queries", "read"]
        params = {"id": self.query_id}
        result = self._request(action, params)
        # add note to current notes
        if result["notes"]:
            new_notes = result["notes"] + '\n\n' + note
        else:
            # TODO: test that this works, or whether above should be changed to if "notes" in result
            new_notes = note
        # update query object with new notes
        action = ["queries", "partial_update"]
        params = {"id": self.query_id, "notes": new_notes}
        self._request(action, params)

    def catch_errors(self, job_type):
        # catch errors, create error messages, and create corrections if possible
        fatal_error_accumulator = []
        error_accumulator = []

        # Check for no ref clip, most likely because reference time is not in video
        if self.ref_clip_id is None:
            fatal_error_accumulator.append("*** Fatal Error: A video clip corresponding to the reference time does "
                                           "not exist in the database. ***")

        # Check for no matches on all but new jobs
        if job_type is not "new" and not self.matches:
            fatal_error_accumulator.append("*** Fatal Error: This is not a new query but there are 0 matches computed '\
                                  'for the previous round. Cannot update without matches. Check database consistency ' \
                                  'for this query")

        # On all but new jobs: Check for user matches if dynamic_target_adjustment is True.
        # Cannot adjust target without user matches to guide the adjustment.
        if job_type is not "new" and self.dynamic_target_adjustment is True:
            no_error = False
            for match in self.matches:
                if match["user_match"] is True:
                    no_error = True
                    break
            if no_error is False:
                error_accumulator.append('*** Error: Dynamic target adjustment is True but there are no user matches '
                                         'provided for the previous round. Changing dynamic target adjustment to False')
                self.dynamic_target_adjustment = False
        fatal_error_message = "\n".join(fatal_error_accumulator)
        error_message = "\n".join(error_accumulator)
        return fatal_error_message, error_message

    def change_process_state(self, process_state, message=None):
        action = ["queries", "partial_update"]
        params = {"id": self.query_id, "process_state": process_state}
        result = self._request(action, params)
        if message:
            self.add_note(message)
        return result["process_state"]

    def compute_similarities(self, hyperparameters):
        """
        Public contract to compute dictionary of averaged similarities for the current job ticket.
        Dictionary structure for similarities is
            { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }

        Hyperparameter dictionary: keys include "default_weights", "default_threshold", "near_miss_default": 0.5,
                                    "streams", "feature_name"

        General logic:
            get target features (initially the reference clip features, scaled by their squared L2 norm)
            get features for all candidate matches (i.e. all clips in search set)
            for each stream type:
                for each split:
                    for each candidate feature of this stream type and split:
                        compute dot product similarity
                        add to list of similarities for this clip and stream
                for each clip:
                    average the similarities in the list over all splits
        """
        # Get the feature dictionary for all video clips (in the search set of interest).
        # Dictionary structure is { <stream type>: {<split #>: { clip#: [<candidate feature>], ...} } }
        candidates = self._get_candidate_features(self.target.splits, hyperparameters)

        # compute similarities and ensemble average them over the splits
        avgd_similarities = {}  # type: dict
        for stream_type, all_splits in self.target.target_features.items():
            similarities = {}  # type: dict
            # compute dot product similarities{} for each split, saved as key:value = clip:array of similarities
            for split, target_feature in all_splits.items():
                for clip, candidate_feature in candidates[stream_type][split].items():
                    similarity = np.dot(target_feature, candidate_feature)
                    similarities[clip] = similarities.get(clip, []) + [similarity]

            # ensemble average over the splits for each id
            for clip_id, sim_array in similarities.items():
                id_len = len(sim_array)
                avg_sim_this_stream = sum(sim_array) / id_len
                # create dictionary item in avgd_similarities for clip_id if it does not exist, and add result:
                avgd_similarities[clip_id] = avgd_similarities.get(clip_id, {})
                avgd_similarities[clip_id].update({stream_type: [avg_sim_this_stream, id_len]})

        # update Ticket similarities
        self.similarities = avgd_similarities

    def compute_scores(self, weights):
        """
        Conditions:
            similarities: { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
            weights: {<stream_type>: <weight>}, e.g. {'rgb': 1.0, 'warped_optical_flow': 1.5}
            return: scores: {<video_clip_id>: score}  where <video_clip_id> is the primary key in the video_clips table
        """
        self.scores = {}
        for video_clip_id, vsim in self.similarities.items():
            ssum = 0
            denom = 0
            for stream_type, w in weights.items():
                ssum += (w * (1 - vsim[stream_type][0])) ** 2
                denom += w ** 2
            vscore = np.sqrt(ssum / denom)
            self.scores[video_clip_id] = 1 - vscore

    def create_final_report(self, hyperparameters, query_result_id):
        # Interact with the API endpoint to get query, video, query rounds, and search set info
        action = ["queries", "read"]
        params = {"id": self.query_id}
        query = self._request(action, params)

        action = ["videos", "read"]
        params = {"id": self.video_id}
        video = self._request(action, params)

        action = ["query-results", "read"]
        params = {"id": query_result_id}
        query_result = self._request(action, params)
        number_of_reviews = query_result["round"] - 1  # initial round is based on default weights

        action = ["search-sets", "read"]
        params = {"id": query["search_set_to_query"]}
        search_set = self._request(action, params)

    # create final report that contains scores of all matches
        file_name = 'final_report_query_{}_{}.csv'.format(query["name"], datetime.now().strftime('%m-%d-%Y_%Hh%Mm%Ss'))
        file = os.path.join('../final_reports/', file_name)
        if not os.path.exists('../final_reports/'):
            os.makedirs('../final_reports/')

        # write the csv file
        with open(file, 'x', newline='') as csvfile:
            reportwriter = csv.writer(csvfile)
            # header information
            reportwriter.writerow(['Query:', query["name"], 'Query pk:', self.query_id])
            reportwriter.writerow(['Search Set queried:', search_set["name"], 'Search set pk:', search_set["id"]])
            reportwriter.writerow(['Reference Video:', video["name"], 'Video pk:', self.video_id])
            reportwriter.writerow(['Reference time:', query["reference_time"]])
            reportwriter.writerow(['number of reviews:', number_of_reviews])
            reportwriter.writerow(['min score for a match:', query_result["match_criterion"]])
            reportwriter.writerow(["max matches to review:", query["max_matches_for_review"]])
            reportwriter.writerow(['streams:', str(hyperparameters.streams)])
            reportwriter.writerow(['stream weights:', str(query_result["weights"])])
            reportwriter.writerow(['Target bootstrapping:', query["use_dynamic_target_adjustment"]])
            reportwriter.writerow(['query notes:', query["notes"]])
            reportwriter.writerow(['Hyperparameters:'])
            reportwriter.writerow(['', 'default weights:', str(hyperparameters.default_weights)])
            reportwriter.writerow(['', 'default threshold:', str(hyperparameters.default_threshold)])
            reportwriter.writerow(['', 'near miss default:', str(hyperparameters.near_miss_default)])
            reportwriter.writerow(['', 'feature name:', str(hyperparameters.feature_name)])
            reportwriter.writerow(['', 'ballast:', str(hyperparameters.ballast)])
            reportwriter.writerow(['', 'mu:', str(hyperparameters.mu)])
            reportwriter.writerow(['', 'f_bootstrap:', str(hyperparameters.f_bootstrap)])
            reportwriter.writerow(['', 'f_memory:', str(hyperparameters.f_memory)])
            reportwriter.writerow(['', 'bootstrap type:', str(hyperparameters.bootstrap_type)])
            if hyperparameters.bootstrap_type == "bagging":
                reportwriter.writerow(['', 'number of bags:', str(hyperparameters.nbags)])
            reportwriter.writerow([''])
            # write out a row for each video clip that is a selected match
            reportwriter.writerow(['List of all clips with scores greater than min(threshold, score of lowest scoring'
                                   ' user validated match)'])
            reportwriter.writerow(['clip #', 'start time', 'match type', 'video pk', 'video clip id', 'score',
                                   'duration', 'notes'])
            clip_rows = []
            for video_clip_id, score in self.matches.items():
                # add a row for each match that is in the set of self.matches.
                # This set includes matches either above the threshold score or explicitly scored
                # by the user in this round, when compute_matches set
                # max_number_matches = float("inf")  and near_miss = 0 before selecting matches for finalization.
                if str(video_clip_id) in self.user_matches:
                    if self.user_matches[str(video_clip_id)] is True:
                        match_type = "user-identified match"
                    else:
                        match_type = "user-identified non-match"
                elif score >= query_result["match_criterion"]:
                    match_type = "inferred match"
                else:
                    match_type = "inferred non-match"

                action = ["video-clips", "read"]
                params = {"id": video_clip_id}
                video_clip = self._request(action, params)
                action = ["matches", "list"]
                params = {"query_result": query_result_id, "video_clip": video_clip_id}
                match = self._request(action, params)
                start_time = int(match["results"][0]["match_video_time_span"].split(",")[0])
                stime = str(timedelta(seconds=start_time))
                clip_rows.append([video_clip['clip'], stime, match_type, video_clip['video'], video_clip_id, score,
                                  video_clip['duration'], video_clip['notes']])
            clip_rows.sort(key=lambda x: x[5], reverse=True)
            for row in clip_rows:
                reportwriter.writerow(row)

        with open(file, 'r') as csvfile:
            # write final report to API
            action = ["queries", "partial_update"]
            params = {"id": self.query_id, "final_report_file": csvfile}
            self._post_file(action, params)

    def create_match(self, qresult, score, user_match, video_clip):
        action = ["matches", "create"]
        params = {
            "query_result": qresult,
            "score": score,
            "user_match": user_match,
            "video_clip": video_clip,
        }
        self._request(action, params)

    def create_query_result(self, nround, hyperparameters):
        # Make list out of dictionary of weights, in the order specified by streams
        weights_values = [hyperparameters.weights[stream] for k, stream in enumerate(hyperparameters.streams)]
        # Interact with API
        action = ["query-results", "create"]
        params = {
            "round": nround,
            "match_criterion": hyperparameters.threshold,
            "weights": weights_values,
            "query": self.query_id,
            "bootstrapped_target": json.dumps(self.target.target_features)
        }
        result = self._request(action, params)
        return result["id"]

    def lowest_scoring_user_match(self):
        min_score = 1
        min_clip = None
        for clip, score in self.scores.items():
            if str(clip) in self.user_matches:
                if self.user_matches[str(clip)] is True:
                    min_score = min(min_score, score)
                    min_clip = clip
        return min_score, min_clip

    def select_clips_to_review(self, threshold=0.8, max_number_matches=20, near_miss=0.5):
        """
        Find matches and near matches for review,
        half being above threshold and half for 1-(1+near_miss)*(1-threshold) < score < threshold.
        For review, matches are chosen randomly from all scores within the specified interval,
        with the total of matches and near matches being equal to or less than max_number_matches.

        Result is the dictionary self.matches with matches and near misses: {<video_clip_id>: score}

        Conditions:
        :param threshold: real threshold, either the initial default or a new threshold_optimum from optimize_weights
        :param max_number_matches:  max number of matches the user wants to review.
        :param near_miss:  range of scores for near misses relative to the range (1-threshold) for hits
        """
        lower_limit = threshold - near_miss * (1 - threshold)
        match_candidates = {k: v for k, v in self.scores.items() if v >= threshold}
        near_match_candidates = {k: v for k, v in self.scores.items() if lower_limit <= v < threshold}

        # randomly select to stay within user defined max number of matches to evaluate
        # Note: if the number of candidates is fewer than the user defined max, use all candidates
        mscores = min(max_number_matches / 2, len(match_candidates)).__int__()
        m_near_scores = min(max_number_matches - mscores, len(near_match_candidates)).__int__()
        match_scores = random.sample(match_candidates.items(), mscores)
        # hold back one slot for the near miss with highest score
        near_match_max = {}
        if m_near_scores > 0:
            m_near_scores = m_near_scores - 1
            near_match_max_key = max(near_match_candidates, key=lambda key: near_match_candidates[key])
            near_match_max = {near_match_max_key: self.scores[near_match_max_key]}
            near_match_candidates.pop(near_match_max_key)
        near_match_scores = random.sample(near_match_candidates.items(), m_near_scores)
        # create dictionary with the random sampling of matches and near matches
        self.matches = dict(match_scores + near_match_scores)
        self.matches.update(near_match_max)

        # make sure reference clip is included if it is in the search set for this ticket
        # Also add back in any video clips that were user validated matches in the previous round and not included yet
        if self.ref_clip_id in self.scores:
            previous_user_evals = {self.ref_clip_id: self.scores[self.ref_clip_id]}
        else:
            previous_user_evals = {}
        if self.user_matches:
            for clip, value in self.user_matches.items():
                if value is True:
                    previous_user_evals.update({int(clip): self.scores[int(clip)]})
        self.matches.update(previous_user_evals)

    def _get_candidate_features(self, splits, hyperparameters):
        # Create video clip feature dictionary with entries like
        # { <stream type>: {<split #>: { clip#: [<target feature>], ...} } }

        # Interact with the API endpoint to get features for the query's search set
        action = ["search-sets", "features"]
        params = {"id": self.search_set}
        features = self._request(action, params)

        # create blank multi-level dictionary for results
        candidate_dict = {}
        for stream in hyperparameters.streams:
            candidate_dict[stream] = {}
            for split in splits:
                candidate_dict[stream][split] = {}

        for tf in features:
            tf_stream = tf["dnn_stream_id"]
            fsplit = tf["dnn_stream_split"]
            feature_vector = tf["feature_vector"]
            name = tf["name"]
            nclip = tf["video_clip_id"]
            if tf_stream in hyperparameters.streams and name == hyperparameters.feature_name and fsplit in splits:
                candidate_dict[tf_stream][fsplit][nclip] = feature_vector
        return candidate_dict

    def _request(self, action, params):
        while True:
            try:
                return self.client.action(self.schema, action, params=params)
            except ConnectionError:
                sleep(0.05)
                msg = 'Try API request again: action = {}, params = {}'.format(action, params)
                logging.warning(msg)

    def _post_file(self, action, params):
        while True:
            try:
                return self.client.action(self.schema, action, params=params, encoding="multipart/form-data")
            except ConnectionError:
                sleep(0.05)
                msg = 'Try API file post by Ticket again: action = {}, params = {}'.format(action, params)
                logging.warning(msg)
