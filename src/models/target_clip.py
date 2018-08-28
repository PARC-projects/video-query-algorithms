import numpy as np
from models.target_bootstrapper import target_bootstrapper


class TargetClip:
    def __init__(self, ticket, hyperparameters):
        """
        :param ticket: ticket instance of Ticket class
        :param hyperparameters: instance of class Hyperparameter, hyperparameters for deep learning ensemble
        """
        self.client = ticket.client
        self.schema = ticket.schema
        self.ticket = ticket
        self.hyperparameters = hyperparameters
        self.ref_clip_features, self.splits = self._get_clip_features(ticket.ref_clip_id)
        self.target_features = {}
        self.client = self.ticket.client
        self.schema = self.ticket.schema

    def compute_target_features(self):
        """
        Method to compute dictionary of target features for the current state of query with id = ticket.query_id.
        Depending on dynamic target adjustment setting, either choose ref clip as the target or adjust the target
        based on all user confirmed matches.

        Output: self.target_features dictionary, of the form { <stream type>: {<split #>:[<feature>], ...} }
                self.splits = splits present within self.target_features
        """
        if not self.ticket.dynamic_target_adjustment:
            self.target_features = self.scaled_ref_clip_features()
        else:
            # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
            # Also return all splits present in the feature dictionaries
            features_4_matches, splits_4_matches = self.features_of_confirmed_matches()

            # Compute the new target using dynamic target adjustment, unless there are no confirmed matches
            if features_4_matches:
                self.target_features = self.dynamic_target_adjustment(features_4_matches, splits_4_matches)
            else:
                self.target_features = self.scaled_ref_clip_features()

    def scaled_ref_clip_features(self):
        ref_features = {}
        for stream, split_features in self.ref_clip_features.items():
            ref_features[stream] = {}
            for split, feature in split_features.items():
                ref_features[stream][split] = self._scale_feature(feature)
        return ref_features

    def features_of_confirmed_matches(self):
        # Interact with the API endpoint to get confirmed matches for the query
        page = 1
        confirmed_matches = []
        while page is not None:
            action = ["matches", "list"]
            params = {"query_result__query": self.ticket.query_id,
                      "page": page
                      }
            results = self.client.action(self.schema, action, params=params)
            confirmed_matches.extend(results["results"])
            page = results["pagination"]["nextPage"]

        # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
        # Make sure to not load features from the same video clip more than once
        confirmed_matches_features = []
        splits_confirmed_matches = set()
        video_clips_seen = []
        for match in confirmed_matches:
            if match["user_match"] and match["video_clip"] not in video_clips_seen:
                match_features, match_splits = self._get_clip_features(match["video_clip"])
                confirmed_matches_features.append(match_features)
                # splits_confirmed_matches is a set, so elements are added only if not already in splits
                splits_confirmed_matches.update(match_splits)
            video_clips_seen.append(match["video_clip"])

        return confirmed_matches_features, splits_confirmed_matches

    def dynamic_target_adjustment(self, list_of_feature_dictionaries, splits):
        """
        :param list_of_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        :param splits: splits for which dynamic target adjustment will be done
        returns dictionary of new target features in the format { <stream type>: {<split #>:[<target feature>], ...} }
        """
        return target_bootstrapper(list_of_feature_dictionaries, splits, self.hyperparameters.streams)

    def _get_clip_features(self, clip_id):
        """
        :param clip_id: primary key of the video clup
        :return: Clip features dictionaries with entries { <stream type>: {<split #>:[<feature>], ...} }
        """
        results = {}
        splits = set()
        for stream_type in self.hyperparameters.streams:
            results[stream_type] = {}

        # Interact with the API endpoint to get features for the video clip
        action = ["video-clips", "features"]
        params = {"id": clip_id}
        feature_dictionaries_as_a_list = self.client.action(self.schema, action, params=params)

        # fill in the feature dictionary for the clip with feature vectors from API feature_dictionaries_as_a_list
        for feature_object in feature_dictionaries_as_a_list:
            stream_type = feature_object["dnn_stream_id"]  # this is actually the stream name, not the primary key
            name = feature_object["name"]
            if stream_type in self.hyperparameters.streams and name == self.hyperparameters.feature_name:
                fsplit = feature_object["dnn_stream_split"]
                splits.add(fsplit)  # splits is a set, so only new splits get added
                results[stream_type][fsplit] = feature_object["feature_vector"]
        return results, splits

    @staticmethod
    def _scale_feature(f):
        return f / np.dot(f, f)
