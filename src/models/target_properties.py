import numpy as np


class TargetProperties:
    def __init__(self, ticket, client, schema, streams, feature_name):
        """
        :param ticket: ticket for computing target features
        :param client: API client for coreapi calls
        :param schema: API schema for coreapi calls
        :param streams: names of distinct DNN streams to include
        :param feature_name: name of embedded DNN feature for the instance of this class
        ticket:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "tuning_update": QueryResult values for search tuning parameters for most recent
                                analysis of the query, including the current round
                "matches": for "revise" updates, matches of previous round
                "dynamic_target_adjustment": dynamic_target_adjustment
            }
        """
        self.ticket = ticket
        self.client = client
        self.schema = schema
        self.streams = streams
        self.feature_name = feature_name
        self.ref_clip_features, self.splits = self._get_clip_features(self.ticket["ref_clip_id"])
        self.target_features = {}

    def compute_target_features(self):
        """
        Method to compute dictionary of target features for the current state of query with id = ticket["query_id"].
        Depending on dynamic target adjustment setting, either choose ref clip as the target or adjust the target
        based on all user confirmed matches.

        Output: self.target_features, also of the form { <stream type>: {<split #>:[<feature>], ...} }
                self.splits = splits present within self.target_features
        """
        if not self.ticket["dynamic_target_adjustment"]:
            self.target_features = self.scaled_ref_clip_features()
        else:
            # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
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
        action = ["matches", "list"]
        params = {"query_result__query": self.ticket["query_id"], "user_match": True}
        confirmed_matches = self.client.action(self.schema, action, params=params)

        # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
        features_confirmed_matches = []
        splits_confirmed_matches = set()
        for match in confirmed_matches["results"]:
            match_features, match_splits = self._get_clip_features(match["video_clip"])
            features_confirmed_matches.append(match_features)
            splits_confirmed_matches.update(match_splits)  # a set, so elements are added only if not already in splits

        return features_confirmed_matches, splits_confirmed_matches

    def dynamic_target_adjustment(self, list_of_feature_dictionaries, splits):
        """
        :param list_of_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        Procedure:
        for each stream:
            for each split:
                average features for all confirmed matches;
                for each x_i in averaged_feature:
                    if x_i < ref_clip_feature_i:
                        target_i = x_i
                    else:
                        target_i = ref_feature_i
                add averaged_feature to target_features
        """
        # First initialize a dictionary for the averaged features and the number of features
        features_avgd = {}
        features_number = {}
        target_scaled = {}
        for stream in self.streams:
            features_avgd[stream] = {}
            features_number[stream] = {}
            target_scaled[stream] = {}
            for splt in splits:
                features_avgd[stream][splt] = np.array(0)
                features_number[stream][splt] = 0

        # add all features for each stream:split, and count how many features of each type
        for feature_dictionary in list_of_feature_dictionaries:  # one feature_dictionary for each match
            for stream_type, split_features in feature_dictionary.items():
                for split, feature in split_features.items():
                    features_avgd[stream_type][split] = features_avgd[stream_type][split] + np.asarray(feature)
                    features_number[stream_type][split] += 1

        # divide by number of features of each type to get the average, and compare with ref clip feature.
        # Compute target_feature as the element-wise minimum of the ref clip feature and features_avgd
        # scale target_feature before returning it
        for stream in self.streams:
            for splt in self.splits:
                features_avgd[stream][splt] = features_avgd[stream][splt] / features_number[stream][splt]
                new_target = np.minimum(features_avgd[stream][splt], self.ref_clip_features)
                target_scaled[stream][splt] = self._scale_feature(new_target)

        return target_scaled

    def _get_clip_features(self, clip_id):
        """
        :param clip_id: primary key of the video clup
        :return: Clip features dictionaries with entries { <stream type>: {<split #>:[<feature>], ...} }
        """
        results = {}
        splits = set()
        for stream_type in self.streams:
            results[stream_type] = {}

        # Interact with the API endpoint to get features for the video clip
        action = ["video-clips", "features"]
        params = {"id": clip_id}
        feature_dictionaries_as_a_list = self.client.action(self.schema, action, params=params)

        # fill in the feature dictionary for the clip with feature vectors from API feature_dictionaries_as_a_list
        for feature_object in feature_dictionaries_as_a_list:
            stream_type = feature_object["dnn_stream_id"]  # this is actually the stream name, not the primary key
            name = feature_object["name"]
            if stream_type in self.streams and name == self.feature_name:
                fsplit = feature_object["dnn_stream_split"]
                splits.add(fsplit)  # splits is a set, so only new splits get added
                results[stream_type][fsplit] = feature_object["feature_vector"]
        return results, splits

    def _scale_feature(self, f):
        return f / np.dot(f, f)
