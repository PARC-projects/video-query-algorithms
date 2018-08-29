import numpy as np


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
        if not self.ticket.dynamic_target_adjustment or self.ticket.tuning_update is None:
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

    def dynamic_target_adjustment(self, list_of_feature_dictionaries, splits):
        """
        :param list_of_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        :param splits: splits for which dynamic target adjustment will be done
        returns dictionary of new target features in the format { <stream type>: {<split #>:[<target feature>], ...} }
        """
        # Check for user_match=False data points, and use if available. Otherwise, use only user_match=True data.
        features_invalid_matches, splits_invalid_matches = self.features_of_invalidated_matches()
        if features_invalid_matches:
            new_target = self._bootstrap_valid_plus_invalid(list_of_feature_dictionaries, features_invalid_matches,
                                                            splits)
        else:
            new_target = self._bootstrap_valid_matches(list_of_feature_dictionaries, splits)
        return new_target

    def features_of_confirmed_matches(self):
        # Interact with the API endpoint to get confirmed matches for the query
        page = 1
        confirmed_matches = []
        while page is not None:
            action = ["matches", "list"]
            params = {"query_result": self.ticket.tuning_update["id"],
                      "page": page
                      }
            results = self.client.action(self.schema, action, params=params)
            confirmed_matches.extend(results["results"])
            page = results["pagination"]["nextPage"]

        # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
        confirmed_matches_features = []
        splits_confirmed_matches = set()
        for match in confirmed_matches:
            if match["user_match"]:
                match_features, match_splits = self._get_clip_features(match["video_clip"])
                confirmed_matches_features.append(match_features)
                # splits_confirmed_matches is a set, so elements are added only if not already in splits
                splits_confirmed_matches.update(match_splits)
        return confirmed_matches_features, splits_confirmed_matches

    def features_of_invalidated_matches(self):
        # Interact with the API endpoint to get invalidated matches for the query
        page = 1
        invalidated_matches = []
        while page is not None:
            action = ["matches", "list"]
            params = {"query_result": self.ticket.tuning_update["id"],
                      "page": page
                      }
            results = self.client.action(self.schema, action, params=params)
            invalidated_matches.extend(results["results"])
            page = results["pagination"]["nextPage"]

        # Load features for invalidated matches into a list of feature dictionaries: [<features dictionary 1>, ...]
        # Make sure to not load features from the same video clip more than once
        invalidated_matches_features = []
        splits_invalidated_matches = set()
        for match in invalidated_matches:
            if match["user_match"] is not True:
                match_features, match_splits = self._get_clip_features(match["video_clip"])
                invalidated_matches_features.append(match_features)
                # splits_confirmed_matches is a set, so elements are added only if not already in splits
                splits_invalidated_matches.update(match_splits)
        return invalidated_matches_features, splits_invalidated_matches

    def scaled_ref_clip_features(self):
        ref_features = {}
        for stream, split_features in self.ref_clip_features.items():
            ref_features[stream] = {}
            for split, feature in split_features.items():
                ref_features[stream][split] = self._scale_feature(feature)
        return ref_features

    def _bootstrap_valid_matches(self, list_of_feature_dictionaries, splits):
        """
        Compute a new target for all features in list_of_features_dictionaries, presuming each one to be
        for a user_match=True video clip

        :param list_of_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        :param splits: splits for which target bootstrapping will be done
        :return w = new target feature dictionary of the format { <stream type>: {<split #>:[<feature>], ...} }
        """
        # First initialize a dictionary for the features to be analyzed and for the new target
        features = {}
        new_target = {}
        for stream in self.hyperparameters.streams:
            features[stream] = {}
            new_target[stream] = {}
            for split in splits:
                features[stream][split] = []

        # Extract features from each match for each (stream, split) duo, and store as a list of features
        for feature_dictionary in list_of_feature_dictionaries:  # one feature_dictionary for each match
            for stream_type, split_features in feature_dictionary.items():
                for split, feature in split_features.items():
                    features[stream_type][split].append(feature)

        # Compute the new target feature
        # need to convert final 1xn numpy array to a simple list, as needed by the Target class
        for stream_type in self.hyperparameters.streams:
            for split in splits:
                X = np.asarray(features[stream_type][split]).T
                M = np.matmul(X.T, X)
                M_inv = np.linalg.inv(M)
                mu = np.sum(M_inv, axis=1).reshape([-1, 1])
                new_target[stream_type][split] = np.dot(X, mu).T.tolist()[0]

        return new_target

    def _bootstrap_valid_plus_invalid(self, list_valid_feature_dictionaries, list_invalid_feature_dictionaries, splits):
        """
        Compute a new target, where features in list_valid_features_dictionaries are for user_match=True clips,
        and features in list_invalid_features_dictionaries are for user_match=False clips.

        :param list_valid_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
         :param list_invalid_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        :param splits: splits for which target bootstrapping will be done
        :return w = new target feature dictionary of the format { <stream type>: {<split #>:[<feature>], ...} }
        """
        # First initialize dictionaries for the features to be analyzed and for the new target
        xfeatures = {}
        yfeatures = {}
        new_target = {}
        for stream in self.hyperparameters.streams:
            xfeatures[stream] = {}
            yfeatures[stream] = {}
            new_target[stream] = {}
            for split in splits:
                xfeatures[stream][split] = []
                yfeatures[stream][split] = []

        # Extract features from each match for each (stream, split) duo, and store as a list of features
        for feature_dictionary in list_valid_feature_dictionaries:  # one feature_dictionary for each match
            for stream_type, split_features in feature_dictionary.items():
                for split, feature in split_features.items():
                    xfeatures[stream_type][split].append(feature)

        # Do the same for each non-match for each (stream, split) duo, and store as a list of features
        for feature_dictionary in list_invalid_feature_dictionaries:  # one feature_dictionary for each match
            for stream_type, split_features in feature_dictionary.items():
                for split, feature in split_features.items():
                    yfeatures[stream_type][split].append(feature)

        # Compute the new target feature
        # need to convert final 1xn numpy array to a simple list, as needed by the Target class
        for stream_type in self.hyperparameters.streams:
            for split in splits:
                X = np.asarray(xfeatures[stream_type][split])
                Y = np.asarray(yfeatures[stream_type][split])
                tr_YYT = np.trace(np.matmul(Y, Y.T))
                scale = self.hyperparameters.mu / tr_YYT
                M = np.eye(Y.shape[1]) + scale * np.matmul(Y.T, Y)
                M_inv = np.linalg.inv(M)
                B = np.matmul(X, np.matmul(M_inv, X.T))
                B_inv = np.linalg.inv(B)
                w_1 = np.matmul(np.matmul(M_inv, X.T), B_inv)
                w_2 = M_inv - np.matmul(np.matmul(w_1, X), M_inv)
                w_3 = np.sum(np.matmul(w_2, scale * Y.T), axis=1).reshape([-1, 1])
                w_final = w_3 + np.sum(w_1, axis=1).reshape([-1, 1])
                new_target[stream_type][split] = w_final.T.tolist()[0]

        return new_target

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
