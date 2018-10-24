from requests import ConnectionError
import numpy as np
from time import sleep
import random
import logging


class TargetClip:
    def __init__(self, ticket, hyperparameters):
        """
        :param ticket: ticket instance of Ticket class
        :param hyperparameters: instance of class Hyperparameter, hyperparameters for deep learning ensemble
        """
        self.client = ticket.client
        self.schema = ticket.schema
        self.bootstrap_target = ticket.dynamic_target_adjustment
        self.tuning_update = ticket.tuning_update
        self.hyperparameters = hyperparameters
        self.ref_clip_features, self.splits = self._get_clip_features(ticket.ref_clip_id)
        self.target_features = ticket.tuning_update["bootstrapped_target"]

    def get_target_features(self):
        """
        Method to compute dictionary of target features for the current state of query with id = ticket.query_id.
        Depending on dynamic target adjustment setting, either choose ref clip as the target or adjust the target
        based on all user confirmed matches.

        Output: self.target_features dictionary, of the form { <stream type>: {<split #>:[<feature>], ...} }
                self.splits = splits present within self.target_features
        """
        if not self.bootstrap_target or self.tuning_update is None:
            self.target_features = self.scaled_ref_clip_features()
        else:
            # Load features for confirmed matches into a list of feature dictionaries: [<features dictionary 1>, ...]
            # Also return all splits present in the feature dictionaries
            features_4_matches, splits_4_matches = self.features_for_matches(user_match_value=True)

            # Compute the new target using dynamic target adjustment, unless there are no confirmed matches
            if features_4_matches:
                self.target_features = self.dynamic_target_adjustment(features_4_matches, splits_4_matches)
            else:
                self.target_features = self.scaled_ref_clip_features()

    def avg_new_old_targets(self, new_target, splits):
        for stream in self.hyperparameters.streams:
            for split in splits:
                new_target[stream][split] = self.hyperparameters.f_memory * new_target[stream][split] + \
                                            (1 - self.hyperparameters.f_memory) * self.target_features[stream][split]
        return new_target

    def dynamic_target_adjustment(self, list_of_feature_dictionaries, splits):
        """
        :param list_of_feature_dictionaries: Clip features dictionaries with
                                             entries { <stream type>: {<split #>:[<feature>], ...} }
        :param splits: splits for which dynamic target adjustment will be done
        returns dictionary of new target features in the format { <stream type>: {<split #>:[<target feature>], ...} }
        """
        # Check for user_match=False data points, and use if available. Otherwise, use only user_match=True data.
        features_invalid_matches, splits_invalid_matches = self.features_for_matches(user_match_value=False)
        if features_invalid_matches:
            new_target = self._bootstrap_valid_plus_invalid(list_of_feature_dictionaries, features_invalid_matches,
                                                            splits)
        else:
            new_target = self._bootstrap_valid_matches(list_of_feature_dictionaries, splits)
        # compute weighted average of new and old target, weighted by f_memory
        return self.avg_new_old_targets(new_target, splits)

    def features_for_matches(self, user_match_value=True):
        """
        General logic:
            get all matches for the query_result corresponding to the ticket used to initialize this Target instance
            for each match with user_match=user_match_value:
                add the feature dictionary for that match to the matches_features list
                update splits encountered as needed

        :param user_match_value: True or False depending on which features are required
        :return: list of features for all matches where user_match=user_match_value, and all splits encountered
        """
        # Interact with the API endpoint to get matches for the query
        page = 1
        matches = []
        while page is not None:
            action = ["matches", "list"]
            params = {"query_result": self.tuning_update["id"], "page": page}
            results = self._request(action, params)
            matches.extend(results["results"])
            page = results["pagination"]["nextPage"]
        # Load features for matches where user_match equals user_match_value into a list of feature dictionaries:
        # [<features dictionary 1>, ...]
        matches_features = []
        splits_matches = set()
        for match in matches:
            if match["user_match"] is user_match_value:
                match_features, match_splits = self._get_clip_features(match["video_clip"])
                matches_features.append(match_features)
                # splits_matches is a set, so elements are added only if not already in splits
                splits_matches.update(match_splits)
        return matches_features, splits_matches

    def scaled_ref_clip_features(self):
        ref_features = {}
        for stream, split_features in self.ref_clip_features.items():
            ref_features[stream] = {}
            for split, feature in split_features.items():
                ref_features[stream][split] = self._scale_feature(feature)
        return ref_features

    def _bootstrap_valid_matches(self, list_of_feature_dictionaries, splits):
        """
        Compute a new target for the features in list_of_features_dictionaries, presuming each one to be
        for a user_match=True video clip.  Randomly select half the features for forming the new target

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

        # select half of the matching clips, at random
        list_of_feature_dictionaries = self._random_fraction(list_of_feature_dictionaries)

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

        # select half of the matching clips and half of invalid match clips, at random
        list_valid_feature_dictionaries = self._random_fraction(list_valid_feature_dictionaries)
        list_invalid_feature_dictionaries = self._random_fraction(list_invalid_feature_dictionaries)

        # Extract features for each match for each (stream, split) duo, and store as a list of features
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
        feature_dictionaries_as_a_list = self._request(action, params)

        # fill in the feature dictionary for the clip with feature vectors from API feature_dictionaries_as_a_list
        for feature_object in feature_dictionaries_as_a_list:
            stream_type = feature_object["dnn_stream_id"]  # this is actually the stream name, not the primary key
            name = feature_object["name"]
            if stream_type in self.hyperparameters.streams and name == self.hyperparameters.feature_name:
                fsplit = feature_object["dnn_stream_split"]
                splits.add(fsplit)  # splits is a set, so only new splits get added
                results[stream_type][fsplit] = feature_object["feature_vector"]
        return results, splits

    def _request(self, action, params):
        while True:
            try:
                return self.client.action(self.schema, action, params=params)
            except ConnectionError:
                sleep(0.05)
                msg = 'Try API request by Target again: action = {}, params = {}'.format(action, params)
                logging.warning(msg)

    def _random_fraction(self, flist):
        # select a random list of items from flist, with fraction set by self.hyperparameters.f_bootstrap
        nmatches = len(flist)
        tmatches = round(nmatches * self.hyperparameters.f_bootstrap)
        tmatches = max(tmatches, 1)  # make sure at least one item is selected
        tsamples = random.sample(range(nmatches), tmatches)
        return [flist[m] for m in tsamples]

    @staticmethod
    def _scale_feature(f):
        return f / np.dot(f, f)
