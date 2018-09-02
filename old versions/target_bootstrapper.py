import numpy as np


def bootstrap_valid_matches(list_of_feature_dictionaries, splits, streams):
    """
    Compute a new target for all features in list_of_features_dictionaries, presuming each one to be
    for a user_match=True video clip

    :param list_of_feature_dictionaries: Clip features dictionaries with
                                         entries { <stream type>: {<split #>:[<feature>], ...} }
    :param splits: splits for which target bootstrapping will be done
    :param streams: streams for which target bootstrapping will be done
    :return w = new target feature dictionary of the format { <stream type>: {<split #>:[<feature>], ...} }
    """
    # First initialize a dictionary for the features to be analyzed and for the new target
    features = {}
    new_target = {}
    for stream in streams:
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
    for stream_type in streams:
        for split in splits:
            X = np.asarray(features[stream_type][split]).T
            M = np.matmul(X.T, X)
            M_inv = np.linalg.inv(M)
            mu = np.sum(M_inv, axis=1).reshape([-1, 1])
            new_target[stream_type][split] = np.dot(X, mu).T.tolist()[0]

    return new_target


def bootstrap_valid_plus_invalid(list_valid_feature_dictionaries, list_invalid_feature_dictionaries, splits, streams):
    """
    Compute a new target, where features in list_valid_features_dictionaries are for user_match=True clips,
    and features in list_invalid_features_dictionaries are for user_match=False clips.

    :param list_valid_feature_dictionaries: Clip features dictionaries with
                                         entries { <stream type>: {<split #>:[<feature>], ...} }
     :param list_invalid_feature_dictionaries: Clip features dictionaries with
                                         entries { <stream type>: {<split #>:[<feature>], ...} }
    :param splits: splits for which target bootstrapping will be done
    :param streams: streams for which target bootstrapping will be done
    :return w = new target feature dictionary of the format { <stream type>: {<split #>:[<feature>], ...} }
    """
    # First initialize dictionaries for the features to be analyzed and for the new target
    xfeatures = {}
    yfeatures = {}
    new_target = {}
    for stream in streams:
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
    for stream_type in streams:
        for split in splits:
            X = np.asarray(xfeatures[stream_type][split])
            Y = np.asarray(yfeatures[stream_type][split])
            M = 2 * np.matmul(Y.T, Y)
            M_inv = np.linalg.inv(M)
            B = np.matmul(X, np.matmul(M_inv, X.T))
            B_inv = np.linalg.inv(B)
            w_1 = np.matmul(np.matmul(M_inv, X.T), B_inv)
            w_2 = M_inv - np.matmul(np.matmul(w_1, X), M_inv)
            w_3 = 2 * np.sum(np.matmul(w_2, Y.T), axis=1).reshape([-1, 1])
            w_final = w_3 + np.sum(w_1, axis=1).reshape([-1, 1])
            new_target[stream_type][split] = w_final.tolist()[0]

    return new_target
