import numpy as np


def target_bootstrapper(list_of_feature_dictionaries, splits, streams):
    """
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
