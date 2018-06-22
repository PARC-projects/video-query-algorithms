"""Make requests for Queries based on processing state
"""
import logging
from api.authenticate import authenticate
import coreapi
import os
import csv
from requests import ConnectionError
from time import sleep


class APILoadRecords:   # base_url is the api url.  The default is the dev default.

    def __init__(self, base_url="http://127.0.0.1:8000/"):
        self.logger = logging.getLogger(__name__)
        # Setup authenticated API client
        self.client = coreapi.Client(auth=authenticate(base_url))
        self.schema = self.client.get(os.path.join(base_url, "docs"))

    def create_or_get_video(self, video_name, video_path):
        # check to see if video already exists, and create it if needed
        action = ["videos", "list"]
        params = {
            "name": video_name,
            "path": video_path
        }
        response = self._request(action, params)
        if response["results"]:
            assert len(response["results"]) == 1
            action = ["videos", "read"]
            params = {"id": response["results"][0]["id"]}
        else:
            action = ["videos", "create"]
            params = {
                "name": video_name,
                "path": video_path,
            }
        video_object = self._request(action, params)
        return video_object

    def create_video_clips_and_features(self, video_object, split_path, duration):
        for csv_file in os.scandir(split_path):
            nsplit = int(split_path[-1])

            if csv_file.is_file() and csv_file.name.endswith('.csv') and not csv_file.name.startswith('.'):
                with open(csv_file.path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    video_name = header[0].split('=')[-1]
                    assert video_name == video_object["name"].split('.')[0]
                    # video_uri = header[1].split('=')[-1]
                    dnn_stream = header[2].split('=')[-1]
                    feature_name = header[3].split('=')[-1]
                    dnn_weights_file_uri = header[4].split('=')[-1]

                    for row in reader:
                        clip = int(row[0])
                        feature_vector = [float(x) for x in row[1:]]
                        clip_id = self._create_or_get_clip(clip, duration, video_object)
                        self._create_feature(feature_vector, nsplit, feature_name, dnn_weights_file_uri,
                                             clip_id, dnn_stream)

    def _create_or_get_clip(self, clip, duration, video_object):

        # check to see if video clip already exists, and create it if needed
        action = ["video-clips", "list"]
        params = {
            "video__name": video_object["name"],
            "clip": clip,
            "duration": duration,
        }
        response = self._request(action, params)

        # if the video clip exists, read it.  Otherwise, create it
        if response["results"]:
            assert len(response["results"]) == 1
            action = ["video-clips", "read"]
            params = {"id": response["results"][0]["id"]}
        else:
            action = ["video-clips", "create"]
            params = {
                "clip": clip,
                "duration": duration,
                "debug_video_uri": video_object["path"],
                "video": video_object["id"],
            }
        clip_object = self._request(action, params)
        return clip_object["id"]

    def _create_feature(self, feature_vector, split, feature_name, dnn_weights_file_uri, clip_id, dnn_stream):
        # check to see if feature already exists, and create it if needed
        action = ["features", "list"]
        params = {
            "video_clip": clip_id,
            "dnn_stream": dnn_stream,
            "dnn_stream_split": split,
        }
        response = self._request(action, params)

        # if the feature already exists, validate there is only one.  If it does not exist, create it
        if response["results"]:
            assert len(response["results"]) == 1
        else:
            action = ["features", "create"]
            params = {
                "dnn_stream_split": split,
                "name": feature_name,
                "dnn_weights_uri": dnn_weights_file_uri,
                "feature_vector": feature_vector,
                "video_clip": clip_id,
                "dnn_stream": dnn_stream,
            }
            self._request(action, params)

    def _request(self, action, params):
        while True:
            try:
                return self.client.action(self.schema, action, params=params)
            except ConnectionError:
                sleep(0.05)
                print('Try again: action = {}, params = {}'.format(action, params))
