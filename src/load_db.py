"""
Loads features created by calcSig_wof into the video-queries-api database.
The features are in csv files in a directory tree specified by calcSig_wOF.py.
"""
from api.authenticate import authenticate
import coreapi
import os
import argparse
import csv


def create_or_get_video(video_name, video_path, duration, client, schema):

    # check to see if video already exists, and create it if needed
    action = ["videos", "list"]
    params = {
        "name": video_name,
        "path": video_path
    }
    response = client.action(schema, action, params=params)
    if response["results"]:
        assert len(response["results"]) == 1
        action = ["videos", "read"]
        params = {"id": response["results"][0]["id"]}
        video_object = client.action(schema, action, params=params)
    else:
        action = ["videos", "create"]
        params = {
            "name": video_name,
            "path": video_path,
        }
        video_object = client.action(schema, action, params=params)

    # create clips and features
    with os.scandir(video_path) as split_dir:
        for split in split_dir:
            if split.is_dir() and not split.name.startswith('.'):
                create_video_clips_and_features(split.path, video_object, duration, client, schema)


def create_video_clips_and_features(split_path, video_object, duration, client, schema):
    for csv_file in os.scandir(split_path):
        nsplit = int(split_path[-1])

        if csv_file.is_file() and csv_file.name.endswith('.csv') and not csv_file.name.startswith('.'):
            with open(csv_file.path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                video_name = header[0].split('=')[-1]
                assert video_name == video_object["name"]
                video_uri = header[1].split('=')[-1]
                dnn_stream = header[2].split('=')[-1]
                feature_name = header[3].split('=')[-1]
                dnn_weights_file_uri = header[4].split('=')[-1]

                for row in reader:
                    clip = int(row[0])
                    feature_vector = [float(x) for x in row[1:]]
                    clip_id = _create_or_get_clip(clip, duration, video_object, client, schema)
                    _create_feature(feature_vector, nsplit, feature_name, dnn_weights_file_uri,
                                    clip_id, dnn_stream, client, schema)


def _create_or_get_clip(clip, duration, video_object, client, schema):

    # check to see if video clip already exists, and create it if needed
    action = ["video-clips", "list"]
    params = {
        "video__name": video_object["name"],
        "clip": clip,
        "duration": duration,
    }
    response = client.action(schema, action, params=params)
    if response["results"]:
        assert len(response["results"]) == 1
        action = ["video-clips", "read"]
        params = {"id": response["results"][0]["id"]}
        clip_object = client.action(schema, action, params=params)
    else:
        action = ["video-clips", "create"]
        params = {
            "clip": clip,
            "duration": duration,
            "debug_video_uri": video_object["path"],
            "video": video_object["id"],
        }
        clip_object = client.action(schema, action, params=params)

    return clip_object["id"]


def _create_feature(feature_vector, split, feature_name, dnn_weights_file_uri, clip_id, dnn_stream, client, schema):

    # check to see if feature already exists, and create it if needed
    action = ["features", "list"]
    params = {
        "video_clip": clip_id,
        "dnn_stream": dnn_stream,
        "dnn_stream_split": split,
    }
    response = client.action(schema, action, params=params)
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
        client.action(schema, action, params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load video clip features into Video-Query database")
    parser.add_argument("src_dir", help="directory with video files")
    parser.add_argument("--duration", type=int, default=10, help="clip duration, s, integer only")
    parser.add_argument("--video_path_type", type=str, choices=['absolute', 'relative'], default='relative',
                        help='relative paths will have a parent specified in the video query api')
    args = parser.parse_args()

    # get authentication header
    auth_coreapi = authenticate()
    # Initialize a client & load the schema document
    client = coreapi.Client(auth=auth_coreapi)
    schema = client.get("http://127.0.0.1:8000/docs/")

    # iterate through video clips
    src_path = args.src_dir
    error_expression = 'video directory name {} and video name {} in file do not match'
    with os.scandir(src_path) as vid:
        for video in vid:
            if video.is_dir() and not video.name.startswith('.'):
                if args.video_path_type == 'absolute':
                    video_path = video.path
                else:
                    video_path = video.name
                create_or_get_video(video.name, video_path, args.duration, client, schema)
