"""
Loads features created by calcSig_wof into the video-queries-api database.
The features are in csv files in a directory tree specified by calcSig_wOF.py.
"""
from api.api_load_records import APILoadRecords
import os
import argparse


def main(args):
    loader = APILoadRecords(args.base_url)

    # load features, clips and videos by iterating through feature csv files stored in specified directory tree:
    # <source directory>/<video names>/<split names>/<csv files titled <<stream>>_<<feature name>>_features.csv >
    src_path = args.src_dir
    with os.scandir(src_path) as vid:
        for video in vid:
            if video.is_dir() and not video.name.startswith('.'):
                if args.video_path_type == 'absolute':
                    video_path = video.path
                else:
                    video_path = video.name
                # TODO: add mp4 or avi to video name
                video_object = loader.create_or_get_video(video.name, video_path)
                with os.scandir(video.path) as split_dir:
                    for split in split_dir:
                        if split.is_dir() and not split.name.startswith('.'):
                            loader.create_video_clips_and_features(video_object, split.path, args.duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load video clip features into Video-Query database")
    parser.add_argument("src_dir", help="directory with video files")
    parser.add_argument("--duration", type=int, default=10, help="clip duration, s, integer only")
    parser.add_argument("--video_path_type", type=str, choices=['absolute', 'relative'], default='relative',
                        help='relative paths will have a parent specified in the video query api')
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/",
                        help='url for video query api')
    arguments = parser.parse_args()

    main(arguments)
