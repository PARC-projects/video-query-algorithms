import psycopg2
import os
import csv

parent_dir = '/Volumes/torres/TSN/SHRP2Videos/SHRP2_Forward_clips_features/'  # mounted tilde drive on Mac
# database connection
conn = psycopg2.connect("host=localhost dbname=video-query user=torres")  # creates Connection object
cur = conn.cursor()   # creates Cursor object for issuing commands

# iterate through video clips
error_expression = 'video directory name {} and video name {} in file do not match'
for video in os.scandir(parent_dir):
    if video.is_dir() and not video.name.startswith('.'):
        video_path = os.path.join(parent_dir, video.name)
        for split in os.scandir(video_path):
            if split.is_dir() and not split.name.startswith('.'):
                split_path = os.path.join(video_path, split.name)
                for csv_file in os.scandir(split_path):
                    if csv_file.is_file() and csv_file.name.endswith('.csv') \
                            and not csv_file.name.startswith('.'):

                        feature_file = os.path.join(split_path, csv_file.name)
                        with open(feature_file, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            video_name = header[0].split('=')[-1]
                            video_uri = header[1].split('=')[-1]
                            cnn_stream = header[2].split('=')[-1]
                            cnn_stream_split = split.name[-1]
                            feature_name = header[3].split('=')[-1]
                            cnn_weights_file_uri = header[4].split('=')[-1]
                            clip_duration = 10
                            for row in reader:
                                clip = row[0]
                                feature_vector = [float(x) for x in row[1:]]
                                # add video name to video table if it is not already there
                                cur.execute(
                                    "INSERT INTO video (name) "
                                    "SELECT (%s) "
                                    "WHERE NOT EXISTS ("
                                        "SELECT id FROM video "
                                        "WHERE name = %s"
                                        ");",
                                    [video_name, video_name]
                                )
                                # get primary key id for this video
                                cur.execute(
                                    "SELECT id FROM video "
                                    "WHERE name = %s;",
                                    [video_name]
                                )
                                video_id = cur.fetchone()
                                # create entry in video_clips table for this clip, unless it already exists
                                cur.execute(
                                    "INSERT INTO video_clips (video_id, clip, duration, debug_video_uri) "
                                    "VALUES (%s, %s, %s, %s) "
                                    "ON CONFLICT ON CONSTRAINT no_duplicate_clips DO NOTHING;",
                                    [video_id, clip, clip_duration, video_uri]
                                )
                                conn.commit()
                                cur.execute(
                                    "SELECT id FROM video_clips "
                                    "WHERE video_id = %s AND clip = %s AND duration = %s;",
                                    [video_id, clip, clip_duration]
                                )
                                clip_id = cur.fetchone()
                                # create row in features table
                                cur.execute(
                                    "INSERT INTO features (video_clip, dnn_stream, dnn_stream_split, "
                                    "dnn_weights_uri, name, feature_vector) "
                                    "VALUES (%s, %s, %s, %s, %s, %s);",
                                    [clip_id, cnn_stream, cnn_stream_split,
                                     cnn_weights_file_uri, feature_name, feature_vector]
                                )
                                conn.commit()
    conn.commit()  # commit to database
# Close communication with database
cur.close()
conn.close()
