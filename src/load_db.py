import psycopg2
import os
import csv

parent_dir = '/Volumes/torres/TSN/SHRP2Videos/SHRP2_Forward_clips_features/' # mounted tilde drive on Mac
# database connection
conn = psycopg2.connect("host=localhost dbname=video-query user=torres")  #creates Connection object
cur = conn.cursor()   #  creates Cursor object for issuing commands

# iterate through video clips
error_expression = 'video directory name {} and video name {} in file do not match'
for video in os.scandir(parent_dir):
    if video.is_dir() and not video.name.startswith('.'):
        video_path = os.path.join(parent_dir,video.name)
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
                            assert video_name == video.name , error_expression.format(video.name, video_name)
                            video_uri = header[1].split('=')[-1]
                            cnn_stream = header[2].split('=')[-1]
                            cnn_stream_split = split.name[-1]
                            feature_name = header[3].split('=')[-1]
                            cnn_weights_file_uri = header[4].split('=')[-1]
                            clip_duration = 10
                            for row in reader:
                                clip = row[0]
                                feature_vector = [float(x) for x in row[1:]]
                                cur.execute(
                                    "INSERT INTO video_clips (video, clip, clip_duration, video_uri) "
                                    "VALUES (%s, %s, %s, %s) "
                                    "ON CONFLICT ON CONSTRAINT no_duplicate_clips DO NOTHING;",
                                    [video_name, clip, clip_duration, video_uri]
                                )
                                conn.commit()
                                cur.execute(
                                    "SELECT id FROM video_clips "
                                    "WHERE video = %s AND clip = %s AND clip_duration = %s;",
                                    [video_name, clip, clip_duration]
                                )
                                clip_id = cur.fetchone()
                                cur.execute(
                                    "INSERT INTO features (video_clip_id, cnn_stream, cnn_stream_split, "
                                    "cnn_weights_file_uri, feature_name, feature) "
                                    "VALUES (%s, %s, %s, %s, %s, %s);",
                                    [clip_id, cnn_stream, cnn_stream_split,
                                     cnn_weights_file_uri, feature_name, feature_vector]
                                )
                                conn.commit()
    conn.commit()  # commit to database
# Close communication with database
cur.close()
conn.close()
