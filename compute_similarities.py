import psycopg2
import numpy as np
"""
Input is the reference clip video and time
For now, assume clip duration of 10 sec.
Also assume warped optical flow and rgb are being used
"""

# call function with reference video, reference clip, feature name, searchset
def compute_similarities(ref_video, ref_clip, search_set='all', streams=['rgb', 'warped_optical_flow'],
                         feature_name='global_pool', clip_duration=10):
    # database connection
    conn = psycopg2.connect("host=localhost dbname=video-query user=torres")
    cur = conn.cursor()
    # query for the video_clip_id corresponding to the reference clip
    cur.execute(
        "SELECT id FROM video_clips "
        "WHERE video = %s AND clip = %s AND clip_duration = %s;",
        [ref_video, ref_clip, clip_duration]
    )
    ref_clip_id = cur.fetchall()
    assert len(ref_clip_id) == 1, "Error in database table, mroe than one reference clip found!"
    # query db for the feature of the reference video clip, for all streams and splits
    cur.execute(
        "SELECT cnn_stream, cnn_stream_split, feature FROM features "
        "WHERE video_clip_id = %s AND feature_name = %s;",
        [ref_clip_id[0], feature_name]
    )
    ref_features = cur.fetchall()
    # set up reference feature dictionary with entries like { <stream type>: {<split #>:[<feature>], ...} }
    ref_dict = {}
    for stream_type in streams:
        ref_dict[stream_type] = {}
    for feature_list in ref_features:
        stream_type = feature_list[0]
        fsplit = feature_list[1]
        feature_vector = feature_list[2]
        if stream_type in streams:
            ref_dict[stream_type][fsplit] = feature_vector

    # compute emsemble averaged similarities for each stream type in streams
    # LATER:  need to add constraint for the target video being in the search set
    avgd_similarities = {}
    for stream_type, all_splits in ref_dict.items():
        # compute similarities for each split
        # load into similarities dictionary with key:value entries of
        # id:[<split 1 similarity>, <split 2 similarity>, <split 3 similarity>], not necessarily in order
        similarities = {}
        for split, feature_vector in all_splits.items():
            # query the features db for each entry with a matching cnn_stream, cnn_stream_split & feature name
            cur.execute(
                "SELECT video_clip_id, feature FROM features "
                "WHERE cnn_stream = %s AND cnn_stream_split = %s AND feature_name = %s;",
                [stream_type, split, feature_name]
            )
            target_features = cur.fetchall()

            # compute dot product similarities for matching cnn_stream: key:value is id:[<similarity
            for target in target_features:
                similarity = np.dot(feature_vector, target[1]) / np.linalg.norm(feature_vector)**2
                similarities[target[0]] = similarities.get(target[0], []) + [similarity]

        # ensemble average over the splits for each id
        # dictionary structure is { video_clip_id: {stream_type: [<avg similarity>, <number of items in ensemble>]} }
        for clip_id, sim_array in similarities.items():
            id_len = len(sim_array)
            avg_sim = sum(sim_array) / id_len
            avgd_similarities[clip_id] = avgd_similarities.get(clip_id, {})
            avgd_similarities[clip_id].update({stream_type: [avg_sim, id_len]})

    return avgd_similarities

# call function with (rgb_emsemble_similarity, wOF_ensemble_similarity, weights)
# compute score for each video


if __name__ == '__main__':
    result = compute_similarities('S06NDS_Sample_120406_1451_00186_Forward', 4, '')
    # check with calculations I did manually