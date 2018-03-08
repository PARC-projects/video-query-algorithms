"""
Public API to algorithms logic chain
"""

def new_matches(ref_clip_id, search_set='all', streams=('rgb', 'warped_optical_flow'),
                        feature_name='global_pool', clip_duration=10):
    """
        Conditions:
            :param ref_clip_id: primary key for reference clip in video_clips table.
            :return: match_indicator: {<video_clip_id>: <True or False>}  for all video clips in search set
    """
    print(ref_clip_id)

    # TODO: Fedrate logic down to compute_similarities.py

    return

def revised_matches(ref_clip_id, user_matches, search_set='all', streams=('rgb', 'warped_optical_flow'),
                        feature_name='global_pool', clip_duration=10):
    """
        Conditions:
            :param ref_clip_id: primary key for reference clip in video_clips table.
            :param user_matches: {<video clip id>: <0 or 1 to indicate whether user says it is a match>}
            :return: match_indicator: {<video_clip_id>: <True or False>} for all video clips in search set
    """
    print(ref_clip_id)

    # TODO: Fedrate logic down to compute_similarities.py

    return
