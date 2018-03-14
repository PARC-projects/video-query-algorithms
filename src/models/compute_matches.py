"""
Public API to algorithms logic chain
"""


def new_matches(new_query_state,
                search_set='all',
                streams=('rgb', 'warped_optical_flow'),
                feature_name='global_pool',
                clip_duration=10):
    """
    Public contract to compute matches of a new query.

    Args:
        new_query_state:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "reference_time": query["reference_time"]
            }
    """
    print(new_query_state["query_id"])

    # TODO: Frank - Fedrate logic down to compute_similarities.py

    return


def revised_matches(revised_query_state,
                    search_set='all',
                    streams=('rgb', 'warped_optical_flow'),
                    feature_name='global_pool',
                    clip_duration=10):
    """
    Public contract to compute matches of a revised query.

    Args:
        revised_query_state:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "reference_time": query["reference_time"],
                "result": result (https://github.com/fetorres/video-query-api/blob/master/src/queries/models/query_result.py),
                "matches": matches (https://github.com/fetorres/video-query-api/blob/master/src/queries/models/match.py)
            }
    """
    print(revised_query_state["query_id"])

    # TODO: Frank - Fedrate logic down to compute_similarities.py

    return
