"""
Public API to algorithms logic chain
"""
import requests
from models.compute_similarities import compute_similarities, optimize_weights, select_matches


def compute_matches(query_update, api_url, default_weights=('rgb': 1.0, 'warped_optical_flow': 1.5),
                    default_threshold=0.8, streams=('rgb', 'warped optical flow')):
    """
    Public contract to compute new matches and scores for a query.
    query_update is an instance of APIRepository

        query_to_update json object:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "result": for "revise" updates, QueryResult values for previous round
                "matches": for "revise" updates, matches of previous round
                "number_of_matches_to_review": number_of_matches
                "current_round": current_round
            }
    """
    updates = query_update.get_status()
    client = query_update.client
    schema = query_update.schema


    # Change process_state to 3: Processing
    # For any 'new' and 'revise' queries to be updated
    change_process_state(updates, 3, client, schema)

    # update queries (update type is "new" or "revise")
    for update_type, query_to_update in updates.items():

        similarities = compute_similarities(query_to_update, api_url, streams)

        # determine weights, threshold, and scores
        if update_type == "revise":
            # load matches that user has inspected
            user_matches = {}
            for match in query_to_update["matches"]:
                if match.user_match is not None:
                    user_matches[match.video_clip] = match.user_match
            scores_optimized, weights, threshold = optimize_weights(similarities, user_matches, streams)
        elif update_type == "new":
            weights = default_weights
            threshold = default_threshold
        else:
            # TODO:  Create some reasonable error message and action
            return "Error"

        # load new matches into db, in a new query_result
        matches = select_matches(similarities, weights, threshold,
                                 query_to_update["number_of_matches_to_review"])
        new_round = query_to_update["current_round"] + 1
        api_weights = []
        for k, stream in enumerate(streams):
            api_weights[k] = weights[stream]
        new_result_id = query_update.create_query_result(query_to_update["query_id"], new_round, threshold, api_weights)
        for video_clip, score in matches:
            query_update.create_match(new_result_id, score, video_clip)

    # Change process_state to 4: Processed
    # For any 'new' and 'revise' queries that were updated
    # TODO: Add email notification to user
    change_process_state(updates, 4, client, schema)


def change_process_state(updates, process_state, client, schema):
    for update_type, query_to_update in updates.items():
        action = ["queries", "partial_update"]
        params = {"id": query_to_update["query_id"], "process_state": process_state}
        response = client.action(schema, action, params=params)

        if response.status_code != requests.codes.ok:
            return "Update of process_state to " + process_state + \
                   " for query " + query_to_update["query_id"] + " failed!"
        return response.json()["id"]
