"""
Public API to algorithms logic chain
"""
from models.compute_similarities import compute_similarities, optimize_weights, select_matches


def compute_matches(query_updater, api_url, default_weights, default_threshold, streams):
    """
    Public contract to compute new matches and scores for a query.
    query_updater is an instance of APIRepository

        query_to_update json object:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "tuing_update": QueryResult values for search tuning parameters for most recent
                                analysis of the query, including the current round
                "matches": for "revise" updates, matches of previous round
            }
    """
    updates = query_updater.get_status()
    client = query_updater.client
    schema = query_updater.schema

    # update queries (update type is "new" or "revise")
    for update_type, query_to_update in updates.items():
        if query_to_update is None:
            continue
        # Change process_state to 3: Processing
        new_state = change_process_state(query_to_update["query_id"], 3, client, schema)
        assert new_state == 3
        # Change process_state to 5: Error if there is not a reference video clip record for the query reference time
        if query_to_update["ref_clip_id"] is None:
            change_process_state(query_to_update["query_id"], 5, client, schema)
            continue

        # compute similarities with all clips in the search set
        similarities = compute_similarities(query_to_update, api_url, streams)

        # determine weights, threshold, and scores
        if update_type == "revise" and query_to_update["matches"]:
            # load matches
            update_matches = {}
            for match in query_to_update["matches"]:
                if match["user_match"] is not None:
                    update_matches[match['video_clip']] = match["user_match"]
                else:
                    update_matches[match['video_clip']] = match["is_match"]
            scores_optimized, weights, threshold = optimize_weights(similarities, update_matches, streams)
        elif update_type == "new" or (update_type == "revise" and not query_to_update["matches"]):
            weights = default_weights
            threshold = default_threshold
        else:
            print("error")
            # TODO:  Create some reasonable error message and action
            return "Error"

        # load new matches into db, in a new query_result
        matches = select_matches(similarities, weights, threshold,
                                 query_to_update["number_of_matches_to_review"])
        new_round = query_to_update["tuning_update"]["round"] + 1
        api_weights = []
        for k, stream in enumerate(streams):
            api_weights.append(weights[stream])
        new_result_id = query_updater.create_query_result(query_to_update["query_id"], new_round,
                                                          threshold, api_weights)
        if matches:
            for video_clip, score in matches.items():
                query_updater.create_match(new_result_id, score, None, video_clip)
        else:
            for match in query_to_update["matches"]:
                query_updater.create_match(new_result_id, match["score"], None, match["video_clip"])

        # Change process_state to 4: Processed
        # TODO: Add email notification to user
        change_process_state(query_to_update["query_id"], 4, client, schema)


def change_process_state(query_id, process_state, client, schema):
    action = ["queries", "partial_update"]
    params = {"id": query_id, "process_state": process_state}
    result = client.action(schema, action, params=params)
    return result["process_state"]
