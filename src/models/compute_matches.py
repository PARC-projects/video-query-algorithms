"""
Public API to algorithms logic chain
"""
from models.compute_similarities import compute_similarities, optimize_weights, select_matches

ref_clip_id_error = "*** Error: A video clip corresponding to the reference time does not exist " \
                    "in the database. ***"


def compute_matches(query_updater, api_url, default_weights, default_threshold, streams):
    """
    Public contract to compute new matches and scores for a query.
    query_updater is an instance of APIRepository

        update_ticket json object:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "tuning_update": for "revise" updates, QueryResult values for search tuning parameters
                                for most recent round of the query
                "matches": for "revise" updates, matches of previous round
                "dynamic_target_adjustment": dynamic_target_adjustment
            }
    """
    updates_needed = query_updater.get_status()

    # update matches for the "new" and "revised" queries in updates
    for update_type, update_ticket in updates_needed.items():
        if update_ticket is None:
            continue

        # Change process_state to 3: Processing, or 5: Error if there is no video clip for the query reference time
        if update_ticket["ref_clip_id"] is None:
            query_updater.change_process_state(update_ticket["query_id"], 5)
            query_updater.add_note(update_ticket["query_id"], ref_clip_id_error)
            continue
        query_updater.change_process_state(update_ticket["query_id"], 3)

        # compute similarities with all clips in the search set
        similarities = compute_similarities(update_ticket, api_url, streams)

        # Error catching, e.g. if there were no matches for the query but dynamic target adjustment was chosen
        if not similarities:
            if "matches" in update_ticket:
                n_matches = len(update_ticket["matches"])
            else:
                n_matches = 0
            error_message = '*** No similarities computed. Dynamic target adjustment is {} and there are {}' \
                            ' matches for the previous round. Check database consistency for this query' \
                .format(update_ticket["dynamic_target_adjustment"], n_matches)
            query_updater.change_process_state(update_ticket["query_id"], 5)
            query_updater.add_note(update_ticket["query_id"], error_message)
            raise Exception('No similarities computed for query {}. Number of matches = {} '
                            'and dynamic target adjustment = {}'
                            .format(update_ticket["query_id"], n_matches, update_ticket["dynamic_target_adjustment"])
                            )

        # determine weights, threshold, and scores for matches
        if update_type == "revise" and update_ticket["matches"]:
            update_matches = {}
            for match in update_ticket["matches"]:
                if match["user_match"] is None:
                    update_matches[match['video_clip']] = match["is_match"]
                else:
                    update_matches[match['video_clip']] = match["user_match"]
            scores_optimized, weights, threshold = optimize_weights(similarities, update_matches, streams)
        elif update_type == "new" or (update_type == "revise" and not update_ticket["matches"]):
            weights = default_weights
            threshold = default_threshold
        else:
            print("error")
            # TODO:  Create some reasonable error message and action
            return "Error"

        # create a new query_result for the next round
        matches = select_matches(similarities, weights, threshold, update_ticket["number_of_matches_to_review"])
        new_round = 1
        if update_type == "revise":
            new_round = update_ticket["tuning_update"]["round"] + new_round
        api_weights = []
        for k, stream in enumerate(streams):
            api_weights.append(weights[stream])
        new_result_id = query_updater.create_query_result(update_ticket["query_id"], new_round, threshold,
                                                          api_weights)

        # create matches for the next round
        if matches:
            for video_clip, score in matches.items():
                query_updater.create_match(new_result_id, score, None, video_clip)
        else:
            # TODO: fix error here or make it process state 5
            for match in update_ticket["matches"]:
                query_updater.create_match(new_result_id, match["score"], None, match["video_clip"])

        # Change process_state to 4: Processed
        # TODO: Add email notification to user
        query_updater.change_process_state(update_ticket["query_id"], 4)
