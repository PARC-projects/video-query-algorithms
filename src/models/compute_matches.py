"""
Public API to algorithms logic chain
"""
from models.compute_similarities import compute_similarities, optimize_weights, select_matches, compute_score
import csv
from datetime import datetime
import os


def compute_matches(query_updater, api_url, default_weights, default_threshold, streams, near_miss_fraction):
    """
    Public contract to compute new matches and scores for a query, either new or revised.
    Creates a final report for a final revision of a query.
    query_updater is an instance of APIRepository, which is a class for interacting with the API

        update_ticket json object:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "tuning_update": for "revise" and "finalize" updates, QueryResult values for search tuning parameters
                                for most recent round of the query
                "matches": for "revise" updates, matches of most recent round of the query
                "user_matches": dictionary of {video_clip: user_match} entries from earlier rounds
                "dynamic_target_adjustment": dynamic_target_adjustment
            }
    """
    # Get info on any queries in the API repository that are waiting for an update
    updates_needed = query_updater.get_status()

    # update queries marked as "new", "revised" or "finalize" in the API database
    for update_type, update_ticket in updates_needed.items():
        if update_ticket is None:
            continue
        query_updater.change_process_state(update_ticket["query_id"], 3)  # Change process state to 3: in progress

        # Check for query errors.  Change process_state to 5 if there is an error in the query, and exit loop
        error_message, corrections = catch_errors(update_ticket)
        if error_message:
            query_updater.change_process_state(update_ticket["query_id"], 5, message=error_message)
            continue
        for k, v in corrections:  # modify update_ticket with any corrections suggested by catch_errors()
            update_ticket[k] = v

        # compute similarities with all clips in the search set
        similarities = compute_similarities(update_ticket, api_url, streams)

        # compute weights, threshold, and scores for matches
        if (update_type == "new") or not update_ticket["matches"]:
            weights = default_weights
            threshold = default_threshold
        elif update_type == "revise" or update_type == "finalize":
            update_matches = {}
            for match in update_ticket["matches"]:
                if match["user_match"]:
                    update_matches[match['video_clip']] = match["user_match"]  # For user_match == True or False
                else:
                    update_matches[match['video_clip']] = match["is_match"]   # For clips the user did not evaluate
            weights, threshold = optimize_weights(similarities, update_matches, streams)
        else:
            raise Exception('update type is invalid')

        # pack new information into a new query_result database entity
        if update_type == 'new':
            new_round = 1
        else:
            new_round = update_ticket["tuning_update"]["round"] + 1
        new_result_id = query_updater.create_query_result(update_ticket["query_id"], new_round, threshold, weights,
                                                          streams)

        # compute scores and matches (for the next round or final report)
        scores = compute_score(similarities, weights)
        if update_type == "finalize":
            max_number_matches = float("inf")  # add all matches to final report
            near_miss = 0  # do not add any near misses to final report
        else:
            max_number_matches = update_ticket["number_of_matches_to_review"]
            near_miss = near_miss_fraction
        matches = select_matches(scores, update_ticket, threshold, max_number_matches, near_miss)

        # catch errors that results in no matches being returned
        if not matches:
            catch_no_matches_error(update_ticket, query_updater)
            continue

        # add new match entities to database
        add_matches_to_database(matches, update_ticket, new_result_id, query_updater)

        # Create a final report if update_type = "finalize" and change process_state to 7: Finalized
        # TODO: Add email notification to user
        if update_type == "finalize":
            create_final_report(matches, update_ticket, query_updater, streams)
            query_updater.change_process_state(update_ticket["query_id"], 7)
            continue

        # Change process_state to 4: Processed
        query_updater.change_process_state(update_ticket["query_id"], 4)


def catch_errors(ticket):
    error_message = None
    update = {}
    if ticket["ref_clip_id"] is None:
        error_message = "*** Error: A video clip corresponding to the reference time does not exist " \
                    "in the database. ***"
    elif ticket["dynamic_target_adjustment"] is True and "matches" not in ticket:
        error_message = '*** Error: No similarities can be computed for query {}. Dynamic target adjustment is {} but there ' \
                        'are 0 matches computed for the previous round. Check database consistency for this query' \
            .format(ticket["query_id"], ticket["dynamic_target_adjustment"])
    elif ticket["dynamic_target_adjustment"] is True:
        good_count = 0
        for match in ticket["matches"]:
            if match["user_match"]:
                good_count += 1
        if good_count == 0:  # user did not validate any matches, but we can recover from this error
            update = {"dynamic_target_adjustment": False}
    return error_message, update


def catch_no_matches_error(ticket, query_updater):
    mround = ticket["tuning_update"]["round"] if "tuning_update" in ticket else 1
    error_message = "*** Error: No matches were found for round {} of query {}! ***".format(mround, ticket["query_id"])
    query_updater.change_process_state(ticket["query_id"], 5, message=error_message)
    return


def add_matches_to_database(matches, ticket, new_result_id, query_updater):
    if "user_matches" in ticket:
        user_matches = ticket["user_matches"]  # use user_match settings if any exist
    else:
        user_matches = {}
    for video_clip, score in matches.items():
        user_match = user_matches.get(str(video_clip))
        query_updater.create_match(new_result_id, score, user_match, video_clip)


def create_final_report(matches, ticket, query_updater, streams):
    # create final report that contains scores of all matches
    file_name = 'final_report_query_{}_{}.csv'.format(ticket["query_id"], datetime.now().strftime('%m-%d-%Y_%Hh%Mm%Ss'))
    file = os.path.join('../final_reports/', file_name)

    # Interact with the API endpoint to get query, video, query rounds, and search set info
    action = ["queries", "read"]
    params = {"id": ticket["query_id"]}
    query = query_updater.client.action(query_updater.schema, action, params=params)

    action = ["videos", "read"]
    params = {"id": ticket["video_id"]}
    video = query_updater.client.action(query_updater.schema, action, params=params)

    last_round = get_last_round(ticket, query_updater)

    action = ["search-sets", "read"]
    params = {"id": query["search_set_to_query"]}
    search_set = query_updater.client.action(query_updater.schema, action, params=params)

    matches_by_user = {}
    if "user_matches" in ticket:
        matches_by_user = ticket["user_matches"]

    # write the csv file
    with open(file, 'x', newline='') as csvfile:
        reportwriter = csv.writer(csvfile)
        # header information
        reportwriter.writerow(['Query:', query["name"], 'Query pk:', ticket["query_id"]])
        reportwriter.writerow(['Video:', video["name"], 'Video pk:', ticket["video_id"]])
        reportwriter.writerow(['Search Set queried:', search_set["name"], 'Search set pk:', search_set["id"]])
        reportwriter.writerow(['last round:', last_round["round"]])
        reportwriter.writerow(['min score for a match:', last_round["match_criterion"]])
        reportwriter.writerow(['streams:', str(streams)])
        reportwriter.writerow(['stream weights:', str(last_round["weights"])])
        reportwriter.writerow([''])
        # write out a row for each video clip that is a match
        reportwriter.writerow(['Algorithm matches, user-identified matches, and user-identified non-matches'])
        reportwriter.writerow(['clip #', 'match type', 'video pk', 'video clip id', 'score', 'duration', 'notes'])
        for video_clip_id, score in matches.items():
            type = "Algorithm match"
            if str(video_clip_id) in matches_by_user:
                if matches_by_user[str(video_clip_id)]:
                    type = "user-identified match"
                else:
                    type = "user-identified non-match"
            action = ["video-clips", "read"]
            params = {"id": video_clip_id}
            video_clip = query_updater.client.action(query_updater.schema, action, params=params)
            reportwriter.writerow([video_clip['clip'], type, video_clip['video'], video_clip_id, score,
                                   video_clip['duration'], video_clip['notes']])

    with open(file, 'r') as csvfile:
        # write final report to API
        action = ["queries", "partial_update"]
        params = {"id": ticket["query_id"], "final_report_file": csvfile}
        query_updater.client.action(query_updater.schema, action, params=params, encoding='multipart/form-data')


def get_last_round(ticket, query_updater):
    page = 1
    last_round = {"round": 0}
    while page is not None:
        action = ["query-results", "list"]
        params = {"query": ticket["query_id"], "page": page}
        query_results = query_updater.client.action(query_updater.schema, action, params=params)
        for round_object in query_results["results"]:
            if round_object["round"] > last_round["round"]:
                last_round = round_object
        page = query_results["pagination"]["nextPage"]
    return last_round
