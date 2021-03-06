"""
Public API to algorithms logic chain
"""
from models import Ticket, TargetClip
import os


def compute_matches(query_updates, hyperparameters):
    """
    Public contract to compute new matches and scores for a query, either new or revised.
    Creates a final report for a final revision of a query.
    query_updates is an instance of APIRepository, which is a class for interacting with the API
    to see if new algorithm tasks need to be performed.

    hyperparameters: for deep learning computations, instance of Hyperparameter class

    General logic:
        check if there are queries to update
        for each query needing updating:
            create a ticket
            check ticket for errors
            create a target (initially the reference clip) and get its features (compute if target bootstrapping)
            compute similarities of all clips in search set to the target
            for all but new queries:
                optimize hyperparameters
            put a new query result in the API database
            compute scores
            create new set of matches for review
            add new matches to API database
            for "finalize" query updates:
                create final report
    """
    # Get info on any queries in the API repository that are waiting for an update
    updates_needed = query_updates.get_status()

    # update queries marked as "new", "revised" or "finalize" in the API database
    for update_type, update_object in updates_needed.items():
        if update_object is None:
            continue
        # Create a Ticket instance for the algorithm task to be done, and
        # change process state to 3: in progress
        ticket = Ticket(update_object, query_updates.url)
        ticket.change_process_state(3)

        # Check for query errors.  Change process_state to 5 if there is an error in the query, and exit loop
        # Add a message in notes if there is recovery from an error
        fatal_error_message, error_message = ticket.catch_errors(update_type)
        if fatal_error_message:
            ticket.change_process_state(5, message=fatal_error_message)
            continue
        if error_message:
            ticket.add_note(error_message)

        # Get the feature dictionary for the target: { <stream type>: {<split #>: [<target feature>], ...} }
        ticket.target = TargetClip(ticket, hyperparameters)
        ticket.target.get_target_features()
        # compute similarities with all clips in the search set
        ticket.compute_similarities(hyperparameters)

        # for revise and finalize jobs, update weights and threshold based on current matches
        if (update_type == "new") or not update_object["matches"]:
            hyperparameters.weights = hyperparameters.default_weights
            hyperparameters.threshold = hyperparameters.default_threshold
        elif update_type == "revise" or update_type == "finalize":
            hyperparameters.optimize_weights(ticket)
        else:
            raise Exception('update type is invalid')

        # pack new information into a new query_result database entity
        if update_type == 'new':
            new_round = 1
        else:
            new_round = ticket.latest_query_result["round"] + 1
        new_result_id = ticket.create_query_result(new_round, hyperparameters)

        # compute scores and determine new set of matches (for the next round or final report)
        ticket.compute_scores(hyperparameters.weights)
        if update_type == "finalize":
            max_number_matches = float("inf")  # add all matches to final report
            # near_miss = 0  # do not add any near misses to final report
            # add misses down to the lowest scoring user match, if its score is less than threshold
            low_score, __ = ticket.lowest_scoring_user_match()
            near_miss = max(hyperparameters.threshold - low_score, 0) / \
                max(1 - hyperparameters.threshold, float(os.environ["COMPUTE_EPS"]))
            # COMPUTE_EPS protects from divide by zero error if threshold happened to be very close to 1
        else:
            max_number_matches = ticket.number_of_matches_to_review
            near_miss = hyperparameters.near_miss_default
        ticket.select_clips_to_review(hyperparameters.threshold, max_number_matches, near_miss)

        # catch errors that results in no matches being returned
        if not ticket.matches:
            catch_no_matches_error(ticket)
            continue

        # add new match entities to database
        ticket.add_matches_to_database(new_result_id)

        # Create a final report if update_type = "finalize" and change process_state to 7: Finalized
        # Otherwise, Change process_state to 4: Processed (for all jobs that are not finalize jobs)
        # TODO: Add email notification to user
        if update_type == "finalize":
            ticket.create_final_report(hyperparameters, new_result_id)
            ticket.change_process_state(7)
            continue
        else:
            ticket.change_process_state(4)


def catch_no_matches_error(ticket):
    mround = ticket.latest_query_result["round"] if ticket.latest_query_result else 1
    error_message = "*** Error: No matches were found for round {} of query {}! ***".format(mround, ticket.query_id)
    ticket.change_process_state(5, message=error_message)
    return
