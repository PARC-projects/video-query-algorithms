"""Make requests for Queries based on processing state
"""
import logging
from api.authenticate import authenticate
import coreapi
import os


class APIRepository:   # base_url is the api url.  The default is the dev default.
    def __init__(self, base_url="http://127.0.0.1:8000/"):
        self.logger = logging.getLogger(__name__)
        # Setup authenticated API client
        self.client = coreapi.Client(auth=authenticate(base_url))
        self.schema = self.client.get(os.path.join(base_url, "docs"))

    def get_status(self):
        """Request queries that meet processing_state requirements
        query response:
            {
                "query_id": query["id"],
                "video_id": query["video"],
                "ref_clip": reference clip number,
                "ref_clip_id": pk for the reference video clip,
                "search_set": search set id
                "number_of_matches_to_review": number_of_matches
                "dynamic_target_adjustment": True or False, dynamically adjust target features for each round
            For 'revise' queries:
                "tuning_update":  QueryResult record for latest round, with round number and
                                    match criterion and weights for tuning the search
                "matches": for "revise" updates, matches of previous round,
                            i.e. with query_results field equal to that of previous round
            }
        """
        try:
            return {
                'revise': self._get_query_ready_for_revision(),
                'new': self._get_query_ready_for_new_matches(),
                'finalize': self._get_query_ready_for_finalize()
            }
        except Exception as e:
            logging.error(e)

    def _get_query_ready_for_revision(self):
        action = ["query-state", "compute-revised", "list"]
        return self.client.action(self.schema, action)

    def _get_query_ready_for_new_matches(self):
        action = ["query-state", "compute-new", "list"]
        return self.client.action(self.schema, action)

    def _get_query_ready_for_finalize(self):
        action = ["query-state", "compute-finalize", "list"]
        return self.client.action(self.schema, action)

    def create_match(self, qresult, score, user_match, video_clip):
        action = ["matches", "create"]
        params = {
            "query_result": qresult,
            "score": score,
            "user_match": user_match,
            "video_clip": video_clip,
        }
        self.client.action(self.schema, action, params=params)

    def create_query_result(self, query, nround, match_criterion, weights):
        action = ["query-results", "create"]
        params = {
            "round": nround,
            "match_criterion": match_criterion,
            "weights": weights,
            "query": query,
        }
        result = self.client.action(self.schema, action, params=params)
        return result["id"]

    def change_process_state(self, query_id, process_state):
        action = ["queries", "partial_update"]
        params = {"id": query_id, "process_state": process_state}
        result = self.client.action(self.schema, action, params=params)
        return result["process_state"]

    def add_note(self, query_id, note):
        # Get current notes by interacting with API
        action = ["queries", "read"]
        params = {"id": query_id}
        result = self.client.action(self.schema, action, params=params)
        # add note to current notes
        new_notes = result["notes"] + '\n\n' + note
        # update query object with new notes
        action = ["queries", "partial_update"]
        params = {"id": query_id, "notes": new_notes}
        result = self.client.action(self.schema, action, params=params)
