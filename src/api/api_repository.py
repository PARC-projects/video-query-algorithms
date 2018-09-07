"""Make requests for Queries to process based on processing state
"""
import logging
from api.authenticate import authenticate
from requests import ConnectionError
import coreapi
import os
from time import sleep


class APIRepository:   # base_url is the api url.  The default is the dev default.
    def __init__(self, base_url="http://127.0.0.1:8000/"):
        self.logger = logging.getLogger(__name__)
        # Setup authenticated API client
        self.url = base_url
        self.client = coreapi.Client(auth=authenticate(self.url))
        try:
            self.schema = self.client.get(os.path.join(self.url, "docs"))
        except ConnectionError:
            sleep(0.05)
            msg = 'Try again to GET schema for APIRepository instance'
            logging.warning(msg)

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
            For 'revise' and 'finalize' queries:
                "tuning_update":  QueryResult record for latest round, with round number and
                                    match criterion and weights for tuning the search
                "matches": matches of previous round,
                            i.e. with query_results field equal to that of previous round
                "user_matches": dictionary of {video_clip: user_match} entries
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
