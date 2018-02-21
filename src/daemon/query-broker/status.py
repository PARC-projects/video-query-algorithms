"""Make requests for Queries based on processing state
"""
import requests
import logging
import json
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8000/"


class QueryStatus():
    """
    Request queries that meet processing_state requirements
    """

    def getStatus(self):
        """
        Request queries that meet processing_state requirements

        Returns:
            {
                compute_similarity: <Query>,
                optimize_weight: <Query>,
                compute_new_matches: <Query>
            }
        """
        self._getStatusComputeSimilarity()
        self._getStatusOptimize()
        self._getStatusNewComputeSimilarity()

    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            BASE_URL + "query-state/compute-similarity")
        logger.info(response['documentation_url'])

    def _getStatusOptimize(self):
        response = self._makeRequest(BASE_URL + "query-state/optimize")
        logger.info(response['documentation_url'])

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            BASE_URL + "query-state/new-compute-similarity")
        logger.info(response['message'])

    def _makeRequest(self, url):
        response = requests.get('https://github.com/timeline.json')
        return response.json()
