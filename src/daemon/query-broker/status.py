"""Make requests for Queries based on processing state
"""
import requests
import logging
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
        logging.debug('get')
        self._getStatusComputeSimilarity()
        self._getStatusOptimize()
        self._getStatusNewComputeSimilarity()

    def _getStatusComputeSimilarity(self):
        url = BASE_URL + "query-state/compute-similarity"
        logging.debug(url)

    def _getStatusOptimize(self):
        url = BASE_URL + "query-state/optimize"
        logging.debug(url)

    def _getStatusNewComputeSimilarity(self):
        url = BASE_URL + "query-state/new-compute-similarity"
        logging.debug(url)
