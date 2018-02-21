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

        return {
            'compute_similarity': self._getStatusComputeSimilarity(),
            'optimize_weight': self._getStatusOptimize(),
            'compute_new_matches': self._getStatusNewComputeSimilarity()
        }

    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            BASE_URL + "query-state/compute-similarity")
        return response

    def _getStatusOptimize(self):
        response = self._makeRequest(BASE_URL + "query-state/optimize")
        return response

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            BASE_URL + "query-state/new-compute-similarity")
        return response

    def _makeRequest(self, url):
        try:
            response = requests.get('https://github.com/timeline.json')
            if response.status_code == requests.codes.ok:
                return response.json()
            else:
                logger.info(response.status_code)
                return None
        except requests.exceptions.HTTPError as errh:
            logger.error("Http Error: " + errh)
        except requests.exceptions.ConnectionError as errc:
            logger.warning("Error Connecting: " + errc)
        except requests.exceptions.Timeout as errt:
            logger.warning("Timeout Error: " + errt)
        except requests.exceptions.RequestException as err:
            logger.warning("OOps: Something Else " + err)
