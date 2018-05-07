"""Make requests for Queries based on processing state
"""
import requests
import logging
from api.authenticate import authenticate

class QueryStatus():
    API_URL = "http://127.0.0.1:8000/" # TODO: Get from env
    logger = logging.getLogger(__name__)
    headers = {}

    def getStatus(self):
        """Request queries that meet processing_state requirements
        """
        try:
            # Get and set authentication headers
            self.headers['Authorization'] = authenticate(self.API_URL)
            return {
                'revision': self._getStatusComputeSimilarity(),
                'new': self._getStatusNewComputeSimilarity()
            }
        except Exception as e:
            logging.error(e)

    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            self.API_URL + "query-state/compute-revised")
        return response

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            self.API_URL + "query-state/compute-new")
        return response

    def _makeRequest(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == requests.codes['ok']:
            return response.json()
        else:
            return {}
