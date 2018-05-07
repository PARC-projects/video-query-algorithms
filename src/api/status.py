"""Make requests for Queries based on processing state
"""
import requests
import logging
import json
import os


class QueryStatus():
    BASE_URL = "http://127.0.0.1:8000/"
    logger = logging.getLogger(__name__)
    headers = {}

    def getStatus(self):
        """Request queries that meet processing_state requirements
        """
        try:
            self._authenticate()
            return {
                'revision': self._getStatusComputeSimilarity(),
                'new': self._getStatusNewComputeSimilarity()
            }
        except Exception as e:
            logging.error(e)

    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/compute-revised")
        return response

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/compute-new")
        return response

    def _makeRequest(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == requests.codes.ok:
            return response.json()
        else:
            return {}

    def _authenticate(self):
        """Request a token and store for future use
        """
        # Make request
        response = requests.post(
            self.BASE_URL + 'api-token-auth/',
            data={
                'username': os.environ['API_CLIENT_USERNAME'],
                'password': os.environ['API_CLIENT_PASSWORD']
            })

        # Set header
        self.headers['Authorization'] = "Token {}".format(
            response.json()['token'])
