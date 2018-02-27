"""Make requests for Queries based on processing state
"""
import requests
import logging
import json
import os


class QueryStatus():
    BASE_URL = "http://127.0.0.1:8000/"
    logger = logging.getLogger(__name__)

    def getStatus(self):
        """
        Request queries that meet processing_state requirements
        Returns:
        """
        self._authenticate()

        return {
            'compute_similarity': self._getStatusComputeSimilarity(),
            'optimize_weight': self._getStatusOptimize(),
            'compute_new_matches': self._getStatusNewComputeSimilarity()
        }

    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/compute-similarity")
        return response

    def _getStatusOptimize(self):
        response = self._makeRequest(self.BASE_URL + "query-state/optimize")
        return response

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/new-compute-similarity")
        return response

    def _makeRequest(self, url):
        try:
            response = requests.get(url)
            print(response)
            if response.status_code == requests.codes.ok:
                return response.json()
            else:
                return {}
        except requests.exceptions.HTTPError as errh:
            self.logger.error("Http Error: " + errh)
        except requests.exceptions.ConnectionError as errc:
            self.logger.warning("Error Connecting: " + errc)
        except requests.exceptions.Timeout as errt:
            self.logger.warning("Timeout Error: " + errt)
        except requests.exceptions.RequestException as err:
            self.logger.warning("OOps: Something Else " + err)

    def _authenticate(self):
        """
        Request a token and set store for future use
        """
        request = {
            'username': os.environ['API_CLIENT_USERNAME'],
            'password': os.environ['API_CLIENT_PASSWORD']
        }
        response = requests.post(
            self.BASE_URL + 'api-token-auth/', data=request)
        print(response.json())
