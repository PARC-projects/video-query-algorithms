"""Make requests for Queries based on processing state
"""
import requests
import logging
import json
import os


class QueryStatus():
    BASE_URL = "http://127.0.0.1:8000/"
    logger = logging.getLogger(__name__)
    headers = {'Authorization': ''}

    def getStatus(self):
        """
        Request queries that meet processing_state requirements
        Returns:
        """

        try:
            self._authenticate()
            return {
                'compute_similarity': self._getStatusComputeSimilarity(),
                'compute_new_matches': self._getStatusNewComputeSimilarity()
            }
        except requests.exceptions.HTTPError as errh:
            self.logger.error("Http Error: " + errh)
        except requests.exceptions.ConnectionError as errc:
            self.logger.warning("Error Connecting: {}".format(errc))
        except requests.exceptions.Timeout as errt:
            self.logger.warning("Timeout Error: " + errt)
        except requests.exceptions.RequestException as err:
            self.logger.warning("OOps: Something Else " + err)


    def _getStatusComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/compute-similarity")
        return response

    def _getStatusNewComputeSimilarity(self):
        response = self._makeRequest(
            self.BASE_URL + "query-state/new-compute-similarity")
        return response

    def _makeRequest(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code == requests.codes.ok:
            return response.json()
        else:
            return {}


    def _authenticate(self):
        """
        Request a token and set store for future use
        """
        # Make request
        response = requests.post(
            self.BASE_URL + 'api-token-auth/',
            data = {
                'username': os.environ['API_CLIENT_USERNAME'],
                'password': os.environ['API_CLIENT_PASSWORD']
            }
        )

        # Set header
        self.headers['Authorization'] = "Token {}".format(
                    response.json()['token'])




