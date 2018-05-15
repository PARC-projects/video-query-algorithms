import requests
import os


def authenticate(api_url="http://127.0.0.1/"):
    """Request a token and store for future use
    """
    # Make request
    response = requests.post(
        api_url + 'api-token-auth/',
        data={
            'username': os.environ['API_CLIENT_USERNAME'],
            'password': os.environ['API_CLIENT_PASSWORD']
        })

    # TODO: Consider bubbling up meaningful loggin message on != 200
    return 'Token {}'.format(response.json()['token'])

