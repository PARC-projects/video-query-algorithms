import requests
import os
import coreapi


def authenticate(api_url="http://127.0.0.1:8000/"):
    # Request a token
    response = requests.post(
        os.path.join(api_url, 'api-token-auth/'),
        data={
            'username': os.environ['API_CLIENT_USERNAME'],
            'password': os.environ['API_CLIENT_PASSWORD']
        })

    if response.status_code != requests.codes.ok:
        print("Authentication failed: ")
        return

    # create a coreapi TokenAuthentication instance with auth.scheme and auth.token
    auth = coreapi.auth.TokenAuthentication(
        scheme='Token',
        token=response.json()['token']
        )
    return auth
