import requests
import os
import coreapi


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

    auth = coreapi.auth.TokenAuthentication(
        scheme='Token',
        token=response.json()['token'],
        domain=os.path.join(api_url,"docs")
    )
    # TODO: Consider bubbling up meaningful loggin message on != 200
    return auth
