import requests
from requests.auth import HTTPBasicAuth

def obtain_dw_api_token(username: str, password: str) -> str:
    """
    Obtain an API token from D-Wave Leap.

    Parameters:
    - username (str): Your D-Wave Leap account username.
    - password (str): Your D-Wave Leap account password.

    Returns:
    - api_token (str): The obtained API token, or None if the request fails.
    """

    # Define the authentication endpoint URL
    auth_url = "https://cloud.dwavesys.com/leap/api/v1/auth/login"

    try:
        # Make a POST request to the login endpoint with your credentials
        response = requests.post(auth_url, auth=HTTPBasicAuth(username, password))

        # Check if the request was successful
        if response.status_code == 200:
            # Assuming the API token is returned in JSON format within 'token' key
            api_token_response = response.json()
            api_token = api_token_response.get('token')
            return api_token
        
        else:
            print(f"Failed to obtain API token. Status code: {response.status_code}")
            return None
        