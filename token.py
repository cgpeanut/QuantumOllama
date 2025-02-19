import requests
from requests.auth import HTTPBasicAuth

def obtain_dw_api_token(username: str, password: str) -> str:


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

    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return None

# Example usage (replace 'your_username' and 'your_password' with your actual credentials)
if __name__ == "__main__":
    username = "your_username"
    password = "your_password"
    api_token = obtain_dw_api_token(username, password)
    if api_token:
        print(f"Obtained API token: {api_token}")