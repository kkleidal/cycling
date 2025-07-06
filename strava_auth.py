import os
import json
import webbrowser
from flask import Flask, redirect, request, url_for, session
from stravalib.client import Client
import time

# Configuration
CLIENT_ID = os.environ.get('STRAVA_CLIENT_ID', 'YOUR_CLIENT_ID')
CLIENT_SECRET = os.environ.get('STRAVA_CLIENT_SECRET', 'YOUR_CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:8000/authorized'
TOKEN_CACHE = 'strava_token.json'

app = Flask(__name__)
app.secret_key = os.urandom(24)

client = Client()

def save_token(token):
    with open(TOKEN_CACHE, 'w') as f:
        json.dump(token, f)

def load_token():
    if os.path.exists(TOKEN_CACHE):
        with open(TOKEN_CACHE, 'r') as f:
            return json.load(f)
    return None

def run_analytics(client):
    # Example: Print athlete profile
    athlete = client.get_athlete()
    print(f"Authenticated as: {athlete.firstname} {athlete.lastname}")
    # Add custom analytics here
    return f"Analytics complete. Authenticated as: {athlete.firstname} {athlete.lastname}. Check your terminal for details."

SCOPE = ['read', 'activity:read_all']

@app.route('/')
def index():
    token = load_token()
    if token:
        client.access_token = token['access_token']
        client.refresh_token = token.get('refresh_token')
        client.token_expires_at = token.get('expires_at')
        return '<a href="/run_analytics">Run Analytics</a> | <a href="/reauth">Re-authenticate with Strava</a>'
    else:
        auth_url = client.authorization_url(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            scope=SCOPE
        )
        return f'<a href="{auth_url}">Authenticate with Strava</a>'

@app.route('/authorized')
def authorized():
    code = request.args.get('code')
    if not code:
        return 'Authorization failed.'
    token_response = client.exchange_code_for_token(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        code=code
    )
    save_token(token_response)
    client.access_token = token_response['access_token']
    client.refresh_token = token_response.get('refresh_token')
    client.token_expires_at = token_response.get('expires_at')
    return redirect(url_for('index'))

@app.route('/run_analytics')
def run_analytics_route():
    token = load_token()
    if not token:
        return redirect(url_for('index'))
    client.access_token = token['access_token']
    client.refresh_token = token.get('refresh_token')
    client.token_expires_at = token.get('expires_at')
    result = run_analytics(client)
    return result

@app.route('/reauth')
def reauth():
    if os.path.exists(TOKEN_CACHE):
        os.remove(TOKEN_CACHE)
    auth_url = client.authorization_url(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE
    )
    return redirect(auth_url)

def authorize_strava(run_analytics=run_analytics):
    token = load_token()
    if token and token.get('expires_at') > time.time():
        client.access_token = token['access_token']
        client.refresh_token = token.get('refresh_token')
        client.token_expires_at = token.get('expires_at')
        run_analytics(client)
    else:
        print('Opening browser for Strava authentication and analytics...')
        webbrowser.open('http://localhost:8000/')
        app.run(port=8000, debug=False)

if __name__ == '__main__':
    authorize_strava()
