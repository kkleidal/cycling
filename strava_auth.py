import json
import os
from pathlib import Path
import socket
import threading
import time
import webbrowser

from flask import Flask, request
from stravalib.client import Client

SCOPE = ["read", "activity:read_all"]


def _find_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _load_token(token_cache):
    token_path = Path(token_cache)
    if token_path.exists():
        with token_path.open("r") as f:
            return json.load(f)
    return None


def _save_token(token_cache, token):
    token_path = Path(token_cache)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with token_path.open("w") as f:
        json.dump(token, f)


def _set_client_token(client, token):
    client.access_token = token["access_token"]
    client.refresh_token = token.get("refresh_token")
    client.token_expires_at = token.get("expires_at")


def _get_client_credentials(client_id_env, client_secret_env):
    client_id = os.environ.get(client_id_env)
    client_secret = os.environ.get(client_secret_env)
    if not client_id or not client_secret:
        raise RuntimeError(
            f"Missing Strava credentials. Set {client_id_env} and {client_secret_env}."
        )
    return int(client_id), client_secret


def _authorize_new_token(
    token_cache, client_id, client_secret, account_label="Strava", redirect_port=8000
):
    app = Flask(__name__)
    code_holder = {}
    done = threading.Event()

    @app.route("/authorized")
    def authorized():
        code_holder["code"] = request.args.get("code")
        done.set()
        return f"{account_label} authorization complete. You can close this window."

    thread = threading.Thread(
        target=lambda: app.run(
            port=redirect_port, host="127.0.0.1", debug=False, use_reloader=False
        ),
        daemon=True,
    )
    thread.start()

    client = Client()
    redirect_uri = f"http://127.0.0.1:{redirect_port}/authorized"
    auth_url = client.authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=SCOPE,
    )
    print(f"Opening browser for {account_label} Strava authorization...")
    webbrowser.open(auth_url)

    if not done.wait(timeout=600):
        raise RuntimeError(f"Timed out waiting for Strava authorization for {account_label}.")

    code = code_holder.get("code")
    if not code:
        raise RuntimeError(f"Strava authorization failed for {account_label}.")

    token = client.exchange_code_for_token(
        client_id=client_id,
        client_secret=client_secret,
        code=code,
    )
    _save_token(token_cache, token)
    return token


def get_authorized_client(
    token_cache="strava_token.json",
    client_id_env="STRAVA_CLIENT_ID",
    client_secret_env="STRAVA_CLIENT_SECRET",
    account_label="Strava",
    redirect_port=None,
):
    client_id, client_secret = _get_client_credentials(client_id_env, client_secret_env)
    client = Client()
    token = _load_token(token_cache)
    if redirect_port is None:
        redirect_port = _find_open_port()

    if token:
        _set_client_token(client, token)
        expires_at = token.get("expires_at")
        if expires_at is None or expires_at <= time.time():
            try:
                refreshed = client.refresh_access_token(
                    client_id=client_id,
                    client_secret=client_secret,
                    refresh_token=client.refresh_token,
                )
                _save_token(token_cache, refreshed)
                token = refreshed
            except Exception:
                token = _authorize_new_token(
                    token_cache,
                    client_id,
                    client_secret,
                    account_label=account_label,
                    redirect_port=redirect_port,
                )
    else:
        token = _authorize_new_token(
            token_cache,
            client_id,
            client_secret,
            account_label=account_label,
            redirect_port=redirect_port,
        )

    _set_client_token(client, token)
    return client


def authorize_strava(run_analytics, **kwargs):
    run_analytics(get_authorized_client(**kwargs))


if __name__ == "__main__":
    client = get_authorized_client()
    athlete = client.get_athlete()
    print(f"Authenticated as: {athlete.firstname} {athlete.lastname}")
