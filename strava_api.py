import datetime
from functools import wraps
import json
import os
from stravalib.client import Client
import time
from threading import Barrier
import _thread as thread
from flask import Flask, request
from gevent.pywsgi import WSGIServer
import numpy as np
from scipy.integrate import cumulative_trapezoid
from cache_to_disk import cache_to_disk
from matplotlib import pyplot as plt

CACHE_DAYS = 365 * 1000


STRAVA_CLIENT_ID = int(os.environ.get('STRAVA_CLIENT_ID', '131882'))
STRAVA_CLIENT_SECRET = os.environ['STRAVA_CLIENT_SECRET']

def requires_token(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if time.time() > self._client.token_expires_at:
            self._refresh_token()
        return fn(self, *args, **kwargs)
    return wrapper

class StravaAPI:
    _instance = None
    _auth_barrier = Barrier(2, timeout=6000)
    _code_response = None

    def __init__(self, token_path: str = os.path.join(os.environ['HOME'], '.strava_token')):

        self._client = Client()
        self._token_path = token_path
        self._refresh_token()

    @staticmethod
    def instance():
        if StravaAPI._instance is None:
            StravaAPI._instance = StravaAPI()
        return StravaAPI._instance

    @property
    @requires_token
    def client(self) -> Client:
        return self._client

    def _refresh_token(self):
        if os.path.exists(self._token_path):
            with open(self._token_path, 'r') as f:
                token_response = json.load(f)
        else:
            _launch_callback_server()
            authorize_url = self._client.authorization_url(
                client_id=STRAVA_CLIENT_ID, redirect_uri="http://localhost:8282/authorized"
            )
            # Have the user click the authorization URL, a 'code' param will be added to the redirect_uri
            # .....
            print('Log in at:', authorize_url)
            
            StravaAPI._auth_barrier.wait()
            code = StravaAPI._code_response

            token_response = self._client.exchange_code_for_token(
                client_id=STRAVA_CLIENT_ID, client_secret=STRAVA_CLIENT_SECRET, code=code
            )
            self._write_token_response_locally(token_response)
        self._set_token_response_on_client(token_response)
        if time.time() > token_response["expires_at"]:
            refresh_response = self._client.refresh_access_token(
                client_id=STRAVA_CLIENT_ID, client_secret=STRAVA_CLIENT_SECRET, refresh_token=self._client.refresh_token
            )
            self._write_token_response_locally(refresh_response)
            self._set_token_response_on_client(refresh_response)

    def _write_token_response_locally(self, token_response):
        with open(self._token_path, 'w') as f:
            json.dump(token_response, f)

    def _set_token_response_on_client(self, token_response):
        access_token = token_response["access_token"]
        refresh_token = token_response["refresh_token"]
        expires_at = token_response["expires_at"]

        # Now store that short-lived access token somewhere (a database?)
        self._client.access_token = access_token
        # You must also store the refresh token to be used later on to obtain another valid access token
        # in case the current is already expired
        self._client.refresh_token = refresh_token

        # An access_token is only valid for 6 hours, store expires_at somewhere and
        # check it before making an API call.
        self._client.token_expires_at = expires_at

_callback_server_launched = False

def _launch_callback_server():
    global _callback_server_launched
    if _callback_server_launched:
        return
    app = Flask(__name__)

    @app.route("/authorized")
    def _():
        StravaAPI._code_response = request.args.get('code')
        StravaAPI._auth_barrier.wait()
        return "OK"

    def start_server():
        http_server = WSGIServer(('', 8282), app)
        http_server.serve_forever()
        
    thread.start_new_thread(start_server, ())
    _callback_server_launched = True

def no_none(lst):
    return [x if x is not None else np.nan for x in lst]

@cache_to_disk(CACHE_DAYS)
def get_power_curve(min_time, max_time, intervals, activity_id):
    power_curve_times = np.logspace(min_time, max_time, intervals)
    streams = StravaAPI.instance().client.get_activity_streams(activity_id, types=['time', 'watts'])
    if 'watts' not in streams or 'time' not in streams:
        return [], []
    assert len(streams['watts'].data) == len(streams['time'].data)
    xp = np.array(no_none(streams['time'].data))
    dir_yp = np.array(no_none(streams['watts'].data))
    xp = xp[~np.isnan(dir_yp)]
    dir_yp = dir_yp[~np.isnan(dir_yp)]
    if len(xp) == 0:
        return [], []
    yp = cumulative_trapezoid(dir_yp, xp, initial=0)
    my_power_curve_times = []
    power_curve_powers = []
    for length in power_curve_times:
        xe = xp + length
        ye = np.interp(xe, xp, yp, right=np.nan)
        xe = xe[~np.isnan(ye)]
        yp_temp = yp[~np.isnan(ye)]
        ye = ye[~np.isnan(ye)]
        if len(xe) == 0:
            continue
        avg = (ye - yp_temp) / length
        my_power_curve_times.append(length)
        power_curve_powers.append(np.max(avg))
    return my_power_curve_times, power_curve_powers

def get_max_power_curve(min_time=0, max_time=np.log10(60 * 60 * 2), intervals=50, days=180):
    after = datetime.datetime.now() - datetime.timedelta(days=days)
    all_power_curve_times = []
    all_power_curve_powers = []
    activities = StravaAPI.instance().client.get_activities(after=after)
    for activity in activities:
        if activity.type == 'Ride':
            my_power_curve_times, power_curve_powers = get_power_curve(min_time, max_time, intervals, activity.id)
            all_power_curve_times.append(my_power_curve_times)
            all_power_curve_powers.append(power_curve_powers)
    
    # Get max length power_curve_times
    power_curve_times = max(all_power_curve_times, key=len)
    # Pad all power_curve_powers to the max time with nans on the right hand side, then stack on the 0th axis:
    all_power_curve_powers_array = np.vstack([np.pad(p, (0, len(power_curve_times) - len(p)), 'constant', constant_values=np.nan) for p in all_power_curve_powers])
    power_curve_powers = np.nanmax(all_power_curve_powers_array, axis=0)
    return power_curve_times, power_curve_powers

def plot_power_curve(power_curve_times, power_curve_powers):
    plt.plot(power_curve_times, power_curve_powers)
    plt.ylabel('Max Average Power (W)')
    plt.xlabel('Time (s)')
    plt.semilogx()

def quadrants(bounds):
    lat1, lon1, lat2, lon2 = bounds
    yield (lat1, lon1, lat1 + (lat2 - lat1) / 2, lon1 + (lon2 - lon1) / 2)
    yield (lat1 + (lat2 - lat1) / 2, lon1, lat2, lon1 + (lon2 - lon1) / 2)
    yield (lat1, lon1 + (lon2 - lon1) / 2, lat1 + (lat2 - lat1) / 2, lon2)
    yield (lat1 + (lat2 - lat1) / 2, lon1 + (lon2 - lon1) / 2, lat2, lon2)

@cache_to_disk(CACHE_DAYS)
def cached_leaf_find_segments_v4(bounds):
    segments = StravaAPI.instance().client.explore_segments(bounds=bounds, activity_type='riding', min_cat=1)
    found = list(segments)
    return found

def bound_area(bounds):
    lat1, lon1, lat2, lon2 = bounds
    return (lat2 - lat1) * (lon2 - lon1)

def recursive_find_segments(bounds, found_previously, initial_bound_area=None, leaves=64):
    if initial_bound_area is None:
        initial_bound_area = bound_area(bounds)
    found = cached_leaf_find_segments_v4(bounds)
    for segment in found:
        if segment.id not in found_previously:
            yield segment
            found_previously.add(segment.id)
    if found and len(found) > 1 and bound_area(bounds) >= initial_bound_area / leaves * 4 - 1e-12:
        for quad in quadrants(bounds):
            yield from recursive_find_segments(quad, found_previously, initial_bound_area=initial_bound_area)


if __name__ == '__main__':
    # power_curve_times, power_curve_powers = get_max_power_curve()
    # plot_power_curve(power_curve_times, power_curve_powers)
    # plt.savefig('power_curve.png')
    # bounds = [42.513076,-72.181450,43.618804,-70.774689]
    # for quad in quadrants(bounds):
    #     print(quad)
    # blah
    bounds = [44.209880, -68.438608, 44.486288, -67.930362]
    # bounds = [44.307772,-68.285002, 44.359806,-68.253086]
    segments = recursive_find_segments(bounds, set())
    segments = (seg for seg in segments if float('%f' % seg.elev_difference) > 75)
    for i, segment in enumerate(segments):
        # if i > 20:
        #     break
        print("%.1f %.1f %.0f %s" % (segment.avg_grade, segment.distance, segment.elev_difference, segment.name))