from stravalib.client import Client
import os

client = Client()
client.access_token = os.environ['STRAVA_ACCESS_TOKEN']
for segment in client.explore_segments(bounds=[37.812685, -122.532434, 37.865305, -122.477496], activity_type='riding'):
    print(segment.distance)