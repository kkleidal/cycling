from collections import Counter, defaultdict
import googlemaps
import os
from datetime import datetime, timedelta
from cache_to_disk import cache_to_disk
from sklearn.metrics.pairwise import haversine_distances
import numpy as np
from OSMPythonTools.overpass import Overpass


CACHE_DAYS = 365


def get_home_address():
    with open(os.path.join(os.environ['HOME'], '.home-address'), 'r') as f:
        return f.read().strip()

def get_api_key():
    with open(os.path.join(os.environ['HOME'], '.cycling-api-key'), 'r') as f:
        return f.read().strip()


@cache_to_disk(CACHE_DAYS)
def time_to_travel(destination, origin=None):
    if origin is None:
        origin = get_home_address()
    now = (datetime.now() + timedelta(days=1)).replace(hour=0, minute=0, second=0)
    gmaps = googlemaps.Client(key=get_api_key())
    directions_result = gmaps.directions(origin,
                                         destination,
                                         mode="driving",
                                         departure_time=now)
    return directions_result[0]['legs'][0]['duration']['value']

# @cache_to_disk(CACHE_DAYS)
# def get_ways_from_points(points):

EARTH_RADIUS = 6371000

def to_radians(points):
    points = np.array(points)
    return np.radians(points)

def pairwise_earth_distance(coords1, coords2):
    D = haversine_distances(to_radians(coords1), to_radians(coords2)) * EARTH_RADIUS  # in meters
    return D

@cache_to_disk(CACHE_DAYS)
def get_primary_surface(points):
    bounding_box = ','.join(map(lambda x: '%.5f' % x, (min([p[0] for p in points]), min([p[1] for p in points]), max([p[0] for p in points]), max([p[1] for p in points]))))
    overpass = Overpass()
    result = overpass.query(f'node({bounding_box}); out body;')
    node_ids = []
    node_locs = []
    for element in result.elements():
        node_ids.append(element.id())
        node_locs.append((element.lat(), element.lon()))
    result = overpass.query(f'way({bounding_box}); out body;')
    node_to_meta = defaultdict(list)
    for element in result.elements():
        my_node_ids = [node.id() for node in element.nodes()]
        for node_id in my_node_ids:
            node_to_meta[node_id].append(element.tags())
    D = pairwise_earth_distance(points, node_locs)
    min_indices = D.argmin(axis=1)
    min_values = D.min(axis=1)
    m = min_values < 30
    indices = min_indices[m]
    surfaces = Counter()
    surfaces.update('unknown' for _ in range(np.count_nonzero(~m)))
    for index in indices:
        node_id = node_ids[index]
        for meta in node_to_meta[node_id]:
            if meta is None or 'surface' not in meta:
                continue
            surfaces.update([meta['surface']])
    if len(surfaces) == 0:
        return 'unknown', 1.0
    primary_surface = max(surfaces, key=surfaces.get)
    primary_surface_percentage = surfaces[primary_surface] / sum(surfaces.values())
    return primary_surface, primary_surface_percentage

# def ovp_main():
#     points = [(42.97751, -71.60349), (42.97777, -71.60337), (42.97809, -71.60313), (42.9783, -71.6029), (42.97851, -71.60261), (42.97863, -71.60233), (42.97885, -71.60194), (42.979, -71.60157), (42.97916, -71.60128), (42.97927, -71.60112), (42.97943, -71.60086), (42.97959, -71.60051), (42.97978, -71.60019), (42.97999, -71.59953), (42.98002, -71.59936), (42.98016, -71.59897), (42.98027, -71.59872), (42.9805, -71.59835), (42.98082, -71.59796), (42.98103, -71.59781), (42.98126, -71.59761), (42.9814, -71.59737), (42.98152, -71.59712), (42.98193, -71.59653), (42.98205, -71.59638), (42.98216, -71.59628), (42.98232, -71.59621), (42.9825, -71.59609), (42.9826, -71.59594), (42.98275, -71.59579), (42.98285, -71.59565), (42.98288, -71.59553), (42.98291, -71.5953), (42.98277, -71.59491), (42.98274, -71.5946), (42.98266, -71.59425), (42.98265, -71.59403), (42.9826, -71.59391), (42.98256, -71.59384), (42.98248, -71.59373), (42.98233, -71.5936), (42.98219, -71.59344), (42.98207, -71.59327), (42.982, -71.59313), (42.98198, -71.59306), (42.98198, -71.59282), (42.98195, -71.59259), (42.98191, -71.59247), (42.98174, -71.59208), (42.98168, -71.59188), (42.98162, -71.59167), (42.98159, -71.59147), (42.98154, -71.59133), (42.9815, -71.59098), (42.98152, -71.59082), (42.98158, -71.59066), (42.98167, -71.59055), (42.98181, -71.59042), (42.98212, -71.59028), (42.98223, -71.59027), (42.98243, -71.59022), (42.98255, -71.59016), (42.98275, -71.58996), (42.98313, -71.58951)]
#     print(get_primary_surface(points))
#     blah

if __name__ == '__main__':
    print(time_to_travel("Boston, MA"), 'seconds')
    # ovp_main()