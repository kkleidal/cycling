import requests
from cache_to_disk import cache_to_disk

CACHE_DAYS = 365 * 1000

def batch(batch_size: int):
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = []
            for i in range(0, len(args[0]), batch_size):
                batch = args[0][i:i + batch_size]
                results.extend(func(batch, *args[1:], **kwargs))
            return results
        return wrapper
    return decorator

@cache_to_disk(CACHE_DAYS)
@batch(100)
def get_elevation(lat_lon_list: list[tuple[float, float]]) -> list[float]:
    url = 'https://api.open-elevation.com/api/v1/lookup'
    params = {
        'locations': '|'.join(f'{lat},{lon}' for lat, lon in lat_lon_list)
    }
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        elevation_data = response.json()
        if 'results' in elevation_data and len(elevation_data['results']) > 0:
            return [res['elevation'] for res in elevation_data['results']]
        else:
            raise ValueError("No elevation data found for the given coordinates.")
    else:
        raise ConnectionError(f"Failed to get elevation data. Status code: {response.status_code}")

if __name__ == '__main__':
    # Example usage
    lat_lon_list = [(44.36862, -68.23814), (44.34928, -68.2288)]
    elevations = get_elevation(lat_lon_list)
    for i, elevation in enumerate(elevations):
        print(f"Elevation at {lat_lon_list[i]}: {elevation} meters")
