from geopy.distance import geodesic

def calculate_cumulative_distance(points):
    total_distance = 0.0
    out = [0.0]
    for i in range(len(points) - 1):
        total_distance += geodesic(points[i], points[i+1]).meters
        out.append(total_distance)
    return out