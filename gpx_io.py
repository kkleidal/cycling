import pandas as pd
import gpxpy
from typing import Optional
import os
from geopy.distance import geodesic

def parse_gpx_to_dataframe(gpx_file_path: str) -> pd.DataFrame:
    """
    Parse a GPX file into a pandas DataFrame with distance and elevation data.
    
    Parameters:
    -----------
    gpx_file_path : str
        Path to the GPX file to parse
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing:
        - distance_km: cumulative distance in kilometers
        - elevation_m: elevation in meters (if available)
        - time: datetime (if available)
        - track_name: str (if available)
    """
    if not os.path.exists(gpx_file_path):
        raise FileNotFoundError(f"GPX file not found: {gpx_file_path}")
    
    # Parse the GPX file
    with open(gpx_file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)
    
    # Prepare data structure for DataFrame
    points_data = []
    
    # Extract track points
    for track_idx, track in enumerate(gpx.tracks):
        track_name = track.name if track.name else f"Track {track_idx}"
        
        for segment_idx, segment in enumerate(track.segments):
            # Initialize variables for calculating cumulative distance
            cumulative_distance = 0.0
            prev_point = None
            
            for point in segment.points:
                point_data = {
                    'track_name': track_name,
                }
                
                # Calculate distance from previous point
                if prev_point:
                    point_distance = geodesic(
                        (prev_point.latitude, prev_point.longitude),
                        (point.latitude, point.longitude)
                    ).kilometers
                    cumulative_distance += point_distance
                
                point_data['distance_km'] = cumulative_distance
                
                # Add elevation if available
                if point.elevation is not None:
                    point_data['elevation_m'] = point.elevation
                
                # Add time information if available
                if point.time is not None:
                    point_data['time'] = point.time
                
                points_data.append(point_data)
                prev_point = point
    
    # Create DataFrame
    if not points_data:
        return pd.DataFrame()  # Return empty DataFrame if no points found
    
    df = pd.DataFrame(points_data)
    
    # Sort by time if available
    if 'time' in df.columns:
        df = df.sort_values('time')
    
    return df
