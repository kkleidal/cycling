import os
from datetime import datetime, timedelta, date
import json
import pickle as pkl
import gzip
import pandas as pd
from tqdm import tqdm

from strava_auth import authorize_strava
from stravalib.client import Client

def run_analytics(gear_mapping, activity_table):
    artificial_date = date(2025, 6, 1)
    if artificial_date is None:
        artificial_date = datetime.now().date()
    activity_table = activity_table[activity_table["activity_date"] <= artificial_date]
    rides = activity_table[activity_table["activity_type"].str.contains("Ride") & activity_table["gear_id"].notna()]
    rides['is_virtual'] = rides['activity_type'].str.contains("Virtual")
    rides['virtual_moving_time_seconds'] = rides['activity_moving_time_seconds'] * rides['is_virtual']
    rides['virtual_distance_meters'] = rides['activity_distance_meters'] * rides['is_virtual']
    rides['virtual_kj'] = rides['activity_kj'] * rides['is_virtual']
    rides['real_moving_time_seconds'] = rides['activity_moving_time_seconds'] * ~rides['is_virtual']
    rides['real_distance_meters'] = rides['activity_distance_meters'] * ~rides['is_virtual']
    rides['real_kj'] = rides['activity_kj'] * ~rides['is_virtual']
    agg = rides[["gear_id", "is_virtual", "virtual_moving_time_seconds", "virtual_distance_meters", "virtual_kj", "real_moving_time_seconds", "real_distance_meters", "real_kj"]].groupby("gear_id").agg({
        "virtual_moving_time_seconds": "sum",
        "virtual_distance_meters": "sum",
        "virtual_kj": "sum",
        "real_moving_time_seconds": "sum",
        "real_distance_meters": "sum",
        "real_kj": "sum",
    }).reset_index()
    agg['gear_id'] = agg['gear_id'].map(gear_mapping)
    # Replace moving time seconds with hours, distance meters with miles:
    agg['virtual_moving_time_hours'] = agg['virtual_moving_time_seconds'] / 3600
    agg['real_moving_time_hours'] = agg['real_moving_time_seconds'] / 3600
    agg['virtual_distance_miles'] = agg['virtual_distance_meters'] / 1609.34
    agg['real_distance_miles'] = agg['real_distance_meters'] / 1609.34
    # Drop seconds and meters columns
    agg = agg.drop(columns=["virtual_moving_time_seconds", "virtual_distance_meters", "real_moving_time_seconds", "real_distance_meters"])
    agg['total_moving_time_hours'] = agg['virtual_moving_time_hours'] + agg['real_moving_time_hours']
    agg['total_distance_miles'] = agg['virtual_distance_miles'] + agg['real_distance_miles']
    agg['total_kj'] = agg['virtual_kj'] + agg['real_kj']
    agg = agg.sort_values(by="total_distance_miles", ascending=False)
    path = f"gear_mileage_analytics-{artificial_date.strftime('%Y-%m-%d')}.csv"
    agg.to_csv(path, index=False)
    print(path)

def accumulate_gear_mileage(client: Client):
    if os.path.exists("gear_mileage_checkpoint.pkl.gz"):
        with gzip.open("gear_mileage_checkpoint.pkl.gz", "rb") as f:
            gear_mileage_checkpoint = pkl.load(f)
    else:
        gear_mileage_checkpoint = {}
    if gear_mileage_checkpoint:
        old_activity_table = pd.DataFrame(gear_mileage_checkpoint["activity_table"])
        prev_oldest_date = old_activity_table["activity_date"].max()
        activity_ids_processed = set(old_activity_table["activity_id"])
        gear_mapping = gear_mileage_checkpoint.get("gear_mapping", {})
    else:
        old_activity_table = None
        prev_oldest_date = None
        activity_ids_processed = set()
        gear_mapping = {}

    rows = []
    oldest_date = prev_oldest_date
    for activity in tqdm(
        client.get_activities(
            after=datetime.combine(oldest_date, datetime.min.time()) - timedelta(days=1) if oldest_date else None
        )
    ):
        if activity.id in activity_ids_processed:
            continue
        activity_ids_processed.add(activity.id)

        start_datetime = activity.start_date_local
        start_date = start_datetime.date()
        if oldest_date is None or start_date > oldest_date:
            oldest_date = start_date

        gear_id = activity.gear_id
        activity_type = activity.type
        activity_distance_meters = float(activity.distance)
        activity_moving_time_seconds = activity.moving_time.seconds
        activity_kj = activity.kilojoules
        rows.append({
            'activity_id': activity.id,
            'activity_date': start_date,
            'gear_id': gear_id,
            'activity_type': activity_type,
            'activity_distance_meters': activity_distance_meters,
            'activity_moving_time_seconds': activity_moving_time_seconds,
            'activity_kj': activity_kj,
        })
    
    if old_activity_table is not None:
        activity_table = pd.concat([old_activity_table, pd.DataFrame(rows)])
    else:
        activity_table = pd.DataFrame(rows)
    seen_gear = set(activity_table["gear_id"])
    for gear_id in seen_gear:
        if gear_id and gear_id not in gear_mapping:
            gear = client.get_gear(gear_id)
            gear_mapping[gear_id] = gear.name
    with gzip.open("gear_mileage_checkpoint.new.pkl.gz", "wb") as f:
        pkl.dump({
            'activity_table': activity_table,
            'gear_mapping': gear_mapping,
        }, f)
    if os.path.exists("gear_mileage_checkpoint.pkl.gz"):
        os.rename("gear_mileage_checkpoint.pkl.gz", "gear_mileage_checkpoint.old.pkl.gz")
    os.rename("gear_mileage_checkpoint.new.pkl.gz", "gear_mileage_checkpoint.pkl.gz")

    run_analytics(gear_mapping, activity_table)

if __name__ == "__main__":
    authorize_strava(accumulate_gear_mileage)