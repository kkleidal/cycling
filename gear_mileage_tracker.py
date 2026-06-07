import argparse
from datetime import date, datetime, timedelta
import gzip
import json
import os
from pathlib import Path
import pickle as pkl

import pandas as pd
from tqdm import tqdm

from strava_auth import get_authorized_client

DEFAULT_CONFIG_PATH = "gear_mileage_tracker_config.json"
DEFAULT_CHECKPOINT_PATH = "gear_mileage_checkpoint.pkl.gz"

SUMMARY_COLS = [
    "gear_id",
    "virtual_kj",
    "real_kj",
    "virtual_moving_time_hours",
    "real_moving_time_hours",
    "virtual_distance_miles",
    "real_distance_miles",
    "total_moving_time_hours",
    "total_distance_miles",
    "total_kj",
]

COMPONENT_ID_COLS = [
    "Component",
    "Date Started",
    "MBR",
    "Total Mileage",
    "Years Since Started",
]

MAINTENANCE_CSV_COLS = [
    "maintenance_key",
    "owner_key",
    "owner_name",
    "bike_key",
    "bike_name",
    "component",
    "sheet_name",
    "date_started",
    "status",
    "pending",
    "max_percent_used",
    "reasons",
    "component_row",
    "first_seen_at",
    "last_seen_at",
    "resolved_at",
    "ken_notified",
    "notified_at",
]


def _resolve_path(base_dir, raw_path):
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _get_sheets_client():
    """Authenticated gspread client."""
    import gspread

    return gspread.oauth()


def _load_tracker_config(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open("r") as f:
        config = json.load(f)

    base_dir = config_path.parent
    config["config_path"] = str(config_path)
    config["base_dir"] = str(base_dir)

    if "spreadsheet_id" not in config:
        raise ValueError("Config must define spreadsheet_id.")
    if "summary_sheet_gid" not in config:
        raise ValueError("Config must define summary_sheet_gid.")
    if "users" not in config or not config["users"]:
        raise ValueError("Config must define at least one user.")

    config.setdefault("maintenance_threshold", 0.8)
    config.setdefault("pending_maintenance_csv", "pending_maintenance.csv")
    config.setdefault("checkpoint_path", DEFAULT_CHECKPOINT_PATH)
    config["pending_maintenance_csv"] = str(
        _resolve_path(base_dir, config["pending_maintenance_csv"])
    )
    config["checkpoint_path"] = str(_resolve_path(base_dir, config["checkpoint_path"]))

    for user in config["users"]:
        if "key" not in user or "name" not in user:
            raise ValueError("Each user must define key and name.")
        user.setdefault("strava", {})
        user["strava"].setdefault("client_id_env", "STRAVA_CLIENT_ID")
        user["strava"].setdefault("client_secret_env", "STRAVA_CLIENT_SECRET")
        user["strava"].setdefault("token_cache", f"strava_tokens/{user['key']}.json")
        user["strava"]["token_cache"] = str(
            _resolve_path(base_dir, user["strava"]["token_cache"])
        )
        user.setdefault("bikes", [])
        if not user["bikes"]:
            raise ValueError(f"User {user['key']} must define at least one bike.")
        for bike in user["bikes"]:
            if "key" not in bike or "name" not in bike or "component_sheet" not in bike:
                raise ValueError(
                    f"Bike entries for user {user['key']} must define key, name, and component_sheet."
                )
            bike.setdefault("summary_row_name", bike["name"])
            bike.setdefault("match_names", [bike["name"]])

    return config


def _load_checkpoint(checkpoint_path, config_users):
    if not os.path.exists(checkpoint_path):
        return {"users": {}}

    with gzip.open(checkpoint_path, "rb") as f:
        checkpoint = pkl.load(f)

    if "users" in checkpoint:
        checkpoint.setdefault("users", {})
        return checkpoint

    # Migrate the old single-user format when possible.
    if len(config_users) == 1 and "activity_table" in checkpoint:
        return {
            "users": {
                config_users[0]["key"]: {
                    "activity_table": checkpoint.get("activity_table", []),
                    "gear_mapping": checkpoint.get("gear_mapping", {}),
                }
            }
        }

    return {"users": {}}


def _save_checkpoint(checkpoint_path, checkpoint):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".new")
    with gzip.open(tmp_path, "wb") as f:
        pkl.dump(checkpoint, f)
    tmp_path.replace(checkpoint_path)


def _find_header_rows(all_values, required_cols):
    return [i for i, row in enumerate(all_values) if all(c in row for c in required_cols)]


def _parse_float(value):
    try:
        text = str(value).replace(",", "").strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def _load_user_state(checkpoint, user_key):
    raw_state = checkpoint["users"].get(user_key, {})
    if raw_state.get("activity_table"):
        activity_table = pd.DataFrame(raw_state["activity_table"])
        if "activity_date" in activity_table.columns:
            activity_table["activity_date"] = pd.to_datetime(
                activity_table["activity_date"]
            ).dt.date
    else:
        activity_table = pd.DataFrame(
            columns=[
                "activity_id",
                "activity_date",
                "gear_id",
                "activity_type",
                "activity_distance_meters",
                "activity_moving_time_seconds",
                "activity_kj",
            ]
        )
    return activity_table, raw_state.get("gear_mapping", {}), raw_state


def _save_user_state(checkpoint, user_key, activity_table, gear_mapping, athlete):
    checkpoint["users"][user_key] = {
        "activity_table": activity_table.to_dict(orient="records"),
        "gear_mapping": gear_mapping,
        "athlete_id": athlete.id,
        "athlete_name": f"{athlete.firstname} {athlete.lastname}",
    }


def _empty_activity_table():
    return pd.DataFrame(
        columns=[
            "activity_id",
            "activity_date",
            "gear_id",
            "activity_type",
            "activity_distance_meters",
            "activity_moving_time_seconds",
            "activity_kj",
        ]
    )


def _bike_lookup(user_config):
    gear_name_to_bike = {}
    for bike in user_config["bikes"]:
        for match_name in bike.get("match_names", []):
            gear_name_to_bike[match_name] = bike
    return gear_name_to_bike


def _fetch_user_activities(client, user_config, cached_activity_table, gear_mapping):
    activity_ids_processed = set(cached_activity_table["activity_id"])
    after = None
    if not cached_activity_table.empty:
        newest_date = cached_activity_table["activity_date"].max()
        after = datetime.combine(newest_date, datetime.min.time()) - timedelta(days=1)

    rows = []
    for activity in tqdm(
        client.get_activities(after=after), desc=f"Fetching {user_config['name']} rides"
    ):
        if activity.id in activity_ids_processed:
            continue

        activity_ids_processed.add(activity.id)
        rows.append(
            {
                "activity_id": activity.id,
                "activity_date": activity.start_date_local.date(),
                "gear_id": activity.gear_id,
                "activity_type": str(activity.type),
                "activity_distance_meters": float(activity.distance),
                "activity_moving_time_seconds": activity.moving_time.seconds,
                "activity_kj": float(activity.kilojoules or 0.0),
            }
        )

    if rows:
        activity_table = pd.concat(
            [cached_activity_table, pd.DataFrame(rows)], ignore_index=True
        )
    else:
        activity_table = cached_activity_table.copy()

    if not activity_table.empty:
        seen_gear_ids = {
            gear_id for gear_id in activity_table["gear_id"].dropna().unique().tolist()
        }
        for gear_id in seen_gear_ids:
            if gear_id not in gear_mapping:
                gear_mapping[gear_id] = client.get_gear(gear_id).name

    return activity_table, gear_mapping


def _normalize_user_rides(user_config, activity_table, gear_mapping, as_of_date=None):
    if activity_table.empty:
        return pd.DataFrame()

    rides = activity_table.copy()
    if as_of_date is not None:
        rides = rides[rides["activity_date"] <= as_of_date]
    rides = rides[
        rides["activity_type"].str.contains("Ride", na=False) & rides["gear_id"].notna()
    ].copy()
    if rides.empty:
        return rides

    rides["gear_name"] = rides["gear_id"].map(gear_mapping)
    bike_by_name = _bike_lookup(user_config)
    rides["bike"] = rides["gear_name"].map(bike_by_name)

    unmatched = sorted(
        {
            gear_name
            for gear_name in rides.loc[rides["bike"].isna(), "gear_name"].dropna().unique()
        }
    )
    if unmatched:
        print(
            f"Skipping unconfigured gear for {user_config['name']}: "
            + ", ".join(unmatched)
        )

    rides = rides[rides["bike"].notna()].copy()
    if rides.empty:
        return rides

    rides["bike_key"] = rides["bike"].map(lambda bike: bike["key"])
    rides["summary_row_name"] = rides["bike"].map(lambda bike: bike["summary_row_name"])
    rides["owner_key"] = user_config["key"]
    rides["owner_name"] = user_config["name"]
    rides["is_virtual"] = rides["activity_type"].str.contains("Virtual", na=False)

    rides["virtual_moving_time_seconds"] = (
        rides["activity_moving_time_seconds"] * rides["is_virtual"]
    )
    rides["virtual_distance_meters"] = rides["activity_distance_meters"] * rides["is_virtual"]
    rides["virtual_kj"] = rides["activity_kj"] * rides["is_virtual"]
    rides["real_moving_time_seconds"] = (
        rides["activity_moving_time_seconds"] * ~rides["is_virtual"]
    )
    rides["real_distance_meters"] = rides["activity_distance_meters"] * ~rides["is_virtual"]
    rides["real_kj"] = rides["activity_kj"] * ~rides["is_virtual"]

    return rides


def _aggregate_rides(config, ride_tables, as_of_date):
    numeric_cols = [
        "virtual_kj",
        "real_kj",
        "virtual_moving_time_hours",
        "real_moving_time_hours",
        "virtual_distance_miles",
        "real_distance_miles",
        "total_moving_time_hours",
        "total_distance_miles",
        "total_kj",
    ]
    base_rows = []
    bike_registry = {}
    for user in config["users"]:
        for bike in user["bikes"]:
            base_rows.append({"gear_id": bike["summary_row_name"]})
            bike_registry[bike["summary_row_name"]] = {
                "owner_name": user["name"],
                "bike_name": bike["name"],
            }

    agg = pd.DataFrame(base_rows).drop_duplicates(subset=["gear_id"])
    for col in numeric_cols:
        agg[col] = 0.0

    non_empty_tables = [table for table in ride_tables if not table.empty]
    if non_empty_tables:
        rides = pd.concat(non_empty_tables, ignore_index=True)
        grouped = (
            rides[
                [
                    "summary_row_name",
                    "virtual_moving_time_seconds",
                    "virtual_distance_meters",
                    "virtual_kj",
                    "real_moving_time_seconds",
                    "real_distance_meters",
                    "real_kj",
                ]
            ]
            .groupby("summary_row_name")
            .agg("sum")
            .reset_index()
            .rename(columns={"summary_row_name": "gear_id"})
        )
        grouped["virtual_moving_time_hours"] = grouped["virtual_moving_time_seconds"] / 3600
        grouped["real_moving_time_hours"] = grouped["real_moving_time_seconds"] / 3600
        grouped["virtual_distance_miles"] = grouped["virtual_distance_meters"] / 1609.34
        grouped["real_distance_miles"] = grouped["real_distance_meters"] / 1609.34
        grouped["total_moving_time_hours"] = (
            grouped["virtual_moving_time_hours"] + grouped["real_moving_time_hours"]
        )
        grouped["total_distance_miles"] = (
            grouped["virtual_distance_miles"] + grouped["real_distance_miles"]
        )
        grouped["total_kj"] = grouped["virtual_kj"] + grouped["real_kj"]
        grouped = grouped.drop(
            columns=[
                "virtual_moving_time_seconds",
                "virtual_distance_meters",
                "real_moving_time_seconds",
                "real_distance_meters",
            ]
        )
        agg = agg.set_index("gear_id")
        grouped = grouped.set_index("gear_id")
        agg.update(grouped)
        agg = agg.reset_index()

    agg = agg.sort_values(by="total_distance_miles", ascending=False).reset_index(drop=True)
    path = f"gear_mileage_analytics-{as_of_date.strftime('%Y-%m-%d')}.csv"
    agg.to_csv(path, index=False)
    print(path)
    return agg


def update_summary_table(gc, spreadsheet_id, summary_sheet_gid, agg, as_of_date):
    import gspread.utils

    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.get_worksheet_by_id(summary_sheet_gid)
    all_values = ws.get_all_values()

    header_indices = _find_header_rows(all_values, SUMMARY_COLS[:4])
    updates = []
    agg_by_gear = agg.set_index("gear_id").to_dict(orient="index")
    found_gears = set()

    for header_idx in header_indices:
        header_row = all_values[header_idx]
        col_map = {col: header_row.index(col) for col in SUMMARY_COLS if col in header_row}

        row_idx = header_idx + 1
        while row_idx < len(all_values):
            row = all_values[row_idx]
            gear_col = col_map.get("gear_id", 0)
            if gear_col >= len(row) or not str(row[gear_col]).strip():
                break

            gear_name = str(row[gear_col]).strip()
            if gear_name in agg_by_gear:
                found_gears.add(gear_name)
                gear_data = agg_by_gear[gear_name]
                for col_name, col_idx in col_map.items():
                    if col_name == "gear_id":
                        continue
                    a1 = gspread.utils.rowcol_to_a1(row_idx + 1, col_idx + 1)
                    updates.append({"range": a1, "values": [[float(gear_data[col_name])]]})
            row_idx += 1

    for row_idx, row in enumerate(all_values, start=1):
        for col_idx, value in enumerate(row, start=1):
            if str(value).strip() == "Last updated:":
                updates.append(
                    {
                        "range": gspread.utils.rowcol_to_a1(row_idx, col_idx + 1),
                        "values": [[as_of_date.isoformat()]],
                    }
                )

    missing_from_sheet = sorted(set(agg["gear_id"]) - found_gears)
    if missing_from_sheet:
        print(
            "Summary rows missing from sheet: " + ", ".join(missing_from_sheet)
        )

    if updates:
        ws.batch_update(updates)
    print(f"Sheet: updated {len(updates)} cells on summary tab.")


def _collect_maintenance_items(gc, config):
    sh = gc.open_by_key(config["spreadsheet_id"])
    threshold = float(config["maintenance_threshold"])
    due = []
    approaching = []

    for user in config["users"]:
        for bike in user["bikes"]:
            try:
                ws = sh.worksheet(bike["component_sheet"])
            except Exception as exc:
                print(
                    f"Skipping maintenance sync for {bike['name']}: "
                    f"unable to open sheet {bike['component_sheet']} ({exc})."
                )
                continue
            all_values = ws.get_all_values()
            header_indices = _find_header_rows(all_values, COMPONENT_ID_COLS)
            if not header_indices:
                print(f"No component table found on sheet {bike['component_sheet']}.")
                continue

            header_idx = header_indices[0]
            header_row = all_values[header_idx]
            col_map = {
                str(cell).strip(): idx
                for idx, cell in enumerate(header_row)
                if str(cell).strip()
            }

            row_idx = header_idx + 1
            while row_idx < len(all_values):
                row = all_values[row_idx]
                component_col = col_map.get("Component", 0)
                if component_col >= len(row) or not str(row[component_col]).strip():
                    break

                def cell(key):
                    idx = col_map.get(key)
                    if idx is None or idx >= len(row):
                        return ""
                    return str(row[idx]).strip()

                component = cell("Component")
                if cell("Date Ended"):
                    row_idx += 1
                    continue

                total_miles = _parse_float(cell("Total Mileage"))
                mbr = _parse_float(cell("MBR"))
                years_since = _parse_float(cell("Years Since Started"))
                ybr = _parse_float(cell("YBR"))
                reasons_due = []
                reasons_soon = []
                max_percent_used = 0.0

                if total_miles is not None and mbr:
                    pct = total_miles / mbr
                    max_percent_used = max(max_percent_used, pct)
                    label = f"{total_miles:.0f}/{mbr:.0f} mi ({pct * 100:.0f}%)"
                    if pct >= 1.0:
                        reasons_due.append(label)
                    elif pct >= threshold:
                        reasons_soon.append(label)

                if years_since is not None and ybr:
                    pct = years_since / ybr
                    max_percent_used = max(max_percent_used, pct)
                    label = f"{years_since:.2f}/{ybr} yr ({pct * 100:.0f}%)"
                    if pct >= 1.0:
                        reasons_due.append(label)
                    elif pct >= threshold:
                        reasons_soon.append(label)

                if reasons_due or reasons_soon:
                    item = {
                        "maintenance_key": f"{user['key']}::{bike['key']}::{component}::{cell('Date Started')}",
                        "owner_key": user["key"],
                        "owner_name": user["name"],
                        "bike_key": bike["key"],
                        "bike_name": bike["name"],
                        "component": component,
                        "sheet_name": bike["component_sheet"],
                        "date_started": cell("Date Started"),
                        "component_row": str(row_idx + 1),
                        "max_percent_used": f"{max_percent_used:.4f}",
                    }
                    if reasons_due:
                        due.append(
                            {
                                **item,
                                "status": "due",
                                "reasons": " | ".join(reasons_due),
                            }
                        )
                    else:
                        approaching.append(
                            {
                                **item,
                                "status": "approaching",
                                "reasons": " | ".join(reasons_soon),
                            }
                        )

                row_idx += 1

    return due, approaching


def print_maintenance_report(due, approaching):
    if not due and not approaching:
        print("\nNo maintenance due or approaching.")
        return

    for title, items in [
        ("MAINTENANCE PAST DUE", due),
        ("MAINTENANCE DUE SOON", approaching),
    ]:
        if not items:
            continue
        print(f"\n=== {title} ===")
        current_bike = None
        for item in items:
            bike_label = f"{item['owner_name']} / {item['bike_name']}"
            if bike_label != current_bike:
                print(f"  {bike_label}:")
                current_bike = bike_label
            prefix = "[OVERDUE]" if item["status"] == "due" else "[SOON]   "
            print(f"    {prefix} {item['component']}: {item['reasons']}")


def sync_maintenance_csv(csv_path, due, approaching, run_date):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    run_stamp = run_date.isoformat()
    current_items = {item["maintenance_key"]: item for item in due + approaching}

    existing_rows = {}
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path, dtype=str).fillna("")
        for _, row in existing_df.iterrows():
            existing_rows[row["maintenance_key"]] = row.to_dict()

    merged_rows = {}

    for key, item in current_items.items():
        existing = existing_rows.get(key, {})
        merged_rows[key] = {
            "maintenance_key": key,
            "owner_key": item["owner_key"],
            "owner_name": item["owner_name"],
            "bike_key": item["bike_key"],
            "bike_name": item["bike_name"],
            "component": item["component"],
            "sheet_name": item["sheet_name"],
            "date_started": item["date_started"],
            "status": item["status"],
            "pending": "true",
            "max_percent_used": item["max_percent_used"],
            "reasons": item["reasons"],
            "component_row": item["component_row"],
            "first_seen_at": existing.get("first_seen_at", run_stamp),
            "last_seen_at": run_stamp,
            "resolved_at": "",
            "ken_notified": existing.get("ken_notified", "false") or "false",
            "notified_at": existing.get("notified_at", ""),
        }

    for key, existing in existing_rows.items():
        if key in merged_rows:
            continue
        merged_rows[key] = {
            "maintenance_key": key,
            "owner_key": existing.get("owner_key", ""),
            "owner_name": existing.get("owner_name", ""),
            "bike_key": existing.get("bike_key", ""),
            "bike_name": existing.get("bike_name", ""),
            "component": existing.get("component", ""),
            "sheet_name": existing.get("sheet_name", ""),
            "date_started": existing.get("date_started", ""),
            "status": "completed",
            "pending": "false",
            "max_percent_used": existing.get("max_percent_used", ""),
            "reasons": existing.get("reasons", ""),
            "component_row": existing.get("component_row", ""),
            "first_seen_at": existing.get("first_seen_at", ""),
            "last_seen_at": existing.get("last_seen_at", ""),
            "resolved_at": existing.get("resolved_at", run_stamp) or run_stamp,
            "ken_notified": existing.get("ken_notified", "false") or "false",
            "notified_at": existing.get("notified_at", ""),
        }

    merged_df = pd.DataFrame(merged_rows.values())
    if merged_df.empty:
        merged_df = pd.DataFrame(columns=MAINTENANCE_CSV_COLS)
    else:
        merged_df["status_rank"] = merged_df["status"].map(
            {"due": 0, "approaching": 1, "completed": 2}
        )
        merged_df = merged_df.sort_values(
            by=["status_rank", "owner_name", "bike_name", "component", "date_started"]
        ).drop(columns=["status_rank"])
        merged_df = merged_df[MAINTENANCE_CSV_COLS]

    merged_df.to_csv(csv_path, index=False)
    pending_count = int((merged_df["pending"] == "true").sum()) if not merged_df.empty else 0
    print(f"Maintenance CSV: wrote {csv_path} with {pending_count} pending items.")


def run_analytics(config, ride_tables, as_of_date=None, update_sheet=False, check_maintenance=False, sync_maintenance_csv_flag=False):
    if as_of_date is None:
        as_of_date = datetime.now().date()

    agg = _aggregate_rides(config, ride_tables, as_of_date)

    if update_sheet or check_maintenance or sync_maintenance_csv_flag:
        gc = _get_sheets_client()
        if update_sheet:
            update_summary_table(
                gc,
                config["spreadsheet_id"],
                config["summary_sheet_gid"],
                agg,
                as_of_date,
            )
        if check_maintenance or sync_maintenance_csv_flag:
            due, approaching = _collect_maintenance_items(gc, config)
            if check_maintenance:
                print_maintenance_report(due, approaching)
            if sync_maintenance_csv_flag:
                sync_maintenance_csv(
                    config["pending_maintenance_csv"], due, approaching, as_of_date
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the tracker config JSON.",
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
        default=None,
        help="Only include activities up to this date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--update-sheet",
        action="store_true",
        help="Write updated mileage totals to the Google Sheet summary tab.",
    )
    parser.add_argument(
        "--check-maintenance",
        action="store_true",
        help="Read component tabs from Google Sheets and report maintenance due/upcoming.",
    )
    parser.add_argument(
        "--sync-maintenance-csv",
        action="store_true",
        help="Reconcile the pending maintenance CSV with the current sheet state.",
    )
    args = parser.parse_args()

    config = _load_tracker_config(args.config)
    checkpoint = _load_checkpoint(config["checkpoint_path"], config["users"])
    ride_tables = []

    for user in config["users"]:
        client = get_authorized_client(
            token_cache=user["strava"]["token_cache"],
            client_id_env=user["strava"]["client_id_env"],
            client_secret_env=user["strava"]["client_secret_env"],
            account_label=user["name"],
        )
        athlete = client.get_athlete()
        print(f"Authenticated {user['name']} as {athlete.firstname} {athlete.lastname}")

        cached_activity_table, gear_mapping, raw_state = _load_user_state(
            checkpoint, user["key"]
        )
        cached_athlete_id = raw_state.get("athlete_id")
        if cached_athlete_id is not None and cached_athlete_id != athlete.id:
            cached_name = raw_state.get("athlete_name", "another athlete")
            print(
                f"Resetting cached activity history for {user['name']}: "
                f"checkpoint belongs to {cached_name}, not {athlete.firstname} {athlete.lastname}."
            )
            cached_activity_table = _empty_activity_table()
            gear_mapping = {}
        activity_table, gear_mapping = _fetch_user_activities(
            client, user, cached_activity_table, gear_mapping
        )
        _save_user_state(checkpoint, user["key"], activity_table, gear_mapping, athlete)
        ride_tables.append(
            _normalize_user_rides(user, activity_table, gear_mapping, as_of_date=args.date)
        )

    _save_checkpoint(config["checkpoint_path"], checkpoint)
    run_analytics(
        config,
        ride_tables,
        as_of_date=args.date,
        update_sheet=args.update_sheet,
        check_maintenance=args.check_maintenance,
        sync_maintenance_csv_flag=args.sync_maintenance_csv,
    )


if __name__ == "__main__":
    main()
