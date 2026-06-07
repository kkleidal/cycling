# TODO

## Gear mileage tracker setup

The tracker is now config-driven so it can aggregate multiple Strava accounts,
update the shared Google Sheet, and keep a CSV ledger of pending maintenance.

1. Copy `gear_mileage_tracker_config.example.json` to:
   ```
   gear_mileage_tracker_config.json
   ```
2. Edit the config with your real bike names, component tab names, and token cache
   locations.
3. Make sure your Strava app credentials are available in the environment:
   ```
   export STRAVA_CLIENT_ID=...
   export STRAVA_CLIENT_SECRET=...
   ```
4. On the first run for each user, a browser window will open for Strava
   authorization. Tokens are stored at the configured `strava.token_cache`
   paths, so Ken and Lauren can each keep an independent token.

## Google Sheets auth setup (one-time)

The `--update-sheet`, `--check-maintenance`, and `--sync-maintenance-csv` flags
require OAuth2 credentials for the Google Sheets API.

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create or
   select a project.
2. Enable the **Google Sheets API** and **Google Drive API** for the project.
3. Go to **APIs & Services -> Credentials -> Create Credentials -> OAuth 2.0 Client ID**.
4. Choose **Desktop app** as the application type.
5. Download the credentials JSON and save it to:
   ```
   ~/.config/gspread/credentials.json
   ```
6. On first run with any Sheets-related flag, a browser window will open asking
   you to authorize. The token is cached at
   `~/.config/gspread/authorized_user.json` and reused on subsequent runs.

## CSV behavior

`pending_maintenance.csv` is a ledger, not a transient report.

- Active items stay in the file with `pending=true`.
- Existing `ken_notified` and `notified_at` fields are preserved across runs.
- When a maintenance item disappears from the sheet because the underlying
  component row was closed out or replaced, the tracker marks it as
  `status=completed` and `pending=false`.

## Example usage

```bash
# Pull latest Strava data for every configured user, update the sheet summary,
# print due/soon maintenance, and reconcile the maintenance CSV.
python gear_mileage_tracker.py \
  --update-sheet \
  --check-maintenance \
  --sync-maintenance-csv

# Just refresh the CSV ledger from the current sheet state:
python gear_mileage_tracker.py --sync-maintenance-csv

# Normal run (analytics export only):
python gear_mileage_tracker.py
```
