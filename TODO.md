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

## Google Sheets auth setup

The tracker now prefers a Google service account for unattended runs. If the
service account key file is present, the script will use it automatically and
skip end-user browser auth.

### Preferred: service account

1. In Google Cloud Console, create or select a project.
2. Enable the **Google Sheets API** and **Google Drive API**.
3. Create a service account under **IAM & Admin -> Service Accounts**.
4. Create a JSON key for that service account and save it to:
   ```
   ~/.config/gspread/service_account.json
   ```
   You can also point the tracker at a different path with
   `GSPREAD_SERVICE_ACCOUNT_FILE=/path/to/key.json`.
5. Open the JSON file and copy the service account email, which looks like:
   `name@project-id.iam.gserviceaccount.com`
6. Share the maintenance spreadsheet with that email as an editor.

### Fallback: desktop OAuth

If no service account key file is present, the tracker falls back to OAuth2
desktop auth.

1. Go to **APIs & Services -> Credentials -> Create Credentials -> OAuth 2.0 Client ID**.
2. Choose **Desktop app** as the application type.
3. Download the credentials JSON and save it to:
   ```
   ~/.config/gspread/credentials.json
   ```
4. On first run with any Sheets-related flag, a browser window will open asking
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
