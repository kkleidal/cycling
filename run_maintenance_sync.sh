#!/bin/zsh

set -euo pipefail

export HOME="/Users/kkleidal"
export PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

source "$HOME/.zprofile"

export STRAVA_CLIENT_ID="131882"

REPO_DIR="/Users/kkleidal/Projects/speed_power_model"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/maintenance_sync.log"

mkdir -p "$LOG_DIR"

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Starting maintenance sync"
  cd "$REPO_DIR"
  /opt/anaconda3/bin/python gear_mileage_tracker.py \
    --update-sheet \
    --check-maintenance \
    --sync-maintenance-csv
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] Maintenance sync completed"
} >>"$LOG_FILE" 2>&1
