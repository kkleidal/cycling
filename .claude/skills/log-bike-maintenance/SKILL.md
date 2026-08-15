---
name: log-bike-maintenance
description: Log completed bike maintenance to the gear-mileage Google Sheet and run the mileage sync. Use when the user says they did maintenance (waxed a chain, replaced tires/cables/cleats/brake pads, added sealant, tubeless conversion, etc.) on a tracked bike, or asks to "log maintenance" / "run the sync" / "run the update". Resets component intervals in the *Comp tabs and appends a human-readable row to the *Log tabs.
---

# Log bike maintenance

Records maintenance the user performed against the gear-mileage tracker's Google Sheet, so component-wear intervals reset and the maintenance report reflects reality.

## System overview

- **Sync script:** `./run_maintenance_sync.sh` — pulls Strava mileage, updates the sheet's summary, checks maintenance, writes `pending_maintenance.csv`. Logs to `logs/maintenance_sync.log`. Exit 0 = success.
- **Spreadsheet ID:** read from `gear_mileage_tracker_config.json` (`spreadsheet_id`). Currently `1GH6rit4DXX69Bw4AXTQRBcG6KRKvyKx0ll7T5TMF6BE`.
- **Auth:** service-account key at `~/.config/gspread/service_account.json` (SA `gear-mileage-cron@cycling-432220.iam.gserviceaccount.com`, shared on the sheet as Editor). Do NOT fall back to personal OAuth — see the `gear-mileage-sheets-auth` memory.
- **Bikes → tabs:** each bike has a `<Bike> Comp` tab (interval tracker) and a `<Bike> Log` tab (history). Bikes: `Emonda` (Ken), `Revolt` (Ken), `Turbo Creo II` (Lauren). Confirm current tab names with `sh.worksheets()`.

## Comp tab layout (the tracker)

Rows 1-3 hold current counters, three columns each = **[total, real, virtual]**:
- Row 1 `Current mileage:` → B1 total, **C1 real, D1 virtual**
- Row 2 `Current hours`   → C2 real, D2 virtual
- Row 3 `Current MJ`      → C3 real, D3 virtual

Row 4 is the header. Component rows follow. Columns:

| Col | Field | Notes |
|-----|-------|-------|
| A | Component | e.g. `Chain Wax`, `Sealant`, `Cleats` |
| B | Date Started | **write** today's date |
| C | Real Mileage? | TRUE/FALSE flag — read only |
| D | Virtual Mileage? | TRUE/FALSE flag — read only |
| E | Mileage Started | **write** current counter |
| F | Hours Started | **write** current counter |
| G | MJ Started | **write** current counter |
| H-K | Date/Mileage/Hours/KJ Ended | usually blank |
| L,M,N | Total Mileage/Hours/MJ | **FORMULA — never write** |
| O | Years Since Started | **FORMULA — never write** |
| P | MBR | mileage-based replacement interval |
| Q | Past MBR | **FORMULA — never write** |
| R | YBR | year-based interval |
| S | Past TBR | **FORMULA — never write** |
| T | Comments | free text, optional |

Only ever write columns **B, E, F, G** (and optionally T). The `Total` formula is
`(IF(Real,C1,0)+IF(Virtual,D1,0)) - E`, so setting Started = the matching current
counter resets the interval to ~0.

## Resetting an interval (maintenance done)

For the component row, set Date Started = today and Started counters = the current
value at **full precision** (read `B1:D3` with `value_render_option='UNFORMATTED_VALUE'`):
- **Real? and Virtual? both TRUE** (e.g. Chain Wax): Started = real+virtual (C+D).
- **Real only** (e.g. Sealant, Tires, Brake Pads): Started = real (C).
- **Virtual only**: Started = virtual (D).

## Log tab layout (history)

Header: `Date, Total Mileage, Total Hours, Total MJ, Real Mileage, Real Hours, Real MJ, Description of Work`.
Append one row per bike per maintenance session (combine multiple items into one description).
Use rounded (1-decimal) counter values to match existing style.

## One-time work with no interval row

Some work has no dedicated Comp row (tubeless conversion, new valve, new tubeless tape,
wheel true, torque-to-spec). Do NOT invent a Comp row — capture it only in the Log
description, matching the existing style (see Émonda Log tubeless-conversion entries).

## Procedure

1. Run `./run_maintenance_sync.sh` first if the user asked (so counters are current), or note it will run after.
2. Map each item the user reports to a component (or Log-only note). Ask only if genuinely ambiguous.
3. Dump the target `*Comp` tabs (`ws.get_all_values()`) to find exact row numbers and the Real?/Virtual? flags.
4. Read full-precision current counters from `B1:D3`.
5. Write Comp resets (B + E:G) with `value_input_option='USER_ENTERED'`; append Log rows.
6. Verify: re-read each reset row's Total Mileage (col L) ≈ 0 and `Past MBR` = FALSE; confirm Log tails.
7. Re-run `./run_maintenance_sync.sh` to refresh the summary tab and `pending_maintenance.csv`, then report the updated past-due / due-soon status.

## Reset script template

```python
import gspread
gc = gspread.service_account(filename='/Users/kkleidal/.config/gspread/service_account.json')
sh = gc.open_by_key('1GH6rit4DXX69Bw4AXTQRBcG6KRKvyKx0ll7T5TMF6BE')
TODAY = 'YYYY-MM-DD'

def current(tab):
    v = sh.worksheet(tab).get('B1:D3', value_render_option='UNFORMATTED_VALUE')
    tot=lambda r: v[r][1]+v[r][2]; real=lambda r: v[r][1]
    return dict(both=(tot(0),tot(1),tot(2)), real=(real(0),real(1),real(2)))

def reset(tab, row, m, h, j):
    ws = sh.worksheet(tab)
    ws.update(values=[[TODAY]], range_name=f'B{row}', value_input_option='USER_ENTERED')
    ws.update(values=[[m,h,j]], range_name=f'E{row}:G{row}', value_input_option='USER_ENTERED')

# example: chain wax (both TRUE) on Emonda row 7
c = current('Emonda Comp'); reset('Emonda Comp', 7, *c['both'])
sh.worksheet('Emonda Log').append_row(
    [TODAY, 3777.1, 228.1, 127.2, 2515.2, 158.8, 86.6, 'Rewax chain'],
    value_input_option='USER_ENTERED')
```

## Safety

- These are durable writes to the user's real spreadsheet. Only write columns B/E/F/G/T.
- If unsure which component a reported item maps to, ask before writing.
- After writing, always verify Total≈0 before reporting success.
