# Course Survival Analysis: 2026 PEG-Light

## Overview

This document covers the end-to-end process of estimating aerodynamic and rolling
resistance parameters from a gravel activity, then using those parameters — along
with CP/W' physiology and gear constraints — to simulate a "just survive" pacing
strategy for the 2026 PEG-Light gravel course.

---

## Input Files

| File | Description |
|------|-------------|
| `~/Downloads/activity_22505007753.tcx` | Gravel ride with power meter — used to estimate Crr |
| `~/Downloads/COURSE_443962779.gpx` | 2026 PEG-Light course GPX |

---

## Step 1: Estimating Crr (and CdA) from the Gravel Activity

### Data

The TCX activity contains 3,049 trackpoints with:
- Duration: ~54 min, Distance: 15.3 km
- Elevation: 87.6 – 205.0 m (117m range, ±14% max grade)
- Power: 0 – 584 W, mean ~148 W

### Method

`fit_params()` from `impute_power.py` fits CdA and Crr simultaneously via L-BFGS-B
optimization over 30-second time-windowed averages of measured power.  It minimizes
the mean-squared error between the physics model power and the measured power.

**Braking-bias mitigation:** On gravel descents, riders brake heavily, so the actual
speed is lower than physics would predict for the grade.  Including these windows
would bias Crr upward (the optimizer compensates for "more resistance" by inflating Crr).
Before fitting, all trackpoints with computed grade < −2% are blanked (power set to NaN),
removing 20% of samples (613 rows) from the calibration windows.

### Result: Joint CdA+Crr Fit

| Parameter | Value |
|-----------|-------|
| CdA | 0.756 m² (hit upper bound) |
| Crr | 0.01082 |
| RMSE | 33.5 W |
| R² | 0.56 |

**The joint CdA+Crr fit is degenerate on this activity.** At gravel speeds (average
~8–10 m/s), aerodynamic drag is a relatively small fraction of total resistance
compared to gravity and rolling resistance.  Without a large speed range or extended
flat sections at high speed, the optimizer cannot reliably separate CdA from Crr —
and CdA drifts to the upper fitting boundary (0.8 m²).

### Fallback: Crr-Only Fit (CdA Fixed)

Since CdA > 0.55 m², the script automatically falls back to a Crr-only fit with CdA
fixed at 0.40 m² — a typical value for an endurance/gravel riding position (hoods,
slight drop, upright-ish).  Only windows with grade > +2% are used, because on climbs
gravity dominates and the drag contribution is small; this makes the Crr estimate
much less sensitive to the assumed CdA.

| Parameter | Value |
|-----------|-------|
| CdA | 0.40 m² *(fixed — typical gravel position)* |
| Crr | **0.01216** *(fitted from 32 uphill 30-s windows)* |
| RMSE | 27.6 W |
| R² | 0.535 |

**Crr = 0.0122** is physically plausible for a rougher gravel surface
(typical range: road ≈ 0.004–0.006, gravel ≈ 0.007–0.015 depending on surface
condition and tire pressure).

---

## Step 2: Course Profile

Parsed from `COURSE_443962779.gpx` using `gpx_io.parse_gpx_to_dataframe()`, with
Savitzky-Golay smoothing (window=51, order=3) applied to elevation before computing grade.

| Metric | Value |
|--------|-------|
| Distance | 48.4 km |
| Elevation range | 169 – 398 m |
| Elevation gain (smoothed) | 1,274 m |
| Max grade | +22.7% |
| Min grade | −21.9% |

---

## Step 3: Pacing Strategy

### Rider / Physiology Parameters

| Parameter | Value |
|-----------|-------|
| CP (FTP) | 245 W |
| W′ | 23,900 J |
| Total weight | 89 kg (77 kg body + 12 kg gear) |

### Gear Constraint

| Parameter | Value |
|-----------|-------|
| Lowest gear | 29T / 38T = 0.763 ratio |
| Wheel circumference | 2.136 m (700×40c) |
| Minimum cadence | 50 RPM |
| **Minimum speed** | **1.36 m/s (4.9 km/h)** |
| Max descent speed | 11.1 m/s (40 km/h, braking cap) |

### Strategy Rules

| Condition | Target Power | Speed |
|-----------|-------------|-------|
| Steep climb (power at min speed > CP) | Power at min speed (forced above CP) | 1.36 m/s (min gear speed) |
| Moderate climb (grade > 2%) | CP = 245 W | Solved from physics |
| Flat (−2% to +2%) | 60% CP = 147 W | Solved from physics |
| Downhill (grade < −2%) | 0 W (coast) | Terminal velocity, capped at 40 km/h |

On segments where even the lowest gear at 50 RPM requires more than CP, the rider
accepts the above-CP power cost and grinds at minimum speed.  The W′ balance model
(Skiba differential model, variable-dt) tracks cumulative W′ depletion and recovery.

**W′ recovery on descents / flats** uses the analytical solution to the Skiba ODE:

    W′bal(t+dt) = W′ − (W′ − W′bal(t)) × exp(−(CP−P)×dt / W′)

---

## Step 4: Simulation Results

| Metric | Value |
|--------|-------|
| **Estimated time** | **3h 00m** |
| Avg power (unweighted) | 120 W |
| Avg speed (distance-weighted) | ~16.1 km/h |
| Segments forced above CP | 23 / 5,070 (0.5%) |
| **Min W′ balance** | **18,023 J (75.4%) at 2.0 km** |
| Verdict | ✅ Rider survives |

The rider has **75% of W′ remaining at the hardest point**, which occurs at ~2 km
into the course (an early climb).  After that, sufficient flat/downhill sections
allow W′ to recover fully.  The strategy is very conservative — no danger of depletion.

### Plot: `course_simulation.png`

Four-panel figure showing:
1. **Elevation** — the course profile (169–398 m, ~1274m gain)
2. **Power** — target power vs. distance; red fill = above-CP sections
3. **W′ Balance** — W′ as % of W′max; green fill, orange warning at 20%
4. **Speed** — speed in km/h; dashed lines for min gear speed (4.9) and descent cap (40)

---

## Utilities Added

### `simulate_course.py`

New script at the project root. CLI entry point for the entire pipeline.

**Functions:**

| Function | Purpose |
|----------|---------|
| `fit_crr_only(df, weight_kg, cda, ...)` | Fit Crr with CdA fixed; uses uphill windows only to reduce CdA sensitivity |
| `load_course(gpx_path, ...)` | Parse GPX, smooth elevation, compute grade profile |
| `solve_speed_for_power(grade_pct, target_power, ...)` | Invert physics equation (find speed given power + grade) via brentq |
| `compute_power_at_speed(grade_pct, speed, ...)` | Forward model: power at given speed/grade |
| `simulate_wbal_variable_dt(w_prime, cp, powers, dt_array)` | Skiba W′ balance with variable-length time steps |
| `pacing_strategy(...)` | Assign power/speed per segment, simulate W′ balance |
| `plot_simulation(...)` | Four-panel matplotlib figure |
| `main()` | CLI: fitting → course load → simulation → plot |

**Existing utilities reused:**

| Source | Used |
|--------|------|
| `gpx_io.parse_gpx_to_dataframe()` | Course GPX parsing |
| `impute_power.load_file()` | TCX activity loading |
| `impute_power.fit_params()` | Initial joint CdA+Crr fit |
| `impute_power._power_np()` | Forward power computation in Crr-only fit and pacing |
| `impute_power._savgol_safe()` | Elevation and grade smoothing |
| `impute_power._compute_kinematics()` | Grade / speed extraction for descent-blanking |
| `estimate_cp.simulate_wbal()` | Reference model for the variable-dt W′ implementation |

**Usage:**

```bash
# Auto-fit Crr from a gravel activity, simulate the course:
python simulate_course.py \
    --course ~/Downloads/COURSE_443962779.gpx \
    --activity ~/Downloads/activity_22505007753.tcx \
    --weight 89 --cp 245 --w-prime 23900 \
    --gear-ratio 0.7632 --wheel-circ 2.136 --min-cadence 50 \
    --max-descent-speed 40

# Manual parameters (skip fitting):
python simulate_course.py \
    --course course.gpx \
    --weight 89 --cp 245 --w-prime 23900 \
    --cda 0.40 --crr 0.0122
```

---

## Caveats / Limitations

1. **CdA estimation from slow gravel rides is unreliable.**  A dedicated aerodynamic
   field test (flat road, varied speed sections) is needed for an accurate CdA estimate.
   The 0.40 m² default is a position-based estimate, not measured.

2. **Pacing is idealized.**  The simulation assumes perfectly steady power through each
   segment; real riding has bursts, braking, cornering.

3. **No fatigue model.**  CP and W′ are assumed constant throughout; in reality they
   degrade over multi-hour efforts (hyperbolic CPM could be used if known).

4. **Wind assumed zero.**  Add `--rho` adjustment for altitude (default 1.2 ≈ ~200m
   elevation) but no headwind/tailwind modeled.

5. **Terrain smoothing.**  Grade is derived from smoothed GPS elevation; short sharp
   ramps may appear shallower than they are.
