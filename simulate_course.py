#!/usr/bin/env python3
"""
Simulate a pacing strategy for a gravel course given rider physiology and gear constraints.

Steps:
1. Estimate Crr/CdA from a gravel activity TCX file with measured power
2. Load the GPX course and compute the grade profile
3. Simulate a "just survive" pacing strategy:
   - Climbs: ride at CP (or minimum gear speed if too steep)
   - Flats: recovery watts (~60% CP)
   - Descents: coast at a comfortable max speed
4. Track W' balance forward through the course
5. Plot: elevation, power, W' balance, speed vs. distance

Usage
-----
    python simulate_course.py \\
        --course course.gpx \\
        --activity gravel_ride.tcx \\
        --weight 89 --cp 245 --w-prime 23900

    # Manual drag/rolling parameters (skip fitting):
    python simulate_course.py \\
        --course course.gpx \\
        --weight 89 --cp 245 --w-prime 23900 \\
        --cda 0.40 --crr 0.0075
"""

import argparse
import sys
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpx_io import parse_gpx_to_dataframe
from impute_power import load_file, fit_params, _power_np, _savgol_safe, _compute_kinematics, DEFAULT_PARAMS


# ---------------------------------------------------------------------------
# Crr-only fitting (when CdA is unreliable)
# ---------------------------------------------------------------------------

# Typical CdA for a gravel / sport-endurance position (hoods, slight drop)
GRAVEL_CDA_DEFAULT = 0.40   # m²


def fit_crr_only(
    df,
    weight_kg: float,
    cda: float = GRAVEL_CDA_DEFAULT,
    rho: float = 1.225,
    loss_dt: float = 2.0,
    inertia_factor: float = 1.05,
    smooth_window: int = 21,
    smooth_poly: int = 3,
    min_power_w: float = 50.0,
    max_accel: float = 0.5,
    window_seconds: int = 30,
    min_active_fraction: float = 0.7,
    min_grade_pct: float = 2.0,
) -> dict:
    """
    Fit only Crr with CdA fixed, using uphill windows where gravity dominates
    and the drag contribution is a small fraction of total power.

    This is appropriate when the activity is too slow or too short for the
    joint CdA+Crr fit to converge (joint fit hits the CdA upper bound).

    Parameters
    ----------
    min_grade_pct : only use windows where average grade exceeds this (%).
                    On climbs, drag is a small fraction of total power, making
                    Crr estimation more reliable.
    """
    from scipy.optimize import minimize_scalar
    import pandas as pd

    base = {**DEFAULT_PARAMS, "weight_kg": weight_kg, "cda": cda,
            "rho": rho, "loss_drivetrain": loss_dt, "inertia_factor": inertia_factor}

    if "power_w" not in df.columns or df["power_w"].isna().all():
        raise ValueError("No measured power data available for fitting.")

    dt_, speed_, grade_, accel_ = _compute_kinematics(df, smooth_window, smooth_poly)

    df_work = df.copy().reset_index(drop=True)
    df_work["_speed"] = speed_
    df_work["_grade"] = grade_
    df_work["_accel"] = accel_

    meas_start = int(df_work["power_w"].notna().values.argmax())
    meas = df_work.iloc[meas_start:].copy().reset_index(drop=True)

    windows = []
    i = 0
    while i + window_seconds <= len(meas):
        chunk = meas.iloc[i: i + window_seconds]
        active = (chunk["power_w"].fillna(0) > 0).mean()
        avg_accel = chunk["_accel"].abs().mean()
        avg_power = chunk["power_w"].mean()
        avg_grade = chunk["_grade"].mean()

        if (active >= min_active_fraction
                and avg_accel <= max_accel
                and avg_power >= min_power_w
                and avg_grade >= min_grade_pct):
            windows.append({
                "grade": avg_grade,
                "speed": chunk["_speed"].mean(),
                "accel": chunk["_accel"].mean(),
                "power": avg_power,
            })
        i += window_seconds

    n_win = len(windows)
    if n_win < 3:
        raise ValueError(
            f"Only {n_win} usable uphill windows (need ≥ 3). "
            "Activity may not have enough sustained climbing."
        )

    wins = pd.DataFrame(windows)
    g_fit = wins["grade"].values
    v_fit = wins["speed"].values
    a_fit = wins["accel"].values
    p_meas = wins["power"].values

    def objective(crr_val):
        p_hat = _power_np(g_fit, v_fit, a_fit,
                          cda, crr_val, rho, weight_kg, 0.0, loss_dt, inertia_factor)
        return float(np.mean((p_hat - p_meas) ** 2))

    result = minimize_scalar(objective, bounds=(0.002, 0.020), method="bounded")
    crr_fit = float(result.x)

    p_hat_cal = _power_np(g_fit, v_fit, a_fit,
                          cda, crr_fit, rho, weight_kg, 0.0, loss_dt, inertia_factor)
    rmse = float(np.sqrt(np.mean((p_hat_cal - p_meas) ** 2)))
    ss_res = np.sum((p_hat_cal - p_meas) ** 2)
    ss_tot = np.sum((p_meas - p_meas.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nCrr-only fit ({n_win} × {window_seconds}-s uphill windows, CdA fixed={cda:.3f}):")
    print(f"  Crr     : {crr_fit:.5f}")
    print(f"  RMSE    : {rmse:.1f} W")
    print(f"  R²      : {r2:.3f}")

    return {**base, "crr": crr_fit, "v_headwind": 0.0}


# ---------------------------------------------------------------------------
# Course loading
# ---------------------------------------------------------------------------

def load_course(gpx_path: str, smooth_window: int = 51, smooth_poly: int = 3):
    """
    Parse a GPX course file and return smoothed distance, elevation, and grade arrays.

    Returns
    -------
    dist_m    : cumulative distance in metres (ndarray)
    elev_m    : smoothed elevation in metres (ndarray)
    grade_pct : smoothed grade in % — positive = uphill (ndarray)
    seg_dist  : per-segment horizontal distance in metres (ndarray, same length)
    """
    df = parse_gpx_to_dataframe(gpx_path)
    if df.empty:
        raise ValueError(f"No track points found in {gpx_path}")

    dist_m = df["distance_km"].values * 1000.0
    elev_raw = df["elevation_m"].values if "elevation_m" in df.columns else np.zeros(len(df))

    # Smooth elevation to reduce GPS noise before computing grade
    elev_m = _savgol_safe(elev_raw, smooth_window, smooth_poly)

    # Segment distances (floor at 0.5 m to avoid division by zero)
    seg_dist = np.diff(dist_m, prepend=dist_m[0])
    seg_dist[0] = seg_dist[1] if len(seg_dist) > 1 else 1.0
    seg_dist = np.maximum(seg_dist, 0.5)

    # Grade = Δelev / horizontal distance × 100
    elev_diff = np.diff(elev_m, prepend=elev_m[0])
    grade_pct = np.clip((elev_diff / seg_dist) * 100.0, -40.0, 40.0)
    # Second smoothing pass to remove any residual spikes
    grade_pct = _savgol_safe(grade_pct, smooth_window, smooth_poly)

    return dist_m, elev_m, grade_pct, seg_dist


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def solve_speed_for_power(
    grade_pct: float,
    target_power: float,
    cda: float,
    crr: float,
    rho: float,
    weight_kg: float,
    v_headwind: float,
    loss_dt: float,
    max_search_speed: float = 50.0,
) -> float:
    """
    Find the groundspeed v (m/s) at which steady-state power equals target_power.

    For coasting (target_power <= 0) on a downhill, returns terminal velocity
    (where drag + rolling balances gravity).  On flat/uphill with target_power
    <= 0, returns 0.
    """
    g = 9.8067
    theta = np.arctan(grade_pct / 100.0)
    f_grav = g * np.sin(theta) * weight_kg   # positive = uphill resistance
    f_roll = g * np.cos(theta) * weight_kg * crr  # always positive (resistance)
    eta = 1.0 - loss_dt / 100.0

    if target_power <= 0:
        # Net driving force from gravity (only relevant on downhills)
        net_drive = -(f_grav + f_roll)      # positive when gravity > rolling
        if net_drive <= 0:
            return 0.0
        # Terminal: 0.5*CdA*rho*v^2 = net_drive  (headwind assumed ~0 for terminal)
        v_term = np.sqrt(2.0 * net_drive / (cda * rho))
        return min(v_term, max_search_speed)

    # Steady-state (zero acceleration):
    #   (f_grav + f_roll + 0.5*CdA*rho*(v + vw)^2) * v / eta = target_power
    def residual(v):
        f_drag = 0.5 * cda * rho * (v + v_headwind) ** 2
        return (f_grav + f_roll + f_drag) * v / eta - target_power

    # If even max_search_speed doesn't produce enough resistance (strong tailwind
    # downhill), clamp to max
    if residual(max_search_speed) < 0:
        return max_search_speed
    # If even the slowest speed already exceeds target (extreme uphill), clamp low
    if residual(0.001) > 0:
        return 0.001
    try:
        return brentq(residual, 0.001, max_search_speed, xtol=1e-4)
    except ValueError:
        return 0.001


def compute_power_at_speed(
    grade_pct: float,
    speed: float,
    cda: float,
    crr: float,
    rho: float,
    weight_kg: float,
    v_headwind: float,
    loss_dt: float,
    inertia_factor: float = 1.05,
) -> float:
    """Forward model: steady-state power (W) required at given speed and grade."""
    return float(_power_np(
        np.array([grade_pct]),
        np.array([speed]),
        np.array([0.0]),   # acceleration = 0 for steady-state
        cda, crr, rho, weight_kg, v_headwind, loss_dt, inertia_factor,
    )[0])


# ---------------------------------------------------------------------------
# W' balance (variable time step)
# ---------------------------------------------------------------------------

def simulate_wbal_variable_dt(
    w_prime: float,
    cp: float,
    powers: np.ndarray,
    dt_array: np.ndarray,
) -> np.ndarray:
    """
    Skiba differential W' balance model adapted for variable-length time steps.

    Depletion (P > CP):
        W'bal -= (P - CP) * dt

    Recovery (P < CP) — analytical integration of the Skiba ODE
        dW'/dt = (W'max - W'bal) * (CP - P) / W'max
    Solution over interval dt:
        W'bal(t+dt) = W' - (W' - W'bal(t)) * exp(-(CP - P) * dt / W')

    Parameters
    ----------
    w_prime  : maximum W' capacity (J)
    cp       : critical power (W)
    powers   : power at each segment (W), shape (n,)
    dt_array : duration of each segment (s), shape (n,)
    """
    n = len(powers)
    wbal = np.empty(n)
    wbal[0] = w_prime

    for i in range(1, n):
        p = float(powers[i])
        dt = float(dt_array[i])
        wb = wbal[i - 1]

        if p > cp:
            wbal[i] = wb - (p - cp) * dt
        else:
            k = (cp - p) / w_prime
            wbal[i] = w_prime - (w_prime - wb) * np.exp(-k * dt)

    return wbal


# ---------------------------------------------------------------------------
# Pacing strategy
# ---------------------------------------------------------------------------

def pacing_strategy(
    grade_pct: np.ndarray,
    seg_dist: np.ndarray,
    cp: float,
    w_prime: float,
    cda: float,
    crr: float,
    rho: float,
    weight_kg: float,
    v_headwind: float,
    loss_dt: float,
    inertia_factor: float,
    min_speed: float,
    max_descent_speed: float,
    target_power: float = 150.0,
    climb_power: float = 150.0,
    climb_threshold_pct: float = 2.0,
    descent_threshold_pct: float = -2.0,
):
    """
    Simulate pacing with separate power targets for climbs and flat/rolling terrain.

    Rules:
    -  Descents (grade < descent_threshold): coast (0 W), speed capped at
       max_descent_speed (braking).
    -  Climbs (grade >= climb_threshold): ride at climb_power.
    -  Everything else: ride at target_power.
    In both non-descent cases: if the grade is so steep that even the lowest
    gear at minimum cadence (min_speed) demands more than the target, the gear
    constraint takes over — ride at min_speed and accept the higher power.

    Returns
    -------
    power_w  : power per segment (W)
    speed_ms : speed per segment (m/s)
    wbal_j   : W' balance per segment (J)
    seg_time : time per segment (s)
    """
    n = len(grade_pct)
    power_w = np.empty(n)
    speed_ms = np.empty(n)

    for i in range(n):
        g = float(grade_pct[i])

        if g < descent_threshold_pct:
            # --- Downhill: coast ---
            power_w[i] = 0.0
            v_term = solve_speed_for_power(
                g, 0.0, cda, crr, rho, weight_kg, v_headwind, loss_dt
            )
            speed_ms[i] = min(v_term, max_descent_speed)
        else:
            # --- Flat/rolling or climb ---
            tgt = climb_power if g >= climb_threshold_pct else target_power
            p_at_min = compute_power_at_speed(
                g, min_speed, cda, crr, rho, weight_kg, v_headwind, loss_dt, inertia_factor
            )
            if p_at_min > tgt:
                # Gear-limited: ride at min_speed, accept whatever power needed
                power_w[i] = max(p_at_min, 0.0)
                speed_ms[i] = min_speed
            else:
                power_w[i] = tgt
                v = solve_speed_for_power(
                    g, tgt, cda, crr, rho, weight_kg, v_headwind, loss_dt
                )
                speed_ms[i] = max(v, min_speed)

    speed_ms = np.maximum(speed_ms, 0.1)

    seg_time = seg_dist / speed_ms
    seg_time[0] = seg_time[1] if n > 1 else 1.0

    wbal_j = simulate_wbal_variable_dt(w_prime, cp, power_w, seg_time)

    return power_w, speed_ms, wbal_j, seg_time


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_simulation(
    dist_km: np.ndarray,
    elev_m: np.ndarray,
    grade_pct: np.ndarray,
    power_w: np.ndarray,
    speed_ms: np.ndarray,
    wbal_j: np.ndarray,
    cp: float,
    w_prime: float,
    min_speed: float,
    max_descent_speed: float,
    total_time_s: float,
    output_path: str,
    course_name: str = "",
    cadence_rpm: np.ndarray = None,
    min_cadence: float = 50.0,
    max_cadence: float = 95.0,
    gear_ratio: float = None,
    wheel_circ: float = None,
) -> None:
    """Five-panel figure: elevation, power, W' balance, speed, cadence vs. distance."""
    hours = int(total_time_s // 3600)
    minutes_part = int((total_time_s % 3600) // 60)
    speed_kmh = speed_ms * 3.6
    wbal_pct = np.clip(wbal_j / w_prime * 100.0, -10.0, 100.0)

    n_panels = 5 if cadence_rpm is not None else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 16 if n_panels == 5 else 13), sharex=True)
    fig.subplots_adjust(hspace=0.06)

    title = "Course Survival Simulation"
    if course_name:
        title += f"  —  {course_name}"
    title += f"   |   Est. finish time: {hours}h {minutes_part:02d}m"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ------------------------------------------------------------------
    # Panel 1 — Elevation
    # ------------------------------------------------------------------
    ax0 = axes[0]
    ax0.fill_between(dist_km, elev_m, elev_m.min() - 10, color="saddlebrown", alpha=0.45)
    ax0.plot(dist_km, elev_m, color="saddlebrown", lw=0.9)
    ax0.set_ylabel("Elevation (m)", fontsize=10)
    ax0.grid(axis="y", alpha=0.25)
    ax0.set_ylim(bottom=elev_m.min() - 20)

    # ------------------------------------------------------------------
    # Panel 2 — Power
    # ------------------------------------------------------------------
    ax1 = axes[1]
    # Light smoothing for display
    smooth_w = max(3, min(20, len(power_w) // 200) | 1)  # odd window, ≤ 20 pts
    smooth_p = np.convolve(power_w, np.ones(smooth_w) / smooth_w, mode="same")
    ax1.plot(dist_km, smooth_p, color="steelblue", lw=1.1, label="Power")
    ax1.axhline(cp, color="red", ls="--", lw=1.0, alpha=0.7, label=f"CP = {cp:.0f} W")
    ax1.fill_between(dist_km, smooth_p, cp, where=(smooth_p > cp),
                     color="red", alpha=0.25, label="Above CP (W′ depleting)")
    ax1.fill_between(dist_km, smooth_p, 0.0, where=(smooth_p <= cp),
                     color="steelblue", alpha=0.12)
    ax1.set_ylabel("Power (W)", fontsize=10)
    ax1.set_ylim(bottom=0, top=max(smooth_p.max() * 1.15, cp * 1.4))
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # Panel 3 — W' balance
    # ------------------------------------------------------------------
    ax2 = axes[2]
    ax2.fill_between(dist_km, wbal_pct, 0, color="mediumseagreen", alpha=0.40)
    ax2.plot(dist_km, wbal_pct, color="mediumseagreen", lw=1.1)
    ax2.axhline(0, color="red", ls="--", lw=1.0, alpha=0.8, label="W′ depleted")
    ax2.axhline(20, color="orange", ls=":", lw=0.9, alpha=0.7, label="20% buffer")
    ax2.set_ylabel("W′ Balance (%)", fontsize=10)
    ax2.set_ylim(-10, 105)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.25)

    # Annotate the minimum W' point
    min_idx = int(np.argmin(wbal_pct))
    offset_x = dist_km[-1] * 0.04
    ax2.annotate(
        f"min {wbal_pct[min_idx]:.0f}%\n@ {dist_km[min_idx]:.1f} km",
        xy=(dist_km[min_idx], wbal_pct[min_idx]),
        xytext=(
            min(dist_km[min_idx] + offset_x, dist_km[-1] * 0.92),
            wbal_pct[min_idx] + 12,
        ),
        fontsize=8,
        color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8),
    )

    # ------------------------------------------------------------------
    # Panel 4 — Speed
    # ------------------------------------------------------------------
    ax3 = axes[3]
    ax3.plot(dist_km, speed_kmh, color="darkorange", lw=1.0)
    ax3.axhline(
        min_speed * 3.6, color="purple", ls="--", lw=0.9,
        label=f"Min gear speed  {min_speed * 3.6:.1f} km/h",
    )
    ax3.axhline(
        max_descent_speed * 3.6, color="gray", ls=":", lw=0.9,
        label=f"Max descent cap  {max_descent_speed * 3.6:.0f} km/h",
    )
    ax3.set_ylabel("Speed (km/h)", fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # Panel 5 — Cadence (only if provided)
    # ------------------------------------------------------------------
    if cadence_rpm is not None:
        ax4 = axes[4]
        ax4.plot(dist_km, cadence_rpm, color="mediumslateblue", lw=1.0)
        ax4.axhline(
            min_cadence, color="red", ls="--", lw=0.9,
            label=f"Min cadence  {min_cadence:.0f} RPM",
        )
        ax4.axhline(
            max_cadence, color="gray", ls=":", lw=0.9,
            label=f"Max cadence (cap)  {max_cadence:.0f} RPM",
        )
        ax4.set_ylabel("Cadence (RPM)", fontsize=10)
        ax4.set_xlabel("Distance (km)", fontsize=10)
        ax4.set_ylim(bottom=0, top=max_cadence * 1.15)
        ax4.legend(fontsize=8, loc="upper right")
        ax4.grid(axis="y", alpha=0.25)
    else:
        ax3.set_xlabel("Distance (km)", fontsize=10)

    for ax in axes:
        ax.set_xlim(dist_km[0], dist_km[-1])

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Zone analysis and sustained efforts
# ---------------------------------------------------------------------------

# Coggan 7-zone system as % of CP/FTP
ZONES = [
    ("Z1 Recovery",    0,     0.55,  "#b0c4de"),
    ("Z2 Endurance",   0.55,  0.75,  "#6db56d"),
    ("Z3 Tempo",       0.75,  0.90,  "#f5c843"),
    ("Z4 Threshold",   0.90,  1.05,  "#f5953a"),
    ("Z5 VO2max",      1.05,  1.20,  "#e05a3a"),
    ("Z6 Anaerobic",   1.20,  1.50,  "#c0392b"),
    ("Z7 Neuromuscular",1.50, 9.99,  "#7b241c"),
]


def zone_for_power(p: float, cp: float) -> int:
    """Return 0-based zone index for a given power and CP."""
    frac = p / cp if cp > 0 else 0
    for i, (_, lo, hi, _) in enumerate(ZONES):
        if lo <= frac < hi:
            return i
    return len(ZONES) - 1


def find_sustained_climb_efforts(
    dist_km: np.ndarray,
    grade_pct: np.ndarray,
    power_w: np.ndarray,
    seg_time: np.ndarray,
    cp: float,
    climb_threshold_pct: float = 2.0,
    min_duration_s: float = 60.0,
    merge_gap_s: float = 30.0,
    top_n: int = 8,
) -> list[dict]:
    """
    Find contiguous uphill blocks and rank them by W' cost (most demanding first).

    Returns list of dicts with keys:
        start_km, end_km, dist_km, duration_s, avg_grade, avg_power,
        max_power, wprime_cost_j
    """
    n = len(grade_pct)
    on_climb = grade_pct >= climb_threshold_pct

    # Build raw blocks (start, end indices) of contiguous climbing
    blocks = []
    in_block = False
    for i in range(n):
        if on_climb[i] and not in_block:
            start = i
            in_block = True
        elif not on_climb[i] and in_block:
            blocks.append((start, i - 1))
            in_block = False
    if in_block:
        blocks.append((start, n - 1))

    # Merge blocks separated by a gap < merge_gap_s
    merged = []
    for blk in blocks:
        if merged:
            gap_time = float(seg_time[merged[-1][1] + 1 : blk[0]].sum())
            if gap_time <= merge_gap_s:
                merged[-1] = (merged[-1][0], blk[1])
                continue
        merged.append(list(blk))

    # Summarise each block
    efforts = []
    for s, e in merged:
        dur = float(seg_time[s: e + 1].sum())
        if dur < min_duration_s:
            continue
        avg_p = float(np.average(power_w[s: e + 1], weights=seg_time[s: e + 1]))
        avg_g = float(np.average(grade_pct[s: e + 1], weights=seg_time[s: e + 1]))
        max_p = float(power_w[s: e + 1].max())
        wp_cost = float(np.sum(
            np.maximum(power_w[s: e + 1] - cp, 0.0) * seg_time[s: e + 1]
        ))
        efforts.append(dict(
            start_km=float(dist_km[s]),
            end_km=float(dist_km[e]),
            dist_km=float(dist_km[e] - dist_km[s]),
            duration_s=dur,
            avg_grade=avg_g,
            avg_power=avg_p,
            max_power=max_p,
            wprime_cost_j=wp_cost,
        ))

    # Sort by W' cost (highest first); for zero-W'-cost climbs, sort by avg power
    efforts.sort(key=lambda x: (x["wprime_cost_j"], x["avg_power"]), reverse=True)
    return efforts[:top_n]


def plot_zone_analysis(
    power_w: np.ndarray,
    seg_time: np.ndarray,
    dist_km: np.ndarray,
    grade_pct: np.ndarray,
    cp: float,
    w_prime: float,
    total_time_s: float,
    climb_efforts: list[dict],
    output_path: str,
    course_name: str = "",
) -> None:
    """
    Two-panel figure:
      Left  — horizontal bar chart of time in each power zone
      Right — table of hardest sustained climb efforts
    """
    # ---- Compute zone times ----
    zone_times = np.zeros(len(ZONES))
    for i in range(len(power_w)):
        z = zone_for_power(float(power_w[i]), cp)
        zone_times[z] += float(seg_time[i])

    zone_labels = [z[0] for z in ZONES]
    zone_colors = [z[3] for z in ZONES]
    zone_pct = zone_times / total_time_s * 100.0

    # ---- Layout ----
    fig = plt.figure(figsize=(14, 7))
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.97, wspace=0.35)

    title = "Zone Distribution & Hardest Climbs"
    if course_name:
        title += f"  —  {course_name}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax_bar = fig.add_subplot(1, 2, 1)
    ax_tbl = fig.add_subplot(1, 2, 2)

    # ------------------------------------------------------------------
    # Left panel — zone bar chart (horizontal)
    # ------------------------------------------------------------------
    y_pos = np.arange(len(ZONES))
    bars = ax_bar.barh(y_pos, zone_pct, color=zone_colors, edgecolor="white",
                       linewidth=0.5, height=0.7)

    # Annotate bars with absolute time (h:mm) and %
    for bar, zt, zpct in zip(bars, zone_times, zone_pct):
        h = int(zt // 3600)
        m = int((zt % 3600) // 60)
        s = int(zt % 60)
        label = (f"{h}h {m:02d}m" if h > 0 else f"{m}m {s:02d}s") + f"  ({zpct:.1f}%)"
        ax_bar.text(
            zpct + 0.4, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=8.5, color="#333333",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(zone_labels, fontsize=9.5)
    ax_bar.set_xlabel("Time in zone (%)", fontsize=10)
    ax_bar.set_xlim(0, max(zone_pct) * 1.55)
    ax_bar.set_title(
        f"Power Zone Distribution\nCP = {cp:.0f} W  |  Total time = "
        f"{int(total_time_s//3600)}h {int((total_time_s%3600)//60):02d}m",
        fontsize=10,
    )
    ax_bar.invert_yaxis()  # Z1 at top
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Right panel — hardest climb efforts table
    # ------------------------------------------------------------------
    ax_tbl.axis("off")

    if not climb_efforts:
        ax_tbl.text(0.5, 0.5, "No sustained climbs found", ha="center", va="center",
                    fontsize=11, color="gray")
    else:
        col_labels = ["#", "At km", "Length", "Duration", "Avg\nGrade", "Avg\nPower",
                      "Peak\nPower", "W′ Cost"]
        rows = []
        for rank, ef in enumerate(climb_efforts, 1):
            dur_m = int(ef["duration_s"] // 60)
            dur_s = int(ef["duration_s"] % 60)
            rows.append([
                str(rank),
                f"{ef['start_km']:.1f}",
                f"{ef['dist_km']:.1f} km",
                f"{dur_m}m {dur_s:02d}s",
                f"{ef['avg_grade']:.1f}%",
                f"{ef['avg_power']:.0f} W",
                f"{ef['max_power']:.0f} W",
                f"{ef['wprime_cost_j']/1000:.1f} kJ" if ef["wprime_cost_j"] > 0 else "—",
            ])

        tbl = ax_tbl.table(
            cellText=rows,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.55)

        # Style header row
        for j in range(len(col_labels)):
            tbl[0, j].set_facecolor("#2c3e50")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Colour rows by avg power zone
        for i, ef in enumerate(climb_efforts, 1):
            z = zone_for_power(ef["avg_power"], cp)
            bg = ZONES[z][3]
            for j in range(len(col_labels)):
                tbl[i, j].set_facecolor(bg)
                tbl[i, j].set_alpha(0.35)

        ax_tbl.set_title(
            "Hardest Sustained Climbs  (ranked by W′ cost)",
            fontsize=10, pad=12,
        )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Zone plot → {output_path}")


# ---------------------------------------------------------------------------
# Garmin course GPX export
# ---------------------------------------------------------------------------

def write_garmin_course_gpx(
    gpx_source_path: str,
    dist_m: np.ndarray,
    power_w: np.ndarray,
    seg_time: np.ndarray,
    grade_pct: np.ndarray,
    climb_efforts: list[dict],
    target_power: float,
    climb_power: float,
    climb_threshold_pct: float,
    descent_threshold_pct: float,
    output_path: str,
) -> int:
    """
    Write a Garmin-compatible GPX course file with course-point waypoints at
    every key pacing transition: climb starts, summits, and descent starts.

    The Edge displays each waypoint name as an alert when you ride past it.
    Names are kept ≤ 25 characters for legibility on the head unit.

    Returns the number of waypoints written.
    """
    import gpxpy
    import gpxpy.gpx

    # ---- Load source track to get lat/lon for every point ----
    with open(gpx_source_path) as f:
        src_gpx = gpxpy.parse(f)

    track_points = []
    for track in src_gpx.tracks:
        for segment in track.segments:
            track_points.extend(segment.points)

    n = min(len(dist_m), len(track_points))

    cum_time = np.cumsum(seg_time)

    # ---- Identify raw pacing label at each segment ----
    raw_labels = []
    for i in range(n):
        g = float(grade_pct[i])
        if g < descent_threshold_pct:
            raw_labels.append("descent")
        elif g >= climb_threshold_pct:
            raw_labels.append("climb")   # gear-limited climbs also count as climb
        else:
            raw_labels.append("flat")

    # ---- Build waypoints from transitions ----
    def _fmt_duration(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m{s:02d}s" if m < 60 else f"{m//60}h{m%60:02d}m"

    def _fmt_power(p):
        return f"{int(round(p))}W"

    # Pre-compute cumulative times for distance-to-duration lookups
    cum_time = np.cumsum(seg_time)

    waypoints = []  # list of (track_point_index, name, symbol)

    # ---- START ----
    # If the first sustained climb effort starts at/near km 0, use its data
    # for the START label so the duration is correct.
    efforts_sorted = sorted(climb_efforts, key=lambda x: x["start_km"])
    opening_effort = efforts_sorted[0] if efforts_sorted and efforts_sorted[0]["start_km"] < 0.5 else None

    if opening_effort:
        ef = opening_effort
        avg_p = ef["avg_power"]
        dur_s = ef["duration_s"]
        max_p = ef["max_power"]
        steep_note = f" peak {_fmt_power(max_p)}" if max_p > climb_power * 1.05 else ""
        name = f"START CLIMB {_fmt_power(avg_p)} ~{_fmt_duration(dur_s)}{steep_note}"
    elif raw_labels[0] == "climb":
        j = 0
        while j < n and raw_labels[j] == "climb":
            j += 1
        dur_s = float(cum_time[min(j, n-1)] - cum_time[0])
        avg_p = float(np.average(power_w[0:min(j,n)], weights=seg_time[0:min(j,n)]))
        name = f"START CLIMB {_fmt_power(avg_p)} ~{_fmt_duration(dur_s)}"
    else:
        name = f"START {_fmt_power(target_power)}"
    waypoints.append((0, name, "Flag, Green"))

    # ---- Waypoints from sustained climb efforts ----
    for ef in efforts_sorted:
        # Skip efforts that start near km 0 — already covered by START above
        if ef["start_km"] < 0.5:
            # Still add the SUMMIT for it
            e_idx = int(np.argmin(np.abs(dist_m / 1000.0 - ef["end_km"])))
            waypoints.append((e_idx, f"SUMMIT | {_fmt_power(target_power)}", "Flag, White"))
            continue
        start_km = ef["start_km"]
        end_km   = ef["end_km"]
        dur_s    = ef["duration_s"]
        avg_p    = ef["avg_power"]
        max_p    = ef["max_power"]

        # Find nearest GPS index to start_km
        s_idx = int(np.argmin(np.abs(dist_m / 1000.0 - start_km)))
        e_idx = int(np.argmin(np.abs(dist_m / 1000.0 - end_km)))

        # Climb start waypoint
        steep_note = f" peak {_fmt_power(max_p)}" if max_p > climb_power * 1.05 else ""
        name = f"CLIMB {_fmt_power(avg_p)} ~{_fmt_duration(dur_s)}{steep_note}"
        waypoints.append((s_idx, name, "Flag, Blue"))

        # Summit / end-of-climb waypoint
        waypoints.append((e_idx, f"SUMMIT | {_fmt_power(target_power)}", "Flag, White"))

    # ---- Major descents (>= 3 min) ----
    MIN_DESCENT_S = 180.0
    in_descent = False
    d_start = 0
    for i in range(n):
        if raw_labels[i] == "descent" and not in_descent:
            in_descent = True
            d_start = i
        elif raw_labels[i] != "descent" and in_descent:
            dur_s = float(cum_time[i-1] - cum_time[d_start])
            if dur_s >= MIN_DESCENT_S:
                name = f"DESCEND coast ~{_fmt_duration(dur_s)}"
                waypoints.append((d_start, name, "Flag, Red"))
            in_descent = False
    if in_descent:
        dur_s = float(cum_time[-1] - cum_time[d_start])
        if dur_s >= MIN_DESCENT_S:
            waypoints.append((d_start, f"DESCEND coast ~{_fmt_duration(dur_s)}", "Flag, Red"))

    # ---- FINISH ----
    waypoints.append((n - 1, "FINISH", "Flag, Green"))

    # Sort by index and deduplicate (keep first if collision)
    waypoints.sort(key=lambda x: x[0])
    seen_idx = set()
    unique_wps = []
    for wp in waypoints:
        if wp[0] not in seen_idx:
            seen_idx.add(wp[0])
            unique_wps.append(wp)
    waypoints = unique_wps

    # Deduplicate (same index might appear twice from adjacent transitions)
    seen = set()
    unique_wps = []
    for wp in waypoints:
        if wp[0] not in seen:
            seen.add(wp[0])
            unique_wps.append(wp)
    waypoints = unique_wps

    # ---- Write output GPX ----
    out_gpx = gpxpy.gpx.GPX()

    # Copy the original track verbatim
    for track in src_gpx.tracks:
        out_gpx.tracks.append(track)

    # Add waypoints
    for (idx, name, sym) in waypoints:
        pt = track_points[idx]
        wpt = gpxpy.gpx.GPXWaypoint(
            latitude=pt.latitude,
            longitude=pt.longitude,
            elevation=pt.elevation,
            name=name,
        )
        # Garmin symbol hint (ignored by non-Garmin parsers)
        wpt.symbol = sym
        # Include km distance as description for reference
        wpt.description = f"{dist_m[idx]/1000:.1f} km"
        out_gpx.waypoints.append(wpt)

    with open(output_path, "w") as f:
        f.write(out_gpx.to_xml())

    return len(waypoints)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--course", required=True, metavar="GPX",
                   help="GPX file for the course to simulate")
    p.add_argument("--activity", metavar="TCX",
                   help="TCX/GPX file with measured power for fitting CdA/Crr")
    p.add_argument("--output", default="course_simulation.png",
                   help="Output plot path (default: course_simulation.png)")
    p.add_argument("--garmin-output", default=None, metavar="GPX",
                   help="Write annotated Garmin course GPX with pacing waypoints "
                        "(default: <output stem>_garmin.gpx)")

    g = p.add_argument_group("rider / physiology")
    g.add_argument("--weight", type=float, default=89.0, metavar="KG",
                   help="Total rider + bike + gear weight (default: 89 kg)")
    g.add_argument("--cp", type=float, default=245.0, metavar="W",
                   help="Critical power / FTP in watts (default: 245 W)")
    g.add_argument("--w-prime", type=float, default=23900.0, metavar="J",
                   help="W′ anaerobic capacity in joules (default: 23900 J)")
    g.add_argument("--target-power", type=float, default=150.0, metavar="W",
                   help="Power target for flat/rolling terrain (default: 150 W)")
    g.add_argument("--climb-power", type=float, default=None, metavar="W",
                   help="Power target for climbs above --climb-threshold "
                        "(default: same as --target-power)")

    g2 = p.add_argument_group("aerodynamics / rolling (override fitting)")
    g2.add_argument("--cda", type=float, default=None, metavar="M2",
                    help="CdA in m² (auto-fit from --activity if not set)")
    g2.add_argument("--crr", type=float, default=None, metavar="COEFF",
                    help="Rolling resistance coefficient (auto-fit if not set)")
    g2.add_argument("--rho", type=float, default=1.2, metavar="KGM3",
                    help="Air density kg/m³ (default: 1.2, slight elevation adjust)")

    g3 = p.add_argument_group("gear constraint")
    g3.add_argument("--gear-ratio", type=float, default=29.0 / 38.0, metavar="RATIO",
                    help="Lowest chainring/cog ratio (default: 29/38 = 0.763)")
    g3.add_argument("--wheel-circ", type=float, default=2.136, metavar="M",
                    help="Wheel circumference in metres (default: 2.136 for 700×40c)")
    g3.add_argument("--min-cadence", type=float, default=50.0, metavar="RPM",
                    help="Minimum sustainable cadence in RPM (default: 50)")
    g3.add_argument("--max-descent-speed", type=float, default=40.0, metavar="KMH",
                    help="Max comfortable descent speed km/h (default: 40)")
    g3.add_argument("--max-cadence", type=float, default=95.0, metavar="RPM",
                    help="Max cadence cap for display (default: 95 RPM — clamps cadence plot)")

    g4 = p.add_argument_group("pacing thresholds")
    g4.add_argument("--climb-threshold", type=float, default=2.0, metavar="PCT",
                    help="Grade %% above which --climb-power applies (default: 2.0)")
    g4.add_argument("--descent-threshold", type=float, default=-2.0, metavar="PCT",
                    help="Grade %% below which to coast (default: -2.0)")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    # ---- Gear constraint ------------------------------------------------
    min_speed = args.gear_ratio * args.min_cadence * args.wheel_circ / 60.0
    max_descent_speed = args.max_descent_speed / 3.6
    print(f"Gear constraint:")
    print(f"  Ratio        : {args.gear_ratio:.4f} ({int(29)}/{int(38)} equivalent)")
    print(f"  Wheel circ   : {args.wheel_circ:.3f} m")
    print(f"  Min cadence  : {args.min_cadence:.0f} RPM")
    print(f"  → Min speed  : {min_speed:.2f} m/s  ({min_speed * 3.6:.1f} km/h)")
    print(f"  → Max descent: {max_descent_speed:.2f} m/s  ({args.max_descent_speed:.0f} km/h)")

    # ---- CdA / Crr -------------------------------------------------------
    if args.cda is not None and args.crr is not None:
        cda, crr = args.cda, args.crr
        print(f"\nUsing manual CdA={cda:.4f} m², Crr={crr:.5f}")
    elif args.activity is not None:
        print(f"\nFitting CdA/Crr from {args.activity} …")
        act_df = load_file(args.activity)
        print(f"  Loaded {len(act_df):,} trackpoints")

        # Braking-bias mitigation:
        # On gravel descents riders brake heavily — speed is lower than physics
        # predicts, which biases Crr upward.  Blank power on descent segments
        # so fit_params ignores those windows entirely.
        dt_, speed_, grade_, accel_ = _compute_kinematics(act_df, 21, 3)
        act_df_fit = act_df.copy()
        descent_mask = grade_ < -2.0
        if "power_w" in act_df_fit.columns:
            n_blanked = int(descent_mask.sum())
            pct_blanked = 100.0 * n_blanked / len(act_df_fit)
            act_df_fit.loc[descent_mask, "power_w"] = np.nan
            print(f"  Blanked {n_blanked:,} descent rows ({pct_blanked:.1f}%) "
                  f"to avoid braking bias in fit")

        fitted = fit_params(act_df_fit, weight_kg=args.weight)
        cda, crr = fitted["cda"], fitted["crr"]

        # Sanity check: CdA > 0.55 m² suggests the joint fit is degenerate
        # (common on slow/short gravel rides where drag << gravity).
        # Fall back to Crr-only fit with a fixed reasonable gravel CdA.
        if cda > 0.55:
            print(f"\n  CdA={cda:.3f} m² looks unrealistically high — joint fit is likely")
            print(f"  degenerate (drag is a small fraction of total power at gravel speeds).")
            print(f"  Falling back to Crr-only fit with fixed CdA={GRAVEL_CDA_DEFAULT:.2f} m²")
            print(f"  (typical gravel/endurance position on the hoods).")
            try:
                fitted_crr = fit_crr_only(act_df_fit, weight_kg=args.weight,
                                          cda=GRAVEL_CDA_DEFAULT, rho=args.rho)
                cda = GRAVEL_CDA_DEFAULT
                crr = fitted_crr["crr"]
            except ValueError as e:
                print(f"  Crr-only fit also failed ({e}); using defaults.")
                cda, crr = GRAVEL_CDA_DEFAULT, 0.0080
    else:
        cda, crr = 0.40, 0.0075
        print(f"\nNo activity file — using defaults CdA={cda}, Crr={crr}")

    print(f"\nPhysics: CdA={cda:.4f} m², Crr={crr:.5f}, ρ={args.rho} kg/m³")

    # ---- Load course -----------------------------------------------------
    print(f"\nLoading course: {args.course}")
    dist_m, elev_m, grade_pct, seg_dist = load_course(args.course)
    dist_km = dist_m / 1000.0
    total_elev_gain = float(np.sum(np.maximum(np.diff(elev_m), 0.0)))

    print(f"  Points       : {len(dist_m):,}")
    print(f"  Total dist   : {dist_km[-1]:.1f} km")
    print(f"  Elev range   : {elev_m.min():.0f}–{elev_m.max():.0f} m")
    print(f"  Elev gain    : {total_elev_gain:.0f} m (smoothed)")
    print(f"  Grade range  : {grade_pct.min():.1f}% to {grade_pct.max():.1f}%")

    # ---- Simulate --------------------------------------------------------
    climb_power = args.climb_power if args.climb_power is not None else args.target_power

    print(f"\nSimulating pacing …")
    print(f"  CP={args.cp:.0f} W | W′={args.w_prime:.0f} J | weight={args.weight:.0f} kg")
    print(f"  Flat/rolling target : {args.target_power:.0f} W "
          f"({args.target_power / args.cp * 100:.0f}% of CP)")
    print(f"  Climb target (>{args.climb_threshold:.0f}%): {climb_power:.0f} W "
          f"({climb_power / args.cp * 100:.0f}% of CP)")
    print(f"  Gear floor: {min_speed * 3.6:.1f} km/h — steeper than this, power floats")

    power_w, speed_ms, wbal_j, seg_time = pacing_strategy(
        grade_pct=grade_pct,
        seg_dist=seg_dist,
        cp=args.cp,
        w_prime=args.w_prime,
        cda=cda,
        crr=crr,
        rho=args.rho,
        weight_kg=args.weight,
        v_headwind=0.0,
        loss_dt=2.0,
        inertia_factor=1.05,
        min_speed=min_speed,
        max_descent_speed=max_descent_speed,
        target_power=args.target_power,
        climb_power=climb_power,
        climb_threshold_pct=args.climb_threshold,
        descent_threshold_pct=args.descent_threshold,
    )

    total_time_s = float(seg_time.sum())
    h = int(total_time_s // 3600)
    m = int((total_time_s % 3600) // 60)
    avg_speed_kmh = float(np.sum(speed_ms * seg_time) / np.sum(seg_time) * 3.6)
    avg_power = float(np.sum(power_w * seg_time) / np.sum(seg_time))
    min_wbal_j = float(wbal_j.min())
    min_wbal_pct = min_wbal_j / args.w_prime * 100.0
    min_idx = int(np.argmin(wbal_j))
    n_forced_above_cp = int((power_w > args.cp * 1.02).sum())

    # Normalized Power, Intensity Factor, TSS
    power_1s = np.repeat(power_w, np.maximum(np.round(seg_time).astype(int), 1))
    rolling_avg = np.convolve(power_1s, np.ones(30) / 30, mode="same")
    np_watts = float((np.mean(rolling_avg ** 4)) ** 0.25)
    if_val = np_watts / args.cp
    tss = (total_time_s / 3600.0) * if_val ** 2 * 100.0

    print(f"\nResults:")
    print(f"  Estimated time  : {h}h {m:02d}m")
    print(f"  Avg power       : {avg_power:.0f} W  ({avg_power/args.cp*100:.0f}% of CP)")
    print(f"  Norm. power     : {np_watts:.0f} W")
    print(f"  Intensity factor: {if_val:.3f}")
    print(f"  TSS             : {tss:.0f}")
    print(f"  Avg speed       : {avg_speed_kmh:.1f} km/h")
    print(f"  Segments > CP   : {n_forced_above_cp:,} "
          f"({100.0 * n_forced_above_cp / len(power_w):.1f}%)")
    print(f"  Min W′ balance  : {min_wbal_j:.0f} J ({min_wbal_pct:.1f}%) "
          f"at {dist_km[min_idx]:.1f} km")

    if min_wbal_j < 0:
        print(f"\n  *** WARNING: W′ fully depleted at {dist_km[min_idx]:.1f} km! ***")
        print(f"  Try reducing --target-power or using lower gearing.")
    elif min_wbal_pct < 10:
        print(f"\n  Very close to W′ depletion — true survival effort.")
    else:
        print(f"\n  Rider survives with {min_wbal_pct:.0f}% W′ remaining at the crux.")

    # ---- Cadence --------------------------------------------------------
    # Cadence in lowest gear: rpm = speed * 60 / (gear_ratio * wheel_circ)
    # Clamped at max_cadence (you shift up on fast sections; min is the gear floor)
    cadence_rpm = np.clip(
        speed_ms * 60.0 / (args.gear_ratio * args.wheel_circ),
        0.0,
        args.max_cadence,
    )
    # Minimum cadence while pedaling (power > 0)
    pedaling_mask = power_w > 0
    if pedaling_mask.any():
        min_cad = float(cadence_rpm[pedaling_mask].min())
        min_cad_km = float(dist_km[pedaling_mask][cadence_rpm[pedaling_mask].argmin()])
        print(f"  Min cadence     : {min_cad:.1f} RPM at {min_cad_km:.1f} km  "
              f"(while pedaling; max capped at {args.max_cadence:.0f} RPM)")

    # Time and avg power at the gear floor (cadence == min_cadence), by contiguous block
    floor_mask = np.abs(cadence_rpm - args.min_cadence) < 0.1
    total_floor_s = float(seg_time[floor_mask].sum())
    if floor_mask.any():
        # Find contiguous runs at the floor
        floor_blocks = []
        in_block = False
        for i in range(len(floor_mask)):
            if floor_mask[i] and not in_block:
                blk_start = i
                in_block = True
            elif not floor_mask[i] and in_block:
                floor_blocks.append((blk_start, i - 1))
                in_block = False
        if in_block:
            floor_blocks.append((blk_start, len(floor_mask) - 1))

        total_m = int(total_floor_s // 60)
        total_s = int(total_floor_s % 60)
        print(f"  Time at {args.min_cadence:.0f} RPM floor : "
              f"{total_m}m {total_s:02d}s total across {len(floor_blocks)} block(s):")
        for s, e in floor_blocks:
            dur = float(seg_time[s:e+1].sum())
            avg_p = float(np.average(power_w[s:e+1], weights=seg_time[s:e+1]))
            dm = int(dur // 60)
            ds = int(dur % 60)
            print(f"    {dist_km[s]:.1f}–{dist_km[e]:.1f} km  "
                  f"{dm}m {ds:02d}s  avg {avg_p:.0f} W")

    # ---- Plot -----------------------------------------------------------
    try:
        import gpxpy
        with open(args.course) as f:
            _gpx = gpxpy.parse(f)
        course_name = _gpx.tracks[0].name if _gpx.tracks else ""
    except Exception:
        course_name = ""

    plot_simulation(
        dist_km=dist_km,
        elev_m=elev_m,
        grade_pct=grade_pct,
        power_w=power_w,
        speed_ms=speed_ms,
        wbal_j=wbal_j,
        cp=args.cp,
        w_prime=args.w_prime,
        min_speed=min_speed,
        max_descent_speed=max_descent_speed,
        total_time_s=total_time_s,
        output_path=args.output,
        course_name=course_name,
        cadence_rpm=cadence_rpm,
        min_cadence=args.min_cadence,
        max_cadence=args.max_cadence,
        gear_ratio=args.gear_ratio,
        wheel_circ=args.wheel_circ,
    )

    # ---- Zone analysis + hardest climbs ----------------------------------
    climb_efforts = find_sustained_climb_efforts(
        dist_km=dist_km,
        grade_pct=grade_pct,
        power_w=power_w,
        seg_time=seg_time,
        cp=args.cp,
        climb_threshold_pct=args.climb_threshold,
    )

    # Print hardest efforts to console as well
    if climb_efforts:
        print(f"\nHardest sustained climbs (ranked by W′ cost):")
        print(f"  {'#':>2}  {'At km':>6}  {'Length':>7}  {'Duration':>9}  "
              f"{'AvgGrade':>8}  {'AvgPwr':>7}  {'PeakPwr':>7}  {'W′Cost':>7}")
        for rank, ef in enumerate(climb_efforts, 1):
            dur_m = int(ef["duration_s"] // 60)
            dur_s = int(ef["duration_s"] % 60)
            wpc = f"{ef['wprime_cost_j']/1000:.1f} kJ" if ef["wprime_cost_j"] > 0 else "—"
            print(f"  {rank:>2}  {ef['start_km']:>6.1f}  {ef['dist_km']:>6.1f}km  "
                  f"{dur_m}m{dur_s:02d}s  "
                  f"{ef['avg_grade']:>7.1f}%  {ef['avg_power']:>6.0f}W  "
                  f"{ef['max_power']:>6.0f}W  {wpc:>7}")

    zones_path = args.output.replace(".png", "_zones.png")
    plot_zone_analysis(
        power_w=power_w,
        seg_time=seg_time,
        dist_km=dist_km,
        grade_pct=grade_pct,
        cp=args.cp,
        w_prime=args.w_prime,
        total_time_s=total_time_s,
        climb_efforts=climb_efforts,
        output_path=zones_path,
        course_name=course_name,
    )

    # ---- Garmin GPX export ----------------------------------------------
    garmin_path = args.garmin_output or args.output.replace(".png", "_garmin.gpx")
    n_wpts = write_garmin_course_gpx(
        gpx_source_path=args.course,
        dist_m=dist_m,
        power_w=power_w,
        seg_time=seg_time,
        grade_pct=grade_pct,
        climb_efforts=climb_efforts,
        target_power=args.target_power,
        climb_power=climb_power,
        climb_threshold_pct=args.climb_threshold,
        descent_threshold_pct=args.descent_threshold,
        output_path=garmin_path,
    )
    print(f"Garmin GPX → {garmin_path}  ({n_wpts} course points)")
    print("  Load onto your Edge via Garmin Connect or copy to the Courses folder on the device.")

    print("\nDone.")


if __name__ == "__main__":
    main()
