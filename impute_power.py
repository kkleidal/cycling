#!/usr/bin/env python3
"""
Impute missing power data from GPX or FIT files using a physics model.

The model estimates rider power output from speed, gradient, and acceleration,
accounting for gravity, rolling resistance, aerodynamic drag, and inertia.

When the file contains a measured-power section, --auto-fit will calibrate CdA
and Crr (and optionally effective headwind) against that data before imputing
the missing section.

Usage
-----
    # Manual parameters
    python impute_power.py ride.fit --weight 80 --cda 0.36
    # Auto-fit from measured section (only weight required)
    python impute_power.py ride.fit --weight 80 --auto-fit
    python impute_power.py ride.fit --weight 80 --auto-fit --fit-headwind
"""

import argparse
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import jax.numpy as jnp

from power_model import f_gravity, f_rolling, f_drag

DEFAULT_PARAMS = dict(
    weight_kg=82.5,      # rider + bike, kg
    cda=0.389,           # drag coefficient × frontal area, m²
    crr=0.005,           # rolling resistance coefficient (road ~0.004-0.006, gravel ~0.007-0.010)
    rho=1.225,           # air density, kg/m³  (sea-level standard)
    loss_drivetrain=2.0, # drivetrain loss, %
    v_headwind=0.0,      # headwind, m/s (positive = into the wind)
    inertia_factor=1.05, # effective mass multiplier for rotating parts
)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _savgol_safe(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = len(arr)
    win = min(window, n)
    if win % 2 == 0:
        win -= 1
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)
    if win > n:
        return arr.copy()
    return savgol_filter(arr, win, poly)


# ---------------------------------------------------------------------------
# Kinematics — shared by compute_model_power and fit_params
# ---------------------------------------------------------------------------

def _compute_kinematics(
    df: pd.DataFrame,
    smooth_window: int,
    smooth_poly: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (dt, speed, grade_pct, accel) as numpy arrays after smoothing.
    """
    n = len(df)

    # time step
    if "time" in df.columns:
        dt = df["time"].diff().dt.total_seconds().values.astype(float)
        dt[0] = dt[1] if n > 1 else 1.0
    else:
        dt = np.ones(n)
    dt = np.maximum(dt, 0.1)

    # raw speed
    if "speed_ms" in df.columns:
        speed_raw = df["speed_ms"].fillna(0).values.astype(float)
    elif "distance_km" in df.columns:
        dist_diff = np.diff(
            df["distance_km"].values, prepend=df["distance_km"].values[0]
        ) * 1000.0
        speed_raw = np.maximum(dist_diff / dt, 0.0)
    else:
        raise ValueError("DataFrame must contain 'speed_ms' or 'distance_km'.")

    speed = _savgol_safe(speed_raw, smooth_window, smooth_poly)
    speed = np.maximum(speed, 0.0)

    # elevation → grade
    if "elevation_m" in df.columns:
        elev_raw = df["elevation_m"].ffill().bfill().values.astype(float)
        elev = _savgol_safe(elev_raw, smooth_window, smooth_poly)
    else:
        elev = np.zeros(n)

    horiz = speed * dt
    elev_diff = np.diff(elev, prepend=elev[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        grade = np.where(horiz > 0.1, (elev_diff / horiz) * 100.0, 0.0)
    grade = np.clip(grade, -40.0, 40.0)
    grade = _savgol_safe(grade, smooth_window, smooth_poly)

    # acceleration
    t_cumul = np.cumsum(dt)
    accel = np.gradient(speed, t_cumul)

    return dt, speed, grade, accel


# ---------------------------------------------------------------------------
# Physics model (pure numpy — used inside the optimizer)
# ---------------------------------------------------------------------------

def _power_np(
    grade: np.ndarray,
    speed: np.ndarray,
    accel: np.ndarray,
    cda: float,
    crr: float,
    rho: float,
    weight_kg: float,
    v_headwind: float,
    loss_drivetrain: float,
    inertia_factor: float,
) -> np.ndarray:
    """Unclamped model power (W) — suitable for use inside optimizer."""
    fg = 9.8067 * np.sin(np.arctan(grade / 100.0)) * weight_kg
    fr = 9.8067 * np.cos(np.arctan(grade / 100.0)) * weight_kg * crr
    v_air = v_headwind + speed
    fd = 0.5 * cda * rho * v_air ** 2
    fi = weight_kg * inertia_factor * accel
    eta = 1.0 - loss_drivetrain / 100.0
    return (fg + fr + fd + fi) * speed / eta


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

def fit_params(
    df: pd.DataFrame,
    weight_kg: float,
    fixed_params: dict | None = None,
    fit_headwind: bool = False,
    smooth_window: int = 21,
    smooth_poly: int = 3,
    min_power_w: float = 30.0,
    max_accel: float = 0.5,
    window_seconds: int = 30,
    min_active_fraction: float = 0.7,
) -> dict:
    """
    Fit CdA and Crr (and optionally v_headwind) to the measured power section.

    Calibration is performed on *time-averaged windows* (default 30 s) rather
    than individual samples.  This averages out the variability in rider effort
    that makes per-sample fits noisy, and retains the terrain-driven signal.

    Parameters
    ----------
    df : DataFrame with ``power_w`` column (NaN where unmeasured)
    weight_kg : rider + bike mass — not fitted, must be supplied
    fixed_params : override any DEFAULT_PARAMS values that should stay fixed
    fit_headwind : also fit an effective headwind term
    min_power_w : exclude windows whose avg power is below this (coasting)
    max_accel : exclude windows with avg |acceleration| above this
    window_seconds : averaging window length in seconds
    min_active_fraction : fraction of window that must have power > 0

    Returns
    -------
    dict of fitted params, suitable for passing to ``impute_power``
    """
    base = {**DEFAULT_PARAMS, "weight_kg": weight_kg, **(fixed_params or {})}

    if "power_w" not in df.columns or df["power_w"].isna().all():
        raise ValueError("No measured power data available for fitting.")

    dt, speed, grade, accel = _compute_kinematics(df, smooth_window, smooth_poly)

    df_work = df.copy().reset_index(drop=True)
    df_work["_speed"] = speed
    df_work["_grade"] = grade
    df_work["_accel"] = accel

    # Only consider the measured section (after leading NaNs)
    meas_start = df_work["power_w"].notna().values.argmax()
    meas = df_work.iloc[meas_start:].copy().reset_index(drop=True)

    # Build non-overlapping time windows
    windows = []
    i = 0
    while i + window_seconds <= len(meas):
        chunk = meas.iloc[i : i + window_seconds]
        active = (chunk["power_w"].fillna(0) > 0).mean()
        avg_accel = chunk["_accel"].abs().mean()
        avg_power = chunk["power_w"].mean()

        if active >= min_active_fraction and avg_accel <= max_accel and avg_power >= min_power_w:
            windows.append({
                "grade": chunk["_grade"].mean(),
                "speed": chunk["_speed"].mean(),
                "accel": chunk["_accel"].mean(),
                "power": avg_power,
            })
        i += window_seconds

    n_win = len(windows)
    if n_win < 5:
        raise ValueError(
            f"Only {n_win} usable {window_seconds}-s windows (need ≥ 5). "
            "Try reducing --window or check the file has sustained power data."
        )

    wins = pd.DataFrame(windows)
    g_fit = wins["grade"].values
    v_fit = wins["speed"].values
    a_fit = wins["accel"].values
    p_meas = wins["power"].values

    rho = base["rho"]
    eta_loss = base["loss_drivetrain"]
    inertia = base["inertia_factor"]

    def mse(r):
        return np.mean(r ** 2)

    if fit_headwind:
        x0 = np.array([np.log(base["cda"]), np.log(base["crr"]), base["v_headwind"]])

        def objective(x):
            cda, crr, vw = np.exp(x[0]), np.exp(x[1]), x[2]
            p_hat = _power_np(g_fit, v_fit, a_fit, cda, crr, rho,
                              weight_kg, vw, eta_loss, inertia)
            return mse(p_hat - p_meas)

        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=[
                           (np.log(0.15), np.log(0.8)),   # CdA: 0.15–0.8 m²
                           (np.log(0.002), np.log(0.015)), # Crr: 0.002–0.015
                           (-10, 10),
                       ])
        cda_fit = float(np.exp(res.x[0]))
        crr_fit = float(np.exp(res.x[1]))
        vw_fit  = float(res.x[2])
    else:
        x0 = np.array([np.log(base["cda"]), np.log(base["crr"])])
        vw_fit = base["v_headwind"]

        def objective(x):
            cda, crr = np.exp(x[0]), np.exp(x[1])
            p_hat = _power_np(g_fit, v_fit, a_fit, cda, crr, rho,
                              weight_kg, vw_fit, eta_loss, inertia)
            return mse(p_hat - p_meas)

        res = minimize(objective, x0, method="L-BFGS-B",
                       bounds=[
                           (np.log(0.15), np.log(0.8)),    # CdA: 0.15–0.8 m²
                           (np.log(0.002), np.log(0.015)), # Crr: 0.002–0.015
                       ])
        cda_fit = float(np.exp(res.x[0]))
        crr_fit = float(np.exp(res.x[1]))

    fitted = {**base, "cda": cda_fit, "crr": crr_fit, "v_headwind": vw_fit}

    # Goodness-of-fit on the calibration windows
    p_hat_cal = _power_np(g_fit, v_fit, a_fit,
                          cda_fit, crr_fit, rho, weight_kg,
                          vw_fit, eta_loss, inertia)
    rmse = float(np.sqrt(np.mean((p_hat_cal - p_meas) ** 2)))
    ss_res = np.sum((p_hat_cal - p_meas) ** 2)
    ss_tot = np.sum((p_meas - p_meas.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nFitted parameters ({n_win} × {window_seconds}-s calibration windows):")
    print(f"  CdA          : {cda_fit:.4f} m²")
    print(f"  Crr          : {crr_fit:.5f}")
    if fit_headwind:
        print(f"  Headwind     : {vw_fit:+.2f} m/s")
    print(f"  Fit RMSE     : {rmse:.1f} W  (window averages)")
    print(f"  Fit R²       : {r2:.3f}  (window averages)")

    return fitted


# ---------------------------------------------------------------------------
# Core physics (JAX path — full-ride model power)
# ---------------------------------------------------------------------------

def compute_model_power(
    df: pd.DataFrame,
    params: dict,
    smooth_window: int = 21,
    smooth_poly: int = 3,
) -> pd.DataFrame:
    """
    Add physics-model power estimates to *df*.

    Added columns: ``speed_smooth_ms``, ``grade_pct``, ``accel_ms2``,
    ``power_model_w``
    """
    p = {**DEFAULT_PARAMS, **params}
    df = df.copy().reset_index(drop=True)

    dt, speed, grade, accel = _compute_kinematics(df, smooth_window, smooth_poly)

    g_jnp = jnp.array(grade)
    v_jnp = jnp.array(speed)

    fg = f_gravity(grade=g_jnp, weight_kg=p["weight_kg"])
    fr = f_rolling(grade=g_jnp, weight_kg=p["weight_kg"], c_rr=p["crr"])
    fd = f_drag(
        v_headwind=float(p["v_headwind"]),
        v_groundspeed=v_jnp,
        cda=p["cda"],
        rho=p["rho"],
    )
    fi = jnp.array(p["weight_kg"] * p["inertia_factor"] * accel)

    p_wheel = (fg + fr + fd + fi) * v_jnp
    eta = 1.0 - p["loss_drivetrain"] / 100.0
    p_rider = np.maximum(np.array(p_wheel) / eta, 0.0)

    df["speed_smooth_ms"] = speed
    df["grade_pct"] = grade
    df["accel_ms2"] = accel
    df["power_model_w"] = p_rider
    return df


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

def impute_power(
    df: pd.DataFrame,
    params: dict | None = None,
    smooth_window: int = 21,
    smooth_poly: int = 3,
    treat_leading_zeros_as_missing: bool = True,
    auto_fit: bool = False,
    fit_headwind: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing power values with physics-model estimates.

    Parameters
    ----------
    auto_fit : if True, fit CdA/Crr from the measured power section first.
               ``params`` must contain ``weight_kg``.
    fit_headwind : (only with auto_fit) also fit an effective headwind.

    Returns
    -------
    (df_out, params_used)
        df_out has columns: speed_smooth_ms, grade_pct, accel_ms2,
        power_model_w, power_source, power_imputed_w
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    if auto_fit:
        p = fit_params(
            df,
            weight_kg=p["weight_kg"],
            fixed_params={k: v for k, v in p.items() if k not in ("cda", "crr", "v_headwind")},
            fit_headwind=fit_headwind,
            smooth_window=smooth_window,
            smooth_poly=smooth_poly,
        )

    df = compute_model_power(df, p, smooth_window, smooth_poly)

    if "power_w" not in df.columns:
        df["power_w"] = np.nan

    missing = df["power_w"].isna()

    if treat_leading_zeros_as_missing:
        valid = (~missing) & (df["power_w"] > 0)
        if valid.any():
            first_valid = int(valid.values.argmax())
            missing = missing.copy()
            missing.iloc[:first_valid] = True

    df["power_source"] = np.where(missing, "model", "measured")
    df["power_imputed_w"] = np.where(missing, df["power_model_w"], df["power_w"])
    return df, p


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_imputed_power(df: pd.DataFrame, output_path: str, smooth_window: int = 30) -> None:
    """
    Save a figure showing physics-model power vs measured power over the full
    ride, with elevation as a gray background fill.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = df.copy()
    df["model_smooth"]   = df["power_model_w"].rolling(smooth_window, center=True, min_periods=1).mean()
    df["measured_smooth"] = df["power_w"].rolling(smooth_window, center=True, min_periods=1).mean()

    first_meas_idx = (df["power_source"] == "measured").values.argmax()
    transition_time = df.loc[first_meas_idx, "time"]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax2 = ax.twinx()
    ax2.fill_between(df["time"], df["elevation_m"], alpha=0.18, color="gray", zorder=0)
    ax2.set_ylabel("Elevation (m)", color="gray", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="gray")
    ax2.set_ylim(0, df["elevation_m"].max() * 3.5)

    ax.plot(df["time"], df["model_smooth"], color="tomato", lw=1.3, alpha=0.85,
            label=f"Physics model ({smooth_window} s avg)", zorder=3)
    ax.plot(df["time"], df["measured_smooth"], color="steelblue", lw=1.3, alpha=0.85,
            label=f"Measured power ({smooth_window} s avg)", zorder=4)

    ax.axvline(transition_time, color="black", lw=1.2, ls="--", alpha=0.6, zorder=5)
    ax.text(transition_time, ax.get_ylim()[1] if ax.get_ylim()[1] > 10 else 500,
            " meter\n on", fontsize=8, va="top", color="black", alpha=0.7)
    ax.axvspan(df["time"].iloc[0], transition_time, alpha=0.07, color="tomato",
               zorder=2, label="Imputed section")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Power (W)", fontsize=11)
    ax.set_title("Physics model vs measured power — full ride", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 90, 5)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.legend(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1],
              fontsize=10, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".gpx"):
        from gpx_io import parse_gpx_to_dataframe
        return parse_gpx_to_dataframe(path)
    elif lower.endswith(".tcx"):
        from tcx_io import parse_tcx_to_dataframe
        return parse_tcx_to_dataframe(path)
    else:
        raise ValueError(f"Unsupported file format (expected .fit, .gpx, or .tcx): {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", help="Input .fit or .gpx file")
    p.add_argument("output", nargs="?",
                   help="Output CSV path (default: <input>_imputed.csv)")

    g = p.add_argument_group("physics parameters")
    g.add_argument("--weight", type=float, default=DEFAULT_PARAMS["weight_kg"],
                   metavar="KG",
                   help=f"Rider + bike weight in kg (default: {DEFAULT_PARAMS['weight_kg']})")
    g.add_argument("--cda", type=float, default=None, metavar="M2",
                   help=f"CdA in m² — ignored when --auto-fit (default: {DEFAULT_PARAMS['cda']})")
    g.add_argument("--crr", type=float, default=None, metavar="COEFF",
                   help=f"Crr — ignored when --auto-fit (default: {DEFAULT_PARAMS['crr']})")
    g.add_argument("--rho", type=float, default=DEFAULT_PARAMS["rho"], metavar="KGM3",
                   help=f"Air density kg/m³ (default: {DEFAULT_PARAMS['rho']})")
    g.add_argument("--headwind", type=float, default=None, metavar="MS",
                   help=f"Headwind m/s — ignored when --auto-fit --fit-headwind "
                        f"(default: {DEFAULT_PARAMS['v_headwind']})")
    g.add_argument("--drivetrain-loss", type=float, default=DEFAULT_PARAMS["loss_drivetrain"],
                   metavar="PCT",
                   help=f"Drivetrain loss %% (default: {DEFAULT_PARAMS['loss_drivetrain']})")

    g2 = p.add_argument_group("auto-fitting")
    g2.add_argument("--auto-fit", action="store_true",
                    help="Fit CdA and Crr from the measured power section")
    g2.add_argument("--fit-headwind", action="store_true",
                    help="Also fit an effective headwind term (requires --auto-fit)")

    g3 = p.add_argument_group("smoothing / misc")
    g3.add_argument("--smooth-window", type=int, default=21, metavar="N",
                    help="Savitzky-Golay window (default: 21 samples)")
    g3.add_argument("--no-treat-zeros", action="store_true",
                    help="Do not treat leading zeros in power as missing")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    params = dict(
        weight_kg=args.weight,
        cda=args.cda if args.cda is not None else DEFAULT_PARAMS["cda"],
        crr=args.crr if args.crr is not None else DEFAULT_PARAMS["crr"],
        rho=args.rho,
        v_headwind=args.headwind if args.headwind is not None else DEFAULT_PARAMS["v_headwind"],
        loss_drivetrain=args.drivetrain_loss,
    )

    print(f"Loading {args.input} …")
    df = load_file(args.input)
    print(f"  {len(df):,} data points loaded")

    if args.auto_fit:
        print("Auto-fitting CdA/Crr from measured power section …")
    else:
        print(f"Using CdA={params['cda']:.3f} m², Crr={params['crr']:.4f}")

    df, params_used = impute_power(
        df,
        params=params,
        smooth_window=args.smooth_window,
        treat_leading_zeros_as_missing=not args.no_treat_zeros,
        auto_fit=args.auto_fit,
        fit_headwind=args.fit_headwind,
    )

    if not args.auto_fit:
        print(f"\nParameters used:")
        print(f"  Weight       : {params_used['weight_kg']:.1f} kg")
        print(f"  CdA          : {params_used['cda']:.4f} m²")
        print(f"  Crr          : {params_used['crr']:.5f}")
        print(f"  Headwind     : {params_used['v_headwind']:+.2f} m/s")

    total = len(df)
    n_model = int((df["power_source"] == "model").sum())
    n_meas  = int((df["power_source"] == "measured").sum())

    print(f"\nResults:")
    print(f"  Total samples  : {total:,}")
    print(f"  Measured power : {n_meas:,}  ({100 * n_meas / total:.1f}%)")
    print(f"  Imputed power  : {n_model:,}  ({100 * n_model / total:.1f}%)")
    if n_model:
        avg_mod = df.loc[df["power_source"] == "model", "power_imputed_w"].mean()
        print(f"  Avg imputed    : {avg_mod:.0f} W")
    if n_meas:
        avg_meas = df.loc[df["power_source"] == "measured", "power_imputed_w"].mean()
        print(f"  Avg measured   : {avg_meas:.0f} W")
    print(f"  Avg overall    : {df['power_imputed_w'].mean():.0f} W")

    stem = args.input.rsplit(".", 1)[0]
    input_ext = args.input.rsplit(".", 1)[-1].lower()

    # --- CSV ---
    csv_path = args.output or stem + "_imputed.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV  → {csv_path}")

    # --- Plot ---
    png_path = stem + "_imputed.png"
    if "elevation_m" in df.columns and "power_model_w" in df.columns:
        plot_imputed_power(df, png_path)
        print(f"Plot → {png_path}")
    else:
        print("Plot skipped (missing elevation or model power column)")

    # --- TCX (only when input was a TCX file) ---
    if input_ext == "tcx":
        from tcx_io import write_tcx_with_imputed_power
        tcx_path = stem + "_imputed.tcx"
        n_patched = write_tcx_with_imputed_power(args.input, tcx_path, df)
        print(f"TCX  → {tcx_path}  ({n_patched} trackpoints patched)")


if __name__ == "__main__":
    main()
