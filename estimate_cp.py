#!/usr/bin/env python3
"""Estimate W' (anaerobic work capacity) from a race FIT file.

Uses Skiba's differential W' balance model with second-by-second power data.
Sweeps W' to find the value where the balance just reaches 0 (never negative)
at the point of maximum depletion — calibrated to a race where the rider was
fully spent.

Usage:
    python3 estimate_cp.py <fit_file> [--cp <watts>]

Example:
    python3 estimate_cp.py ~/Downloads/i137974391.fit --cp 245

Outputs:
    - CP and W' estimates printed to stdout
    - wbal_simulation.png: plot of power and W' balance over time
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fitparse import FitFile


def load_power_from_fit(path):
    """Extract second-by-second power from a FIT file."""
    fitfile = FitFile(path)
    powers = []
    for record in fitfile.get_messages("record"):
        power = record.get_value("power")
        powers.append(power if power is not None else 0)
    return np.array(powers, dtype=float)


def simulate_wbal(w_prime, cp, powers):
    """Simulate W' balance using Skiba's differential model, second by second.

    When P > CP: W'bal decreases by (P - CP) per second
    When P <= CP: W'bal recovers exponentially toward W'max
        dW'/dt = (W'max - W'bal) * (CP - P) / W'max
    """
    n = len(powers)
    wbal = np.empty(n)
    wbal[0] = w_prime

    for i in range(1, n):
        p = powers[i]
        if p > cp:
            wbal[i] = wbal[i-1] - (p - cp)
        else:
            wbal[i] = wbal[i-1] + (w_prime - wbal[i-1]) * (cp - p) / w_prime

    return wbal


def find_wprime(cp, powers, low=5000, high=40000, step=100):
    """Sweep W' to find the value where min balance = 0 (never negative)."""
    best_wp = None
    best_diff = float("inf")
    for wp_test in range(low, high, step):
        wbal = simulate_wbal(wp_test, cp, powers)
        min_wbal = wbal.min()
        if abs(min_wbal) < best_diff:
            best_diff = abs(min_wbal)
            best_wp = wp_test
    return best_wp


def main():
    parser = argparse.ArgumentParser(description="Estimate W' from a race FIT file")
    parser.add_argument("fit_file", help="Path to .fit file from a maximal effort/race")
    parser.add_argument("--cp", type=int, default=245, help="Critical Power in watts (default: 245)")
    args = parser.parse_args()

    powers = load_power_from_fit(args.fit_file)
    cp = args.cp
    print(f"Loaded {len(powers)} seconds of power data")
    print(f"Avg: {powers.mean():.0f}W, Max: {powers.max():.0f}W")

    best_wp = find_wprime(cp, powers)
    wbal = simulate_wbal(best_wp, cp, powers)
    min_idx = np.argmin(wbal)

    print(f"\nCP = {cp} W")
    print(f"W' = {best_wp} J ({best_wp/1000:.1f} kJ)")
    print(f"Minimum W' balance occurs at t = {min_idx}s ({min_idx/60:.1f} min)")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    times = np.arange(len(powers)) / 60

    # Power
    axes[0].plot(times, powers, color="gray", lw=0.5, alpha=0.5)
    kernel = np.ones(10) / 10
    powers_smooth = np.convolve(powers, kernel, mode="same")
    axes[0].plot(times, powers_smooth, color="steelblue", lw=1)
    axes[0].axhline(cp, color="red", ls="--", alpha=0.5, label=f"CP = {cp} W")
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # W' balance as percentage
    pct = wbal / best_wp * 100
    axes[1].fill_between(times, pct, 0, alpha=0.3, color="steelblue")
    axes[1].plot(times, pct, color="steelblue", lw=1)
    axes[1].axhline(0, color="red", ls="--", alpha=0.5)
    axes[1].set_ylabel("W' Balance (%)")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylim(min(pct.min() - 5, -10), 105)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"W' Balance — CP={cp}W, W'={best_wp/1000:.1f}kJ", fontsize=13)
    plt.tight_layout()
    plt.savefig("wbal_simulation.png", dpi=150)
    print(f"\nPlot saved to wbal_simulation.png")


if __name__ == "__main__":
    main()
