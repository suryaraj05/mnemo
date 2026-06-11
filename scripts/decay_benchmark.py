#!/usr/bin/env python3
"""Compare exponential vs power-law decay weights (Phase 9 benchmark)."""

from __future__ import annotations

import argparse

from mnemo.decay import weight_exponential, weight_power_law


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decay curves for Mnemo policy tuning.")
    parser.add_argument("--half-life", type=float, default=30.0, help="Exponential half-life (days)")
    parser.add_argument("--tau", type=float, default=7.0, help="Power-law scale τ (days)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Power-law exponent α")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib chart if installed")
    args = parser.parse_args()

    days = [0, 1, 5, 10, 30, 90, 180]
    print(f"half_life={args.half_life}d  tau={args.tau}d  alpha={args.alpha}")
    print(f"{'days':>6}  {'exponential':>12}  {'power_law':>12}")
    exp_vals: list[float] = []
    pl_vals: list[float] = []
    for t in days:
        exp_w = weight_exponential(float(t), args.half_life)
        pl_w = weight_power_law(float(t), args.tau, args.alpha)
        exp_vals.append(exp_w)
        pl_vals.append(pl_w)
        print(f"{t:6d}  {exp_w:12.4f}  {pl_w:12.4f}")

    at_t10 = weight_exponential(10.0, args.half_life) - weight_power_law(10.0, args.tau, args.alpha)
    winner = "exponential" if at_t10 > 0 else "power_law"
    print(f"\nAt t=10d, {winner} retains a higher weight (delta={abs(at_t10):.4f}).")

    if not args.plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to use --plot.")
        return

    grid = [i * 0.5 for i in range(0, 361)]
    exp_curve = [weight_exponential(t, args.half_life) for t in grid]
    pl_curve = [weight_power_law(t, args.tau, args.alpha) for t in grid]

    plt.figure(figsize=(8, 4))
    plt.plot(grid, exp_curve, label=f"exponential (half-life={args.half_life}d)")
    plt.plot(grid, pl_curve, label=f"power-law (τ={args.tau}, α={args.alpha})")
    plt.xlabel("age (days)")
    plt.ylabel("retrieval weight")
    plt.title("Mnemo temporal decay curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
