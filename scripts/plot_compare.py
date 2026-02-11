import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path):
    xs = []
    bi = []
    zi = []
    ratio = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["n"]))
            bi.append(float(row["bi_us"]))
            zi.append(float(row["zint_us"]))
            ratio.append(float(row["zint_over_bi"]))
    return xs, bi, zi, ratio


def plot_one(csv_path: Path, out_png: Path, title: str, xlabel: str):
    xs, bi, zi, ratio = read_csv(csv_path)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True)

    ax0.plot(xs, bi, marker="o", label="baseline")
    ax0.plot(xs, zi, marker="o", label="zint")
    ax0.set_xscale("log", base=2)
    ax0.set_ylabel("time (us)")
    ax0.grid(True, which="both", linestyle="--", alpha=0.3)
    ax0.set_title(title)
    ax0.legend()

    ax1.plot(xs, ratio, marker="o", color="black")
    ax1.axhline(1.0, color="red", linewidth=1, alpha=0.7)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("zint / baseline")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True, help="csv path")
    ap.add_argument("--out", type=Path, required=True, help="png path")
    ap.add_argument("--title", type=str, required=True, help="plot title")
    ap.add_argument("--xlabel", type=str, default="n (u64 limbs)", help="x axis label")
    args = ap.parse_args()

    plot_one(args.csv, args.out, args.title, args.xlabel)


if __name__ == "__main__":
    main()

