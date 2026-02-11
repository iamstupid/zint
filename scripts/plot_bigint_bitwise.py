import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    rows = []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                (
                    int(row["n"]),
                    row["case"],
                    row["op"],
                    float(row["us"]),
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = read_rows(args.csv)

    # case -> op -> list[(n, us)]
    series = defaultdict(lambda: defaultdict(list))
    for n, scase, op, us in rows:
        series[scase][op].append((n, us))

    cases = sorted(series.keys())
    if not cases:
        raise SystemExit("no data")

    fig, axes = plt.subplots(len(cases), 1, figsize=(8, 3.5 * len(cases)), dpi=150, sharex=True)
    if len(cases) == 1:
        axes = [axes]

    for ax, scase in zip(axes, cases):
        ax.set_title(f"bigint bitwise ops (case={scase})")
        for op, pts in sorted(series[scase].items()):
            pts.sort(key=lambda x: x[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, marker="o", label=op)

        ax.set_xscale("log", base=2)
        ax.set_ylabel("time (us)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("n (u64 limbs)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)


if __name__ == "__main__":
    main()

