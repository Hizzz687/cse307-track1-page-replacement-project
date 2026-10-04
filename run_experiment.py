"""Run the Track 1 experiment and write reproducible CSV/SVG results."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw

from src.page_replacement import run_all_policies

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"


def generate_trace(length: int = 2400, seed: int = 307) -> tuple[list[int], int]:
    rng = random.Random(seed)
    shift = length // 2
    trace: list[int] = []
    for index in range(shift):
        if index % 41 < 5:
            trace.extend(range(0, 8))
        else:
            hot = rng.choice((0, 1, 2, 3, 4, 5))
            trace.append(hot)
        if len(trace) >= shift:
            break
    trace = trace[:shift]
    while len(trace) < shift:
        trace.append(rng.choice(range(0, 8)))

    while len(trace) < length:
        if rng.random() < 0.70:
            burst = rng.sample(range(16, 32), rng.randint(3, 6))
            trace.extend(burst)
        else:
            trace.append(rng.randrange(0, 32))
    return trace[:length], shift


def write_svg(rows: list[dict[str, object]], path: Path) -> None:
    width, height = 900, 520
    left, bottom, chart_w, chart_h = 110, 80, 730, 350
    max_ratio = 1.0
    colors = {"fifo": "#2563eb", "lru": "#16a34a", "optimal": "#7c3aed", "learned": "#ea580c"}
    policies = ["fifo", "lru", "optimal", "learned"]
    segments = ["before_shift", "after_shift"]
    x_positions = [left + 150, left + 510]
    bars: list[str] = []
    for group, segment in enumerate(segments):
        for policy_index, policy in enumerate(policies):
            match = next(row for row in rows if row["policy"] == policy and row["segment"] == segment)
            ratio = float(match["hit_ratio"])
            bar_h = ratio / max_ratio * chart_h
            x = x_positions[group] + policy_index * 65
            y = bottom + chart_h - bar_h
            bars.append(
                f'<rect x="{x}" y="{y:.1f}" width="42" height="{bar_h:.1f}" '
                f'fill="{colors[policy]}"/><text x="{x+21}" y="{bottom+chart_h+20}" '
                f'text-anchor="middle" font-size="11">{policy}</text>'
                f'<text x="{x+21}" y="{y-5:.1f}" text-anchor="middle" font-size="11">'
                f'{ratio:.3f}</text>'
            )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial"
 font-weight="bold">Hit ratio before and after workload shift</text>
<line x1="{left}" y1="{bottom}" x2="{left}" y2="{bottom+chart_h}" stroke="#111"/>
<line x1="{left}" y1="{bottom+chart_h}" x2="{left+chart_w}" y2="{bottom+chart_h}" stroke="#111"/>
<text x="35" y="{bottom+chart_h/2}" transform="rotate(-90 35 {bottom+chart_h/2})"
 font-size="13" font-family="Arial">Hit ratio</text>
<text x="{x_positions[0]+100}" y="{height-25}" text-anchor="middle" font-size="14">Locality / sequential</text>
<text x="{x_positions[1]+100}" y="{height-25}" text-anchor="middle" font-size="14">Random / bursty</text>
{''.join(bars)}
</svg>"""
    path.write_text(svg, encoding="utf-8")


def write_png(rows: list[dict[str, object]], path: Path) -> None:
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    colors = {"fifo": "#2563eb", "lru": "#16a34a", "optimal": "#7c3aed", "learned": "#ea580c"}
    draw.text((250, 18), "Hit ratio before and after workload shift", fill="black")
    left, bottom, chart_w, chart_h = 110, 430, 730, 350
    draw.line((left, bottom - chart_h, left, bottom), fill="black", width=2)
    draw.line((left, bottom, left + chart_w, bottom), fill="black", width=2)
    for group, segment in enumerate(("before_shift", "after_shift")):
        for policy_index, policy in enumerate(("fifo", "lru", "optimal", "learned")):
            match = next(row for row in rows if row["policy"] == policy and row["segment"] == segment)
            ratio = float(match["hit_ratio"])
            x = 260 + group * 360 + policy_index * 65
            top = bottom - int(ratio * chart_h)
            draw.rectangle((x, top, x + 42, bottom), fill=colors[policy])
            draw.text((x, bottom + 8), policy, fill="black")
            draw.text((x, max(45, top - 18)), f"{ratio:.3f}", fill="black")
    draw.text((270, 480), "Locality / sequential", fill="black")
    draw.text((620, 480), "Random / bursty", fill="black")
    image.save(path)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    trace, split = generate_trace()
    training_trace, _ = generate_trace(seed=308)
    policies = run_all_policies(trace, capacity=8, training_trace=training_trace)
    rows: list[dict[str, object]] = []
    for policy, result in policies.items():
        for segment, start, end in (("before_shift", 0, split), ("after_shift", split, len(trace))):
            metrics = result.segment(start, end)
            rows.append({"policy": policy, "segment": segment, **metrics})
    with (RESULTS / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    write_svg(rows, RESULTS / "hit_ratio.svg")
    write_png(rows, RESULTS / "hit_ratio.png")
    (RESULTS / "trace_summary.txt").write_text(
        f"seed=307\nrequests={len(trace)}\nshift_index={split}\ncache_capacity=8\nunique_pages=32\n",
        encoding="utf-8",
    )
    print("Experiment complete.")
    for row in rows:
        print(
            f"{row['policy']:>7} {row['segment']:<12} "
            f"faults={row['faults']:>4} hit_ratio={row['hit_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
