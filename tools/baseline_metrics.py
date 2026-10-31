#!/usr/bin/env python3
"""Cross-character baseline table: absolute sizes and scale-free proportions.

Reads every out/baseline/<char>/report.json (no Blender needed) and
writes:
    out/baseline/baseline_metrics.json   full table
    out/baseline/baseline_metrics.csv    same, spreadsheet friendly
    out/baseline/baseline_metrics.md     human summary with the range

Why proportions: the 12 heads differ in absolute size, so a single-character
landmark contract cannot generalise. Ratios (eye separation / head width, nose
projection / head depth, ...) are what the parametric face generator has to
reproduce; the spread across the 12 defines the target distribution.

    python3 tools/baseline_metrics.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out" / "modeler-baseline"

FIELDS = [
    ("character", "角色"),
    ("front_tris", "前头三角面"),
    ("watertight", "水密"),
    ("width_mm", "头宽 x"),
    ("depth_mm", "头深 y"),
    ("height_mm", "头高 z"),
    ("eye_sep_mm", "眼孔间距"),
    ("eye_w_mm", "眼孔宽"),
    ("eye_h_mm", "眼孔高"),
    ("eye_z_mm", "眼孔中心高"),
    ("eye_sep_over_w", "眼距/头宽"),
    ("eye_w_over_w", "眼宽/头宽"),
    ("eye_h_over_h", "眼高/头高"),
    ("eye_z_over_h", "眼心高/头高"),
    ("depth_over_w", "头深/头宽"),
    ("width_over_h", "头宽/头高"),
]


def hole_box(hole: dict) -> tuple[float, float]:
    (x0, z0), (x1, z1) = hole["bbox_mm"]
    return abs(x1 - x0), abs(z1 - z0)


def row_from(report: dict) -> dict:
    raw = (report.get("raw_stats") or {}).get("front") or {}
    front = (report.get("front_stats") or [{}])[0]
    dims = raw.get("dims") or front.get("dims") or [None, None, None]
    w, d, h = (dims + [None, None, None])[:3]
    holes = sorted(report.get("eye_holes") or [], key=lambda x: x["centre_mm"][0])
    row = {
        "character": report.get("character"),
        "front_tris": raw.get("triangles", front.get("tris")),
        "watertight": raw.get("watertight"),
        "width_mm": w,
        "depth_mm": d,
        "height_mm": h,
        "eye_sep_mm": None,
        "eye_w_mm": None,
        "eye_h_mm": None,
        "eye_z_mm": None,
    }
    if len(holes) == 2:
        a, b = holes[0]["centre_mm"], holes[1]["centre_mm"]
        row["eye_sep_mm"] = round(abs(b[0] - a[0]), 2)
        row["eye_z_mm"] = round((a[1] + b[1]) / 2, 2)
        wa, ha = hole_box(holes[0])
        wb, hb = hole_box(holes[1])
        row["eye_w_mm"] = round((wa + wb) / 2, 2)
        row["eye_h_mm"] = round((ha + hb) / 2, 2)
    for num, den, key in [
        ("eye_sep_mm", "width_mm", "eye_sep_over_w"),
        ("eye_w_mm", "width_mm", "eye_w_over_w"),
        ("eye_h_mm", "height_mm", "eye_h_over_h"),
        ("eye_z_mm", "height_mm", "eye_z_over_h"),
        ("depth_mm", "width_mm", "depth_over_w"),
        ("width_mm", "height_mm", "width_over_h"),
    ]:
        try:
            row[key] = round(row[num] / row[den], 4)
        except (TypeError, ZeroDivisionError):
            row[key] = None
    return row


def main() -> int:
    rows = []
    for d in sorted(p for p in BASE.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rp = d / "report.json"
        if not rp.exists():
            continue
        report = json.loads(rp.read_text(encoding="utf-8"))
        if report.get("status") != "ok":
            continue
        rows.append(row_from(report))
    if not rows:
        print("no ok reports")
        return 1

    # sparkline-ish summary per ratio: min / median / max across characters
    summary = {}
    for key in ("eye_sep_over_w", "eye_w_over_w", "eye_h_over_h", "eye_z_over_h", "depth_over_w", "width_over_h"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            summary[key] = {
                "min": round(min(vals), 4),
                "median": round(statistics.median(vals), 4),
                "max": round(max(vals), 4),
                "spread": round(max(vals) - min(vals), 4),
            }
    payload = {"schema": "headshell/baseline-metrics@1", "characters": rows, "ratio_summary": summary}
    (BASE / "baseline_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with open(BASE / "baseline_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[k for k, _ in FIELDS])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k, _ in FIELDS})

    def cell(v):
        return "—" if v is None else v

    lines = ["| 角色 | 前头三角面 | 水密 | 头宽 | 头深 | 头高 | 眼距 | 眼宽 | 眼高 | 眼距/头宽 |", "|---|---:|:--:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            "| {character} | {tris} | {tight} | {w} | {d} | {h} | {sep} | {ew} | {eh} | {ratio} |".format(
                character=r["character"],
                tris=f'{r["front_tris"]:,}' if r.get("front_tris") else "—",
                tight="是" if r.get("watertight") else "否",
                w=cell(r.get("width_mm")),
                d=cell(r.get("depth_mm")),
                h=cell(r.get("height_mm")),
                sep=cell(r.get("eye_sep_mm")),
                ew=cell(r.get("eye_w_mm")),
                eh=cell(r.get("eye_h_mm")),
                ratio=cell(r.get("eye_sep_over_w")),
            )
        )
    lines += ["", "### 比例分布（12 个角色的范围）", ""]
    names = {
        "eye_sep_over_w": "眼孔间距 / 头宽",
        "eye_w_over_w": "眼孔宽 / 头宽",
        "eye_h_over_h": "眼孔高 / 头高",
        "eye_z_over_h": "眼心高 / 头高",
        "depth_over_w": "头深 / 头宽",
        "width_over_h": "头宽 / 头高",
    }
    for key, s in summary.items():
        lines.append(f"- {names[key]}：{s['min']} – {s['max']}（中位 {s['median']}，跨度 {s['spread']}）")
    (BASE / "baseline_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(rows)} characters -> baseline_metrics.{{json,csv,md}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
