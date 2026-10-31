#!/usr/bin/env python3
"""One-command baseline build for a modeler head-shell package.

    python3 tools/run_baseline.py --char "character-a"
    python3 tools/run_baseline.py --all            # every extracted package, serial
    python3 tools/run_baseline.py --all --dry-run  # show the plan only

Pipeline per character (strictly serial, one Blender at a time):
  1. raw statistics straight from the delivered STL (tools/stl_tools.py, numpy)
  2. if the front/back piece exceeds the memory-safe triangle budget, write a
     grid-decimated proxy under out/baseline/_proxy/ (the delivered
     file is untouched; the proxy only feeds the preview renders)
  3. Blender (tools/blender-safe.sh, 8 GiB cap, flock) renders clay views,
     eye-opening geometry and the painted front using the modeler preview
  4. merge everything into out/baseline/<char>/report.json

Exit code 0 only if every requested character produced status="ok".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pylib"))
sys.path.insert(0, str(ROOT / "tools"))

import stl_tools  # noqa: E402

SRC_ROOT = ROOT / "assets" / "incoming" / "modeler-20261020"
OUT_ROOT = ROOT / "out" / "modeler-baseline"
PROXY_DIR = OUT_ROOT / "_proxy"
JSON_THRESHOLD = 9_000_000  # above this a raw import will not fit the 8 GiB guard
PROXY_TARGET = 2_500_000


def characters() -> list[str]:
    if not SRC_ROOT.exists():
        return []
    return sorted(
        p.name for p in SRC_ROOT.iterdir() if p.is_dir() and (p / "前头.stl").exists()
    )


def build(char: str, force: bool = False, dry_run: bool = False, paint_only: bool = False,
          turntable: int = 0, turntable_only: bool = False) -> dict:
    src = SRC_ROOT / char
    out = OUT_ROOT / char
    report_path = out / "report.json"
    if report_path.exists() and not force:
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            has_turntable = bool(existing.get("turntable"))
            if existing.get("status") == "ok" and (not turntable or has_turntable):
                return {"character": char, "status": "cached", "seconds": 0}
        except json.JSONDecodeError:
            pass

    t0 = time.time()
    front = src / "前头.stl"
    back = src / "后头.stl"
    if not front.exists():
        return {"character": char, "status": "error", "error": "前头.stl missing"}

    steps: list[str] = []
    front_stats = stl_tools.stats(str(front))
    steps.append(f"front raw: {front_stats['triangles']} tris")
    back_stats = None
    if back.exists():
        back_stats = stl_tools.stats(str(back))
        steps.append(f"back raw: {back_stats['triangles']} tris")

    front_render = None
    if front_stats["triangles"] > JSON_THRESHOLD:
        fmt = "front1" if False else "front"
        proxy = PROXY_DIR / f"{char}_前头_{PROXY_TARGET//1_000_000}M.stl"
        info = stl_tools.decimate(str(front), str(proxy), target_tris=PROXY_TARGET)
        steps.append(f"front proxy: {info['src_triangles']} -> {info['dst_triangles']} tris (grid {info['grid_mm']} mm)")
        front_render = proxy
    back_render = None
    if back_stats and back_stats["triangles"] > JSON_THRESHOLD:
        proxy = PROXY_DIR / f"{char}_后头_{PROXY_TARGET//1_000_000}M.stl"
        info = stl_tools.decimate(str(back), str(proxy), target_tris=PROXY_TARGET)
        steps.append(f"back proxy: {info['src_triangles']} -> {info['dst_triangles']} tris")
        back_render = proxy

    metrics_path = out / "_raw_stats.json"
    out.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"front": front_stats, "back": back_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cmd = [
        str(ROOT / "tools" / "blender-safe.sh"),
        "-b",
        "--factory-startup",
        "-P",
        str(ROOT / "tools" / "baseline_render.py"),
        "--",
        "--src",
        str(src),
        "--out",
        str(out),
        "--raw-metrics",
        str(metrics_path),
    ]
    if front_render:
        cmd += ["--front-override", str(front_render)]
    if back_render:
        cmd += ["--back-override", str(back_render)]
    if paint_only:
        cmd.append("--paint-only")
    if turntable:
        cmd += ["--turntable", str(turntable)]
    if turntable_only:
        cmd.append("--turntable-only")

    if dry_run:
        return {"character": char, "status": "dry-run", "steps": steps, "cmd": cmd}

    log_path = out / "blender.log"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
    tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]

    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"character": char, "status": "error", "error": f"blender exit {proc.returncode}"}
    if proc.returncode != 0 and report.get("status") == "ok":
        # a stale ok report must not mask a crashed Blender run
        report["status"] = "error"
        report["error"] = f"blender exit {proc.returncode}"
    report["steps"] = steps
    report["blender_exit"] = proc.returncode
    report["seconds_total"] = round(time.time() - t0, 1)
    if proc.returncode != 0 and report.get("status") != "ok":
        report["blender_tail"] = tail
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "character": char,
        "status": report.get("status", "error"),
        "seconds": report["seconds_total"],
        "tris": report.get("total_tris"),
        "proxy": report.get("render_proxy"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--paint-only", action="store_true", help="only re-render the painted views")
    ap.add_argument("--turntable", type=int, default=0, help="render N yaw views + top/bottom per character")
    ap.add_argument("--turntable-only", action="store_true")
    args = ap.parse_args()

    chars = characters()
    if args.char:
        chars = [c for c in chars if c in set(args.char)]
        missing = set(args.char) - set(chars)
        if missing:
            print(f"unknown character(s): {sorted(missing)}", file=sys.stderr)
            return 2
    if not args.all and not args.char:
        ap.error("pass --all or --char NAME")
    if not chars:
        print("no extracted packages", file=sys.stderr)
        return 1

    results = []
    for i, c in enumerate(chars, 1):
        res = build(
            c,
            force=args.force,
            dry_run=args.dry_run,
            paint_only=args.paint_only,
            turntable=args.turntable,
            turntable_only=args.turntable_only,
        )
        results.append(res)
        print(f"[{i}/{len(chars)}] {c}: {res['status']}  {res.get('seconds','')}s", flush=True)
        if args.dry_run:
            for s in res.get("steps", []):
                print("    " + s)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "build-log.json").write_text(
        json.dumps(
            {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "results": results},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    bad = [r for r in results if r["status"] not in {"ok", "cached", "dry-run"}]
    print(f"done: {len(results)-len(bad)}/{len(results)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
