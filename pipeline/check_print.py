#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print-quality check for the split head shell (C1 wall, C2 watertight,
C3 vent ray, C4 flange radial deviation) + report.md.

Usage:
  python3 pipeline/check_print.py --front out/pipeline/v48/shell_front.stl \
      --back out/pipeline/v48/shell_back.stl --report out/pipeline/v48/report.md

Checks reuse the FROZEN command code embedded in pipeline/metrics.json (t4),
so the measured values are directly comparable to the contract thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if os.path.isdir(os.path.join(ROOT, ".pylib")):
    sys.path.insert(0, os.path.join(ROOT, ".pylib"))   # 自带 .pylib 引导（不依赖调用方设置 PYTHONPATH）
METRICS = os.path.join(HERE, "metrics.json")

T5 = {  # t5 acceptance thresholds (report section)
    "wall_median_mm": [2.9, 3.1],
}


def run_command(cmd, cwd):
    p = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True)
    out = None
    try:
        out = json.loads(p.stdout)
    except Exception:
        pass
    return p.returncode, out, (p.stderr or "")[-200:]


def retarget(cmd, args):
    """Replace the argv tail of a frozen 'python3 - <args> <<'PY' command."""
    head, tail = cmd.split(" <<'PY'\n", 1)
    head = re.sub(r"python3 - .*$", "python3 - " + args, head)
    return head + " <<'PY'\n" + tail


def gate(name, value, lo=None, hi=None):
    ok = True
    if lo is not None and value < lo:
        ok = False
    if hi is not None and value > hi:
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True)
    ap.add_argument("--back", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    d = json.load(open(METRICS))
    front = args.front if os.path.isabs(args.front) else os.path.join(ROOT, args.front)
    back = args.back if os.path.isabs(args.back) else os.path.join(ROOT, args.back)
    outdir = os.path.dirname(os.path.abspath(args.report if os.path.isabs(args.report)
                                             else os.path.join(ROOT, args.report)))
    os.makedirs(outdir, exist_ok=True)
    grid = os.path.join(outdir, "grid.json")

    checks = []  # (id, scope, command, exit, values, gates[(label, ok, detail)])

    def add(cid, scope, cmd, rc, val, gates):
        checks.append({"id": cid, "scope": scope, "command": cmd, "exit": rc,
                       "values": val, "gates": gates})

    # ---- C2 watertight (both parts) ----
    for label, path in (("front", front), ("back", back)):
        cmd = retarget(d["watertight"]["command"], os.path.relpath(path, ROOT))
        rc, out, err = run_command(cmd, ROOT)
        g = []
        if out:
            for k, thr in d["watertight"]["thresholds"].items():
                v = out.get(k)
                g.append({"label": "watertight.%s == %s" % (k, thr), "ok": v == thr,
                          "value": v})
        add("C2 watertight", label, cmd, rc, out or {"error": err}, g)

    # ---- C1 wall (both parts) ----
    for label, path in (("front", front), ("back", back)):
        cmd = retarget(d["wall"]["command"], os.path.relpath(path, ROOT) + " 6000")
        rc, out, err = run_command(cmd, ROOT)
        g = []
        if out:
            med, p05 = out.get("median_mm"), out.get("p05_mm")
            lo, hi = d["wall"]["thresholds"]["median_mm"]
            g.append({"label": "wall median in contract [%.2f, %.2f]" % (lo, hi),
                      "ok": gate("med", med, lo, hi), "value": med})
            g.append({"label": "wall median in t5 [%.1f, %.1f]" % tuple(T5["wall_median_mm"]),
                      "ok": gate("med5", med, *T5["wall_median_mm"]), "value": med})
            g.append({"label": "wall p05 >= %.2f" % d["wall"]["thresholds"]["p05_mm_min"],
                      "ok": gate("p05", p05, d["wall"]["thresholds"]["p05_mm_min"]), "value": p05})
        add("C1 wall", label, cmd, rc, out or {"error": err}, g)

    # ---- C3 vents (grid mode, on the assembled shell: ray must hit the far inner wall) ----
    if os.path.exists(grid):
        import trimesh
        fa = trimesh.load(front)
        ba = trimesh.load(back)
        asm = trimesh.util.concatenate([fa, ba])
        asm_path = os.path.join(outdir, "assembly_check.stl")
        asm.export(asm_path)
        cmd = retarget(d["vents"]["command_grid"],
                       "%s grid %s 0" % (os.path.relpath(asm_path, ROOT), os.path.relpath(grid, ROOT)))
        rc, out, err = run_command(cmd, ROOT)
        g = []
        if out:
            thr = d["vents"]["ray_check"]["thresholds"]
            for key in ("open_center_pass", "corners_open_pass", "ring_closed_pass"):
                g.append({"label": "vents %s == %s" % (key, thr[key]),
                          "ok": out.get(key) == thr[key], "value": out.get(key)})
        add("C3 vents", "back", cmd, rc, out or {"error": err}, g)
    else:
        add("C3 vents", "back", "(grid.json missing)", 1, {"error": "grid.json missing"},
            [{"label": "grid.json present", "ok": False, "value": None}])

    # ---- C4 flange (both parts) ----
    for label, path in (("front", front), ("back", back)):
        cmd = retarget(d["flange"]["command"],
                       "%s 0 8 2" % os.path.relpath(path, ROOT))
        rc, out, err = run_command(cmd, ROOT)
        g = []
        if out:
            rthr = d["flange"]["radial_deviation"]["thresholds"]
            g.append({"label": "flange present", "ok": out.get("tab_present") is True,
                      "value": out.get("tab_present")})
            g.append({"label": "thickness in [1.7, 2.3]", "ok": gate("t", out.get("thickness_mm", -9), 1.7, 2.3),
                      "value": out.get("thickness_mm")})
            g.append({"label": "radial_dev p95 <= %.2f" % rthr["radial_dev_p95_mm_max"],
                      "ok": gate("dp", out.get("radial_dev_p95_mm", 9), None, rthr["radial_dev_p95_mm_max"]),
                      "value": out.get("radial_dev_p95_mm")})
            g.append({"label": "radial_dev max <= %.2f" % rthr["radial_dev_max_mm_max"],
                      "ok": gate("dm", out.get("radial_dev_max_mm", 9), None, rthr["radial_dev_max_mm_max"]),
                      "value": out.get("radial_dev_max_mm")})
            g.append({"label": "jitter p95 <= %.2f" % rthr["jitter_p95_mm_max"],
                      "ok": gate("j", out.get("jitter_p95_mm", 9), None, rthr["jitter_p95_mm_max"]),
                      "value": out.get("jitter_p95_mm")})
        add("C4 flange", label, cmd, rc, out or {"error": err}, g)

    # ---- report.md ----
    fails = [c for c in checks if any(not g["ok"] for g in c["gates"]) or c["exit"] != 0]
    lines = []
    lines.append("# 打印检查报告（C1–C4）\n")
    lines.append("生成：pipeline/check_print.py ｜ 契约：pipeline/metrics.json（冻结 v1）\n")
    lines.append("输入：`%s`、`%s`\n" % (args.front, args.back))
    stage_map = [("S1 抽壳 3mm（壁厚/水密）", ["C1 wall", "C2 watertight"]),
                 ("S2 40 三角通风孔（贯穿）", ["C3 vents"]),
                 ("S3 竖向分件 + 8×2 法兰", ["C4 flange"])]
    lines.append("## 阶段门禁汇总（S1 / S2 / S3 分列）\n")
    lines.append("| 阶段 | 检查 | 结果 |")
    lines.append("|---|---|---|")
    for stname, ids in stage_map:
        for c in checks:
            if c["id"] in ids:
                ok = all(g["ok"] for g in c["gates"]) and c["exit"] == 0
                lines.append("| %s | %s · %s | %s |" % (stname, c["id"], c["scope"], "✅ PASS" if ok else "❌ FAIL"))
    lines.append("")
    if fails:
        lines.append("## ❌ 失败项（%d）—— 照实列出，不隐藏\n" % len(fails))
        for c in fails:
            for g in c["gates"]:
                if not g["ok"]:
                    lines.append("- **%s / %s**：%s → 实测 `%s`" % (c["id"], c["scope"], g["label"], g["value"]))
            if c["exit"] != 0:
                lines.append("- **%s / %s**：命令退出码 %d" % (c["id"], c["scope"], c["exit"]))
    else:
        lines.append("## ✅ 全部门禁通过\n")
    lines.append("\n## 明细（每项检查：命令 / 退出码 / 数值 / 门限）\n")
    for c in checks:
        lines.append("### %s · %s\n" % (c["id"], c["scope"]))
        lines.append("命令：\n```bash\n%s\n```\n" % c["command"].split(" <<'PY'")[0] + "  # 完整命令（含内联代码）见同目录 commands.sh；代码原文亦存于 pipeline/metrics.json")
        lines.append("退出码：`%d`\n" % c["exit"])
        if isinstance(c["values"], dict):
            keys = [k for k in ("watertight", "boundary_edges", "nonmanifold_edges", "winding_consistent",
                                "degenerate_faces", "median_mm", "p05_mm", "mean_wall_volume_over_area_mm",
                                "hits", "misses", "open_center_pass", "corners_open_pass", "ring_closed_pass",
                                "vents_total", "tab_present", "thickness_mm", "radial_dev_p95_mm",
                                "radial_dev_max_mm", "jitter_p95_mm") if k in c["values"]]
            lines.append("数值：" + "；".join("%s = %s" % (k, c["values"][k]) for k in keys) + "\n")
        lines.append("门限：\n")
        for g in c["gates"]:
            lines.append("- [%s] %s → `%s`" % ("PASS" if g["ok"] else "FAIL", g["label"], g["value"]))
        lines.append("")
    report_path = args.report if os.path.isabs(args.report) else os.path.join(ROOT, args.report)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # sidecar: every check command verbatim & re-runnable
    sh_path = os.path.join(os.path.dirname(report_path), "commands.sh")
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n# 每项检查的完整命令（与 report.md 一一对应）；在 %s 下执行\ncd %s\n" % (ROOT, ROOT))
        for c in checks:
            f.write("\n# %s · %s  (exit=%d)\n%s\n" % (c["id"], c["scope"], c["exit"], c["command"]))

    print(json.dumps({"checks": len(checks), "failed": len(fails),
                      "report": args.report,
                      "summary": [{"id": c["id"], "scope": c["scope"], "exit": c["exit"],
                                   "pass": all(g["ok"] for g in c["gates"])} for c in checks]},
                     ensure_ascii=False, indent=2))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
