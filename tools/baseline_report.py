#!/usr/bin/env python3
"""Aggregate every out/baseline/<char>/report.json into one HTML report.

    python3 tools/baseline_report.py
    python3 tools/baseline_report.py --out out/baseline/index.html

No network, no external assets: images are referenced relatively so the folder
can be copied anywhere. Only measurement numbers and our own renders are
published - the delivered meshes and the modeler's preview stay non-public.
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out" / "modeler-baseline"

VIEWS = [
    ("clay_front", "前壳 · 正视"),
    ("clay_three_quarter", "前壳 · 3/4"),
    ("clay_side", "前壳 · 侧视"),
    ("clay_back", "后壳 · 背面"),
    ("back_three_quarter", "后壳 · 3/4"),
    ("back_side", "后壳 · 侧视"),
]


def load_reports() -> list[dict]:
    out = []
    for d in sorted(p for p in BASE.iterdir() if p.is_dir() and not p.name.startswith("_")):
        rp = d / "report.json"
        if not rp.exists():
            continue
        try:
            r = json.loads(rp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        r["_dir"] = d.name
        out.append(r)
    return out


def fmt(v, md=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{md}f}"
    return str(v)


def img_tag(rel: str, alt: str, cls: str = "") -> str:
    return (
        f'<a href="{html.escape(rel)}"><img class="{cls}" src="{html.escape(rel)}" '
        f'alt="{html.escape(alt)}" loading="lazy"></a>'
    )


def card(r: dict) -> str:
    d = r["_dir"]
    front = (r.get("front_stats") or [{}])[0]
    raw = (r.get("raw_stats") or {}).get("front") or {}
    proxy = r.get("render_proxy")
    status = r.get("status", "?")
    pf = r.get("printability_front") or {}
    pb = r.get("printability_back") or {}
    def pair(key, unit=""):
        a, b = pf.get(key), pb.get(key)
        if a is None and b is None:
            return "—"
        return f"{fmt(a)}{unit} / {fmt(b)}{unit}"
    rows = [
        ("三角面（原始）", f'{raw.get("triangles", front.get("tris", "—")):,}' if raw.get("triangles") or front.get("tris") else "—"),
        ("尺寸 W×D×H (mm)", " × ".join(fmt(v) for v in (raw.get("dims") or front.get("dims") or []))),
        ("水密（可切片）", '<b class="bad">否</b>' if not raw.get("watertight") else "是"),
        ("边界边 / 非流形边", f'{raw.get("boundary_edges", "—")} / {raw.get("non_manifold_edges", "—")}'),
        ("体积 cm³（前 / 后）", pair("volume_cm3")),
        ("表面积 cm²（前 / 后）", pair("area_cm2")),
        ("壁厚中位 mm（前 / 后）", pair("wall_median_mm")),
        ("壁厚 P05 mm（前 / 后）", pair("wall_p05_mm")),
        ("壁厚最小 mm（前 / 后）", pair("wall_min_mm")),
        ("薄于 2mm 占比", f'{pf.get("wall_below_2mm_ratio", "—")} / {pb.get("wall_below_2mm_ratio", "—")}'),
        ("悬垂面积占比", f'{pf.get("overhang_area_ratio", "—")} / {pb.get("overhang_area_ratio", "—")}'),
        ("渲染代理", "是（网格简化）" if proxy else "否"),
    ]
    RAW_CELLS = {"水密（可切片）"}
    table = "".join(
        f"<tr><th>{html.escape(k)}</th>"
        f"<td>{v if k in RAW_CELLS else html.escape(fmt(v))}</td></tr>"
        for k, v in rows
    )
    # copy the modeler's own preview into the character folder so the report is
    # self-contained and the reference sits next to our render
    preview = r.get("preview")
    prev_rel = ""
    if preview and Path(preview).exists():
        src_prev = Path(preview)
        dst_prev = BASE / d / f"_preview{src_prev.suffix.lower()}"
        if not dst_prev.exists() or dst_prev.stat().st_mtime < src_prev.stat().st_mtime:
            import shutil

            shutil.copy2(src_prev, dst_prev)
        prev_rel = f"{d}/{dst_prev.name}"
    tiles = []
    for key, label in VIEWS:
        f = BASE / d / f"{key}.png"
        if f.exists():
            tiles.append((f"{d}/{key}.png", label))
    if prev_rel and (BASE / prev_rel).exists():
        tiles.append((prev_rel, "建模师效果图（参考）"))
    # turntable: lay the individual views out as a grid so any viewer (offline
    # single file, preview pane, phone) gets the same responsive result
    views = []
    for name in (r.get("turntable") or []):
        f = BASE / d / f"{name}.png"
        if not f.exists():
            continue
        if name.startswith("view_top"):
            label = "顶视"
        elif name.startswith("view_bottom"):
            label = "底视"
        else:
            label = name.split("_")[-1] + "°"
        views.append((f"{d}/{name}.png", label))
    if not views:
        views = sorted(
            (f"{d}/{p.name}", p.stem.split("_")[-1] + "°")
            for p in (BASE / d).glob("view_*.png")
            if "top" not in p.stem and "bottom" not in p.stem
        )
    sheet_html = (
        '<figure class="sheet"><div class="views-grid">'
        + "".join(
            f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" alt="{html.escape(label)}" '
            f'loading="lazy"><span>{html.escape(label)}</span></a>'
            for rel, label in views
        )
        + '</div><figcaption>前片转台 10 视角（8 方位 + 顶 / 底）</figcaption></figure>'
    )
    gallery = "".join(
        f'<figure>{img_tag(rel, label)}<figcaption>{html.escape(label)}</figcaption></figure>'
        for rel, label in tiles
    )
    badge = "ok" if status == "ok" else status
    return f"""
    <section class="card" id="{html.escape(d)}">
      <header><h2>{html.escape(d)}</h2><span class="badge {badge}">{html.escape(badge)}</span></header>
      <div class="body">
        <table class="metrics">{table}</table>
        <div class="gallery">{gallery}</div>
      </div>
      <div class="sheets">{sheet_html}</div>
    </section>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE / "index.html"))
    args = ap.parse_args()
    reports = load_reports()
    if not reports:
        print("no reports found")
        return 1
    ok = sum(1 for r in reports if r.get("status") == "ok")
    nav = "".join(
        f'<a href="#{html.escape(r["_dir"])}">{html.escape(r["_dir"])}</a>' for r in reports
    )
    total_tris = sum((r.get("raw_stats") or {}).get("front", {}).get("triangles", 0) or 0 for r in reports)
    body = "".join(card(r) for r in reports)
    doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>建模师头壳模型 · 基准分析</title>
<style>
:root {{ color-scheme: dark; --bg:#14161a; --panel:#1c2026; --line:#2c323b; --txt:#e6e9ef; --dim:#98a2b3; --ok:#3fb950; --bad:#f85149; --accent:#58a6ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt); font:15px/1.6 -apple-system,"Segoe UI","Noto Sans SC",sans-serif; }}
header.top {{ padding:28px 32px 16px; border-bottom:1px solid var(--line); }}
h1 {{ margin:0 0 6px; font-size:22px; }}
.sub {{ color:var(--dim); font-size:13px; }}
nav {{ display:flex; flex-wrap:wrap; gap:8px; padding:14px 32px; border-bottom:1px solid var(--line); }}
nav a {{ color:var(--accent); text-decoration:none; font-size:13px; padding:2px 8px; border:1px solid var(--line); border-radius:999px; }}
main {{ padding:20px 32px 60px; display:grid; gap:22px; grid-template-columns:minmax(0,1fr); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; overflow:hidden; min-width:0; }}
.card header {{ display:flex; align-items:center; gap:12px; padding:14px 18px; border-bottom:1px solid var(--line); }}
.card h2 {{ margin:0; font-size:17px; }}
.badge {{ margin-left:auto; font-size:12px; padding:2px 9px; border-radius:999px; background:#22303f; color:var(--accent); }}
.badge.ok {{ background:#16301f; color:var(--ok); }}
.body {{ display:grid; grid-template-columns:minmax(280px,340px) 1fr; gap:18px; padding:16px 18px; }}
table.metrics {{ border-collapse:collapse; font-size:13px; align-self:start; }}
table.metrics th {{ text-align:left; color:var(--dim); font-weight:500; padding:4px 12px 4px 0; white-space:nowrap; }}
table.metrics td {{ padding:4px 0; font-variant-numeric:tabular-nums; }}
td b.bad {{ color:#f85149; }} td b.ok {{ color:#3fb950; }}
.body {{ display:grid; grid-template-columns:minmax(240px,300px) 1fr; gap:18px; padding:16px 18px; }}
.sheets {{ display:grid; gap:14px; padding:0 18px 18px; min-width:0; }}
figure.sheet {{ margin:0; min-width:0; }}
figure.sheet img {{ width:100%; height:auto; max-width:100%; display:block; border-radius:8px; border:1px solid var(--line); }}
.views-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }}
.views-grid a {{ position:relative; display:block; background:#11151a; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.views-grid a img {{ border:0; border-radius:0; aspect-ratio:4/5; object-fit:contain; background:#0d1013; }}
.views-grid a span {{ position:absolute; left:8px; bottom:6px; font-size:12px; color:#cfd6e2; background:rgba(10,12,16,.65); padding:1px 6px; border-radius:6px; }}
figure.sheet figcaption {{ font-size:12px; color:var(--dim); padding:6px 2px; }}
@media (max-width: 980px) {{ .body {{ grid-template-columns:1fr; }} }}
.gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:12px; }}
figure {{ margin:0; background:#11151a; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
figure img {{ width:100%; display:block; aspect-ratio:4/5; object-fit:cover; }}
figcaption {{ font-size:12px; color:var(--dim); padding:6px 8px; }}
footer {{ padding:18px 32px 40px; color:var(--dim); font-size:12px; border-top:1px solid var(--line); }}
</style></head><body>
<header class="top">
  <h1>建模师头壳模型 · 基准分析</h1>
  <div class="sub">{len(reports)} 个角色 · {ok} 个成功 · 原始三角面合计 {total_tris:,} · 生成于 {time.strftime('%Y-%m-%d %H:%M')}</div>
</header>
<nav>{nav}</nav>
<main>{body}</main>
<footer>素材为建模师提供、非公开：报告只含测量数值与本项目自有渲染，不包含原始网格或可再分发的贴图。</footer>
</body></html>"""
    out = Path(args.out)
    out.write_text(doc, encoding="utf-8")
    print(f"{len(reports)} characters, {ok} ok -> {out}")
    return 0 if ok == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
