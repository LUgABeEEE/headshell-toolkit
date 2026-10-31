#!/usr/bin/env python3
"""Per-character multi-view contact sheet.

    python3 tools/contact_sheet.py                 # every character
    python3 tools/contact_sheet.py --char "character-a"

Composes view_000..view_315 (turntable) + view_top / view_bottom into one PNG
(<char>/turntable_sheet.png, 5 columns) and a second sheet that puts the painted
front, the clay views and the modeler preview side by side (overview_sheet.png).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out" / "modeler-baseline"

TILE_W, TILE_H = 320, 410
COLS = 5
LABEL_H = 26
BG = (18, 21, 26)
FG = (200, 208, 220)


CJK_CANDIDATES = (
    "~/.fonts/NotoSansSC-Medium.otf",
    "~/.fonts/NotoSansCJK-Regular.ttc",
    "~/.fonts/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "DejaVuSans.ttf",
)


def _font(size: int = 15):
    for name in CJK_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def sheet(entries: list[tuple[Path, str]], out: Path, title: str) -> bool:
    """entries: (image path, label). Renders a COLS-wide grid."""
    if not entries:
        return False
    rows = (len(entries) + COLS - 1) // COLS
    W = COLS * TILE_W
    H = rows * (TILE_H + LABEL_H) + 44
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    font, title_font = _font(15), _font(20)
    draw.text((14, 12), title, fill=FG, font=title_font)
    for i, (path, label) in enumerate(entries):
        cx = (i % COLS) * TILE_W
        cy = 44 + (i // COLS) * (TILE_H + LABEL_H)
        try:
            im = Image.open(path).convert("RGB")
        except OSError:
            continue
        im.thumbnail((TILE_W - 12, TILE_H - 12), Image.LANCZOS)
        ox = cx + (TILE_W - im.width) // 2
        oy = cy + (TILE_H - im.height) // 2
        canvas.paste(im, (ox, oy))
        draw.text((cx + 8, cy + TILE_H + 4), label, fill=FG, font=font)
    canvas.save(out, optimize=True)
    return True


def build(char_dir: Path) -> list[Path]:
    report = {}
    rp = char_dir / "report.json"
    if rp.exists():
        report = json.loads(rp.read_text(encoding="utf-8"))
    yaw = [n for n in report.get("turntable", []) if n.startswith("view_") and n not in ("view_top", "view_bottom")]
    if not yaw:
        yaw = sorted(p.stem for p in char_dir.glob("view_*.png") if p.stem not in ("view_top", "view_bottom"))
    entries = []
    for name in yaw:
        f = char_dir / f"{name}.png"
        if f.exists():
            deg = name.split("_")[-1]
            entries.append((f, f"{deg}°"))
    for extra, label in (("view_top", "顶视"), ("view_bottom", "底视")):
        f = char_dir / f"{extra}.png"
        if f.exists():
            entries.append((f, label))
    made = []
    if sheet(entries, char_dir / "turntable_sheet.png", f"{char_dir.name} · 转台 {len(entries)} 视角"):
        made.append(char_dir / "turntable_sheet.png")

    overview = []
    for name, label in (
        ("clay_front", "前壳 · 正视"),
        ("clay_three_quarter", "前壳 · 3/4"),
        ("clay_side", "前壳 · 侧视"),
        ("clay_back", "后壳 · 背面"),
        ("back_three_quarter", "后壳 · 3/4"),
        ("back_side", "后壳 · 侧视"),
    ):
        f = char_dir / f"{name}.png"
        if f.exists():
            overview.append((f, label))
    previews = sorted(char_dir.glob("_preview.*"))
    if previews:
        overview.append((previews[0], "建模师效果图"))
    if sheet(overview, char_dir / "overview_sheet.png", f"{char_dir.name} · 外观一览"):
        made.append(char_dir / "overview_sheet.png")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", action="append", default=None)
    args = ap.parse_args()
    dirs = [d for d in sorted(BASE.iterdir()) if d.is_dir() and not d.name.startswith("_")]
    if args.char:
        dirs = [d for d in dirs if d.name in set(args.char)]
    total = 0
    for d in dirs:
        made = build(d)
        total += len(made)
        print(f"{d.name:<20} {', '.join(p.name for p in made) or 'no views'}")
    print(f"{total} sheets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
