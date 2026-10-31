#!/usr/bin/env python3
"""Make a single-file, offline HTML: every <img> inlined as a base64 JPEG/PNG.

    PYTHONPATH=.pylib python3 tools/report_embed.py
    # out/baseline/index.html -> index_embedded.html

The normal report uses relative image paths, which break as soon as the page is
opened on its own (or through a file-preview pane that only serves one file).
This variant is one file you can copy anywhere: images are downscaled to a
readable size and embedded.
"""
from __future__ import annotations

import argparse
import base64
import io
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "out" / "modeler-baseline"
SRC = re.compile(r'(<img[^>]+src=")([^"]+)(")')


def encode(path: Path, max_w: int, quality: int) -> str:
    im = Image.open(path)
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    im = im.convert("RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(BASE / "index.html"))
    ap.add_argument("--out", default=str(BASE / "index_embedded.html"))
    ap.add_argument("--max-width", type=int, default=1100)
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    src = Path(args.src)
    html = src.read_text(encoding="utf-8")
    root = src.parent
    cache: dict[str, str] = {}
    missing = []

    def repl(m: re.Match) -> str:
        rel = m.group(2)
        if rel.startswith("data:"):
            return m.group(0)
        p = (root / rel).resolve()
        key = str(p)
        if key not in cache:
            if not p.exists():
                missing.append(rel)
                cache[key] = ""
            else:
                cache[key] = encode(p, args.max_width, args.quality)
        if not cache[key]:
            return m.group(0)
        return m.group(1) + cache[key] + m.group(3)

    out_html = SRC.sub(repl, html)
    out_html = out_html.replace(
        "<title>建模师头壳模型 · 基准分析</title>",
        "<title>建模师头壳模型 · 基准分析（单文件）</title>",
    )
    Path(args.out).write_text(out_html, encoding="utf-8")
    print(f"{len(cache) - len(missing)} images embedded, {len(missing)} missing -> {args.out}"
          f"  ({Path(args.out).stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
