"""Timelapse GIFs of the channel masks, for looking at results by eye.

``rivmapper gif rivmapper.yaml`` writes one GIF per reach to
``{river}/gifs/{river}_r{NNN}_{start}_{end}.gif`` and an ``index.html``
that shows them all on one page. Each frame is one year, drawn in two
colours: blue for this year's channel, grey for ground that was channel in
any earlier year. With cleaned masks (``masks/cleaned/``) the frame shows the
processed mask (left) next to the cleaned one (right), with channel that
``clean_masks`` reconnected (the ``filled`` band) in teal; without them, only
the processed mask.
"""

from __future__ import annotations

import html
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import River

log = logging.getLogger(__name__)

OUTSIDE = (0.93, 0.93, 0.93)
PREV = (0.80, 0.80, 0.80)
NOW = (0.10, 0.30, 0.72)
FILLED = (0.0, 0.70, 0.65)
FRAME_MS = 450
LAST_FRAME_MS = 2000
PANEL_PX = 620


def _read(path: Path, band: int = 1) -> np.ndarray | None:
    import rasterio

    if not path.exists():
        return None
    with rasterio.open(path) as s:
        if band > s.count:
            return None
        return s.read(band)


def reach_gif(river: River, o: int, years: list[int], qa: pd.DataFrame, out: Path,
              cleaned: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    p = river.paths
    kinds = ["processed", "cleaned"] if cleaned else ["processed"]
    path_of = {"processed": p.processed_mask, "cleaned": p.cleaned_mask}
    stacks = {k: {y: _read(path_of[k](o, y)) for y in years} for k in kinds}
    fills = {y: _read(p.cleaned_mask(o, y), 5) for y in years} if cleaned else {}
    first = next((a for a in stacks["processed"].values() if a is not None), None)
    if first is None:
        raise FileNotFoundError(f"r{o:03d}: no processed masks")
    inside = np.zeros(first.shape, bool)
    for a in stacks["processed"].values():
        if a is not None:
            inside |= a != 255
    h, w = inside.shape
    rot = h > w * 1.2
    prev: dict[str, np.ndarray | None] = {k: None for k in kinds}
    q = qa[qa.ds_order == o].set_index("year")
    frames = []
    for y in years:
        row = q.loc[y] if y in q.index else None
        status = str(row["status"]) if row is not None else "no data"
        imgs = []
        for k in kinds:
            a = stacks[k][y]
            if a is None and k == "cleaned":
                a = stacks["processed"][y]  # no cleaned file: nothing removed that year
            rgb = np.ones((h, w, 3))
            rgb[~inside] = OUTSIDE
            if prev[k] is not None:
                rgb[prev[k]] = PREV
            if a is not None and status != "no_data" and not (a == 255).all():
                ch = a == 1
                rgb[ch] = NOW
                f = fills.get(y) if k == "cleaned" else None
                if f is not None:
                    rgb[f == 1] = FILLED
                prev[k] = ch if prev[k] is None else prev[k] | ch
            imgs.append(np.rot90(rgb) if rot else rgb)
        ih, iw = imgs[0].shape[:2]
        sc = PANEL_PX / max(ih, iw)
        fig_h = ih * sc / 100 + 1.0
        fig, axes = plt.subplots(1, len(kinds), figsize=(len(kinds) * iw * sc / 100 + 0.2, fig_h),
                                 dpi=100, squeeze=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.85 / fig_h, bottom=0.01, wspace=0.03)
        cols = {"processed": "channel_km2", "cleaned": "channel_km2_cleaned"}
        titles = {"processed": "Before cleaning", "cleaned": "After learned cleaning"}
        for ax, img, k in zip(axes[0], imgs, kinds, strict=True):
            ax.imshow(img, interpolation="nearest")
            ax.set_axis_off()
            v = row[cols[k]] if row is not None and cols[k] in row.index else np.nan
            t = titles[k] if cleaned else "Processed mask"
            ax.set_title(t + (f"   {float(v):.1f} km²" if pd.notna(v) else ""), fontsize=10)
        fig.suptitle(f"{river.name} reach {o}   {y}   {status}", fontsize=12, weight="bold",
                     x=0.01, ha="left", y=0.995)
        handles = [Patch(color=NOW, label="current channel"),
                   Patch(color=PREV, label="previously wetted")]
        if cleaned:
            handles.append(Patch(color=FILLED, label="reconnected"))
        fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=len(handles),
                   frameon=False, bbox_to_anchor=(0.99, 1.0))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB").quantize(colors=32, method=Image.MEDIANCUT))
    durations = [FRAME_MS] * len(frames)
    durations[-1] = LAST_FRAME_MS
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True)
    return out


def write_index(river: River, gifs: list[tuple[int, Path]], out_dir: Path) -> Path:
    cards = "\n".join(
        f'<figure id="r{o}"><figcaption>Reach {o}</figcaption>'
        f'<img src="{html.escape(g.name)}" loading="lazy" alt="reach {o}"></figure>'
        for o, g in gifs
    )
    links = " ".join(f'<a href="#r{o}">{o}</a>' for o, _ in gifs)
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(river.name)} timelapses</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#222}}
nav{{position:sticky;top:0;background:#fff;padding:6px 0;border-bottom:1px solid #ddd}}
nav a{{margin-right:8px}} figure{{margin:24px 0}} figcaption{{font-weight:600;margin-bottom:4px}}
img{{max-width:100%;height:auto;border:1px solid #eee}}</style></head><body>
<h1>{html.escape(river.name)}: channel timelapses</h1><nav>Reach: {links}</nav>
{cards}</body></html>"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def make_gifs(river: River, reaches: list[int] | None = None) -> Path:
    """Write the GIFs and the index page for one river; returns the index path."""
    p = river.paths
    if not p.process_qa_table.exists():
        raise FileNotFoundError(f"{river.name}: no process_masks QA table — run process_masks")
    qa = pd.read_csv(p.process_qa_table, keep_default_na=False, na_values=[""])
    years = sorted(int(y) for y in qa["year"].unique())
    orders = sorted(int(o) for o in qa["ds_order"].unique())
    if reaches:
        orders = [o for o in orders if o in set(reaches)]
    cleaned = p.cleaned_masks_dir.exists() and any(p.cleaned_masks_dir.glob("*.tif"))
    out_dir = p.river_dir / "gifs"
    done = []
    for o in orders:
        out = out_dir / f"{river.name}_r{o:03d}_{years[0]}_{years[-1]}.gif"
        try:
            reach_gif(river, o, years, qa, out, cleaned)
        except FileNotFoundError as e:
            log.warning("  %s", e)
            continue
        done.append((o, out))
        log.info("  reach %d: %s", o, out.name)
    return write_index(river, done, out_dir)
