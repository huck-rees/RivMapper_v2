"""Labelling bundle: a local, offline page for labelling candidate pieces.

``rivmapper label rivmapper.yaml --river X --n 300`` samples pieces from the
piece tables written by ``clean_masks`` (stratified by reach, loop vs lobe,
and size), renders one image per piece and writes
``{river}/labels/bundle/index.html``. Open it in a browser straight from
disk: no server, no upload. Each image shows, left to right,

1. the year's true-colour composite (``export_rgb``; if missing, the water
   frequency), with the piece outlined in red;
2. the channel mask the year before, 3. the year itself, 4. the year after:
   main corridor dark blue, the piece red, other pieces orange.

Keys: 1 = channel, 2 = not channel, 3 = unsure, ← / → to move. Labels are
kept in the browser while you work; "Export CSV" saves
``{river}_labels.csv``, which goes in ``{river}/labels/`` for
``rivmapper ml train``. Optional suggested labels (``suggestions.csv`` in
the labels folder, same columns) are shown as hints, never pre-selected.
"""

from __future__ import annotations

import html
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import River
from . import clean

log = logging.getLogger(__name__)

PANEL = 360


def sample(pieces: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Stratified sample: reach × (loop / lobe) × size tercile, evenly."""
    if pieces.empty:
        return pieces
    d = pieces.copy()
    d["kind"] = np.where(d["contacts"] >= 2, "loop", "lobe")
    d["size_bin"] = pd.qcut(d["area_w2"].rank(method="first"), 3, labels=False)
    strata = d.groupby(["ds_order", "kind", "size_bin"])
    per = max(1, math.ceil(n / max(strata.ngroups, 1)))
    out = pd.concat(
        [g.sample(min(len(g), per), random_state=seed) for _, g in strata],
        ignore_index=True,
    )
    if len(out) > n:
        out = out.sample(n, random_state=seed)
    return out.sort_values(["ds_order", "year"]).reset_index(drop=True)


def _crop_box(lab: np.ndarray, pid: int, w_px: float):
    rr, cc = np.nonzero(lab == pid)
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    # the piece fills about half the frame, with at least ~12 W of river
    # around it for context
    half = int(max(1.0 * max(r1 - r0, c1 - c0), 6 * w_px, 30))
    rc, ccn = (r0 + r1) // 2, (c0 + c1) // 2
    return (max(rc - half, 0), min(rc + half, lab.shape[0]),
            max(ccn - half, 0), min(ccn + half, lab.shape[1]))


def _mask_panel(ch, corridor, lab, pid):
    rgb = np.ones(ch.shape + (3,))
    rgb[ch] = (0.65, 0.75, 0.95)
    rgb[corridor] = (0.10, 0.30, 0.72)
    rgb[(lab > 0) & (lab != pid)] = (0.98, 0.62, 0.25)
    rgb[lab == pid] = (0.85, 0.10, 0.10)
    return rgb


def render(river: River, row, stack, labs, corridors, w_px, freq, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    from . import units

    y, pid = int(row["year"]), int(row["piece"])
    lab = labs[y]
    r0, r1, c0, c1 = _crop_box(lab, pid, w_px)
    sl = (slice(r0, r1), slice(c0, c1))
    edge = (lab == pid) & ~ndi.binary_erosion(lab == pid)
    edge = ndi.binary_dilation(edge, iterations=1)
    rgb_path = river.paths.rgb(int(row["ds_order"]), y)
    if rgb_path.exists():
        import rasterio

        with rasterio.open(rgb_path) as s:
            raw = np.moveaxis(s.read(), 0, -1)[sl].astype(float)
        valid = raw.sum(-1) > 0
        if valid.sum() > 20:
            # stretch this crop to its own 2–98 % range for contrast
            lo, hi = np.percentile(raw[valid], [2, 98])
            bg = np.clip((raw - lo) / max(hi - lo, 1.0), 0, 1)
            bg[~valid] = 0
            bg_title = f"Landsat {y} (median, stretched)"
        else:
            bg = np.zeros(raw.shape)
            bg_title = f"Landsat {y}: no clear scenes"
    else:
        bg = np.stack([1 - freq[sl] * 0.8] * 3, -1)
        bg_title = "water frequency (no RGB)"
    bg = bg.copy()
    bg[edge[sl]] = (1.0, 0.1, 0.1)
    panels = [(bg, bg_title)]
    for k in (-1, 0, 1):
        yy = y + k
        a = stack.get(yy)
        if a is None:
            panels.append((np.full(bg.shape, 0.9), f"{yy}: no data"))
            continue
        ch = a[0] == 1
        if k == 0:
            cor = corridors[y]
            pan = _mask_panel(ch, cor, lab, pid)
        else:
            pan = np.ones(ch.shape + (3,))
            pan[ch] = (0.10, 0.30, 0.72)
            pan[edge] = (0.85, 0.10, 0.10)
        panels.append((pan[sl], f"{yy}" + (" (this year)" if k == 0 else "")))
    fig, axes = plt.subplots(1, 4, figsize=(4 * PANEL / 100, PANEL / 100 + 0.35), dpi=100)
    for ax, (img, t) in zip(axes, panels, strict=True):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(t, fontsize=8)
        ax.set_axis_off()
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.9, 0.02, 0)
    fig.savefig(out, dpi=100)
    plt.close(fig)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Piece labelling – __RIVER__</title>
<style>
:root{--bg:#fafafa;--fg:#1d1d1f;--mut:#6b6b70;--line:#d8d8dc;--ch:#1f4fb8;--no:#c0392b;--un:#8a8a8f}
@media (prefers-color-scheme: dark){:root{--bg:#18181a;--fg:#ececee;--mut:#a0a0a6;--line:#3a3a3e}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between}
h1{font-size:18px;margin:0}.mut{color:var(--mut)}
img{width:100%;height:auto;border:1px solid var(--line);margin:10px 0;background:#fff}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;padding:8px 14px;border-radius:6px;border:1px solid var(--line);background:transparent;color:var(--fg);cursor:pointer}
button.sel{color:#fff}button.ch.sel{background:var(--ch);border-color:var(--ch)}
button.no.sel{background:var(--no);border-color:var(--no)}button.un.sel{background:var(--un);border-color:var(--un)}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--ch)}
.meta{font-size:13px}.hint{font-size:13px;margin-top:4px}
</style></head><body><main>
<header><h1>__RIVER__: is the red piece river channel?</h1>
<span class="mut" id="count"></span></header>
<div class="bar"><div id="prog"></div></div>
<div class="meta mut" id="meta"></div>
<img id="img" alt="piece">
<div class="btns">
<button class="ch" data-l="channel">1 · Channel</button>
<button class="no" data-l="not_channel">2 · Not channel</button>
<button class="un" data-l="unsure">3 · Unsure</button>
<span style="flex:1"></span>
<button id="prev">← Prev</button><button id="next">Next →</button>
<button id="nextun">Next unlabelled</button>
<button id="exp">Export CSV</button></div>
<div class="hint mut" id="hint"></div>
<p class="mut" style="font-size:13px">Left: true-colour composite, piece outlined in red.
Then the channel mask the year before, the year itself (dark blue = main corridor, red = this piece,
orange = other pieces) and the year after. Channel = part of the active river, including side channels
and cut-off loops still open at both ends. Not channel = floodplain or flood water, lakes, artefacts.
Labels stay in this browser until you export; save the CSV as <code>__CSVNAME__</code> in the river's
<code>labels</code> folder.</p>
</main><script>
const ITEMS=__ITEMS__;const KEY="rivmapper-labels-__RIVER__-__BUNDLE__";
let lab={};try{lab=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;try{i=+localStorage.getItem(KEY+"-i")||0}catch(e){}
const $=s=>document.querySelector(s);
function save(){try{localStorage.setItem(KEY,JSON.stringify(lab));localStorage.setItem(KEY+"-i",i)}catch(e){}}
function show(){i=Math.max(0,Math.min(ITEMS.length-1,i));const it=ITEMS[i];
$("#img").src=it.img;const n=Object.keys(lab).length;
$("#count").textContent=`${i+1} / ${ITEMS.length} · ${n} labelled`;
$("#prog").style.width=(100*n/ITEMS.length)+"%";
$("#meta").textContent=`reach ${it.ds_order} · ${it.year} · ${it.kind} · ${it.area_w2.toFixed(1)} W² · touches corridor ${it.contacts}×`;
$("#hint").textContent=it.suggestion?`Suggested: ${it.suggestion.replace("_"," ")}`:"";
document.querySelectorAll("button[data-l]").forEach(b=>b.classList.toggle("sel",lab[it.id]===b.dataset.l));save()}
function set(l){lab[ITEMS[i].id]=l;i++;show()}
document.querySelectorAll("button[data-l]").forEach(b=>b.onclick=()=>set(b.dataset.l));
$("#prev").onclick=()=>{i--;show()};$("#next").onclick=()=>{i++;show()};
$("#nextun").onclick=()=>{const j=ITEMS.findIndex(t=>!lab[t.id]);if(j>=0){i=j;show()}};
document.addEventListener("keydown",e=>{if(e.key==="1")set("channel");else if(e.key==="2")set("not_channel");
else if(e.key==="3")set("unsure");else if(e.key==="ArrowLeft"){i--;show()}else if(e.key==="ArrowRight"){i++;show()}});
$("#exp").onclick=()=>{const rows=["river,ds_order,year,piece,row,col,label"];
for(const t of ITEMS){if(lab[t.id])rows.push([t.river,t.ds_order,t.year,t.piece,t.row,t.col,lab[t.id]].join(","))}
const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows.join("\\n")+"\\n"],{type:"text/csv"}));
a.download="__CSVNAME__";a.click()};
show();
</script></body></html>
"""


def build(river: River, n: int = 300, seed: int = 0, bundle: str = "bundle") -> Path:
    """Write the labelling bundle; returns the path of index.html."""
    from .windows import RiverLayout

    p = river.paths
    tables = []
    for f in sorted(p.pieces_dir.glob(f"{river.name}_r*_pieces.csv")):
        try:
            t = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if len(t):
            tables.append(t)
    if not tables:
        raise FileNotFoundError(
            f"no piece tables in {p.pieces_dir} — run `rivmapper run … --steps clean_masks` first"
        )
    pieces = pd.concat(tables, ignore_index=True)
    chosen = sample(pieces, n, seed)
    sugg = {}
    sfile = p.labels_dir / "suggestions.csv"
    if sfile.exists():
        s = pd.read_csv(sfile)
        sugg = {(int(r.ds_order), int(r.year), int(r.row), int(r.col)): r.label for r in s.itertuples()}
    qa = pd.read_csv(p.process_qa_table, keep_default_na=False, na_values=[""])
    years = sorted(qa["year"].unique().tolist())
    layout = RiverLayout.from_paths(p)
    out_dir = p.labels_dir / bundle
    (out_dir / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for o, g in chosen.groupby("ds_order"):
        w = layout.width_px(int(o))
        stack = clean.reach_stack(river, int(o), years)
        obs = [a for a in stack.values() if a is not None]
        inside = np.zeros(obs[0].shape[1:], bool)
        for a in obs:
            inside |= a[0] != 255
        freq = np.mean([a[0] == 1 for a in obs], axis=0)
        labs, cors = {}, {}
        from . import units

        for y in sorted(g["year"].unique()):
            a = stack[int(y)]
            others = {k: (stack[y + k][0] == 1) if stack.get(y + k) is not None else None
                      for k in (-2, -1, 1, 2)}
            lab, cor, _ = units.pieces(a[0] == 1, inside, w, core=a[1] == 1, others=others,
                                       freq_all=freq)
            labs[int(y)], cors[int(y)] = lab, cor
        for _, row in g.iterrows():
            pid = int(labs[int(row["year"])][int(row["row"]), int(row["col"])])
            if pid == 0:
                continue
            row = row.copy()
            row["piece"] = pid
            iid = f"r{int(o):03d}_{int(row['year'])}_{pid}"
            render(river, row, stack, labs, cors, w, freq, out_dir / "img" / f"{iid}.png")
            items.append({
                "id": iid, "img": f"img/{iid}.png", "river": river.name,
                "ds_order": int(o), "year": int(row["year"]), "piece": pid,
                "row": int(row["row"]), "col": int(row["col"]),
                "kind": "loop" if row["contacts"] >= 2 else "lobe",
                "area_w2": float(row["area_w2"]), "contacts": int(row["contacts"]),
                "suggestion": sugg.get((int(o), int(row["year"]), int(row["row"]), int(row["col"])), ""),
            })
        log.info("  reach %d: %d images", int(o), len(g))
    page = (PAGE.replace("__ITEMS__", json.dumps(items))
            .replace("__RIVER__", html.escape(river.name))
            .replace("__BUNDLE__", bundle)
            .replace("__CSVNAME__", f"{river.name}_labels.csv"))
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    pd.DataFrame(items).to_csv(out_dir / "items.csv", index=False)
    return index
