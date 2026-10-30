"""Learned cleaning: pieces, labels, training, clean_masks, labelling bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rasterio

from rivmapper_v2 import Project, run_pipeline
from rivmapper_v2.process import units
from rivmapper_v2.process.synthetic import make_river
from rivmapper_v2.process.windows import RiverLayout, offset

pytest.importorskip("sklearn")


def test_pieces_split_corridor_from_lobe_and_loop():
    W = 4.0
    ch = np.zeros((80, 200), bool)
    ch[38:42, 1:-1] = True  # main channel, 1 W wide
    ch[30:38, 90:110] = True  # bank lobe (attached once)
    ch[50:53, 40:160] = True  # side loop ...
    ch[42:50, 40:43] = True  # ... joined at both ends
    ch[42:50, 157:160] = True
    inside = np.ones_like(ch)
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    lab, cor, tab = units.pieces(ch, inside, W)
    assert cor[40, 10] and cor[40, 190]  # the main channel is corridor
    assert not lab[40, 10]
    lobe, loop = lab[33, 100], lab[51, 100]
    assert lobe and loop and lobe != loop
    assert tab.loc[lobe, "contacts"] == 1
    assert tab.loc[loop, "contacts"] >= 2
    assert set(units.FEATURES) <= set(tab.columns)
    # the interior pixel lies in its own piece
    for pid, r in tab.iterrows():
        assert lab[int(r["row"]), int(r["col"])] == pid


def test_corridor_bridges_gaps_and_raster_edge():
    W = 4.0
    ch = np.zeros((60, 200), bool)
    ch[28:32, 0:90] = True  # runs off the raster's left edge
    ch[28:32, 93:200] = True  # a 3-px gap (cloud), then off the right edge
    inside = np.ones_like(ch)  # the reach covers the whole grid
    lab, cor, tab = units.pieces(ch, inside, W)
    assert cor[30, 20] and cor[30, 150]  # both sides are main channel
    assert not lab.any()


def _truth_labels(river, r, synth):
    """Label every piece from the synthetic truth: channel if mostly true channel."""
    from rivmapper_v2.process import clean

    qa = pd.read_csv(r.paths.process_qa_table, keep_default_na=False, na_values=[""])
    years = sorted(qa["year"].unique())
    layout = RiverLayout.from_paths(r.paths)
    rows = []
    for o in layout.ds_orders:
        table, labs = clean.reach_pieces(r, o, years, layout.width_px(o), qa[qa.ds_order == o])
        g = layout.grids[o]
        r0, c0 = offset(g, synth.grid)
        for _, p in table.iterrows():
            t = synth.years.index(int(p["year"]))
            truth = synth.truth[t][r0 : r0 + g.height, c0 : c0 + g.width]
            m = labs[int(p["year"])] == p["piece"]
            frac = truth[m].mean()
            rows.append({"river": r.name, "ds_order": o, "year": int(p["year"]),
                         "piece": int(p["piece"]), "row": int(p["row"]), "col": int(p["col"]),
                         "label": "channel" if frac >= 0.5 else "not_channel"})
    return pd.DataFrame(rows)


def test_clean_masks_end_to_end(tmp_path):
    synth = make_river(["flood", "lake", "braided"], shape=(120, 200), width_px=6, seed=4)
    project = Project.load(synth.write_project(tmp_path))
    t = run_pipeline(project, steps=["process_masks", "clean_masks"])
    assert set(t.status) == {"ok"}, t.to_string()
    r = project.river("Synth")
    assert any(r.paths.pieces_dir.glob("*_pieces.csv"))
    assert not r.paths.cleaned_masks_dir.exists()  # labelling mode: no model

    labels = _truth_labels(synth, r, synth)
    assert {"channel", "not_channel"} <= set(labels.label)
    r.paths.labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(r.paths.labels_dir / "Synth_labels.csv", index=False)

    from rivmapper_v2.cli import main
    from rivmapper_v2.process import clean, ml

    model = tmp_path / "pieces.joblib"
    # the CLI path: matches labels to pieces, then refuses too few labels
    code = main(["ml", "train", str(project.config_path), "--out", str(model)])
    assert code == 0 or not model.exists()
    # train directly with a lower minimum (a small synthetic river)
    qa = pd.read_csv(r.paths.process_qa_table, keep_default_na=False, na_values=[""])
    years = sorted(qa["year"].unique())
    layout = RiverLayout.from_paths(r.paths)
    data = []
    for o, g in labels.groupby("ds_order"):
        table, labs = clean.reach_pieces(r, int(o), years, layout.width_px(int(o)),
                                         qa[qa.ds_order == o])
        data.append(ml.match_labels(g, table, {(int(o), y): a for y, a in labs.items()}))
    data = pd.concat(data, ignore_index=True)
    assert len(data) == len(labels)  # every label found its piece by location
    metrics = ml.train(data, model, min_labels=8)
    assert model.exists() and model.with_suffix(".json").exists()
    assert metrics["n_labels"] == len(labels)

    project = Project.load(project.config_path)
    project.rivers = [
        x.model_copy(update={"process": x.process.model_copy(update={"clean_model": model})})
        for x in project.rivers
    ]
    t = run_pipeline(project, steps=["clean_masks"])
    assert t.iloc[0]["status"] == "ok", t.iloc[0]["message"]
    r = project.river("Synth")
    qa = pd.read_csv(r.paths.process_qa_table)
    assert {"channel_km2_cleaned", "pieces_removed"} <= set(qa.columns)
    # cleaning only removes pixels, and removes more false than true channel
    layout = RiverLayout.from_paths(r.paths)
    false_rm = true_rm = 0
    for ti, y in enumerate(synth.years):
        for o in layout.ds_orders:
            if not r.paths.cleaned_mask(o, y).exists():
                continue
            with rasterio.open(r.paths.processed_mask(o, y)) as s:
                before = s.read(1) == 1
            with rasterio.open(r.paths.cleaned_mask(o, y)) as s:
                after = s.read(1) == 1
            assert not (after & ~before).any()
            g = layout.grids[o]
            r0, c0 = offset(g, synth.grid)
            truth = synth.truth[ti][r0 : r0 + g.height, c0 : c0 + g.width]
            gone = before & ~after
            false_rm += int((gone & ~truth).sum())
            true_rm += int((gone & truth).sum())
    assert false_rm > true_rm


def test_label_bundle(tmp_path):
    from rivmapper_v2.process import label

    synth = make_river(["flood"], shape=(100, 160), width_px=6, seed=5)
    project = Project.load(synth.write_project(tmp_path))
    run_pipeline(project, steps=["process_masks", "clean_masks"])
    r = project.river("Synth")
    index = label.build(r, n=12)
    html = index.read_text()
    assert "Export CSV" in html and "Synth_labels.csv" in html
    imgs = list((index.parent / "img").glob("*.png"))
    assert 0 < len(imgs) <= 12
    items = pd.read_csv(index.parent / "items.csv")
    assert {"ds_order", "year", "row", "col", "piece"} <= set(items.columns)


def test_default_run_skips_export_rgb():
    from rivmapper_v2.pipeline import EXPLICIT_STEPS, STEPS

    assert "export_rgb" in STEPS and "export_rgb" in EXPLICIT_STEPS
    assert "clean_masks" in STEPS and "clean_masks" not in EXPLICIT_STEPS
