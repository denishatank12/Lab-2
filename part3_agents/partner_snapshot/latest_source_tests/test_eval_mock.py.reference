"""The evaluation harness runs end to end on the real scenario/red-team files with mock models."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from agents import evaluate as ev  # noqa: E402
from agents.backends import BackendConfig, MockBackends  # noqa: E402
from agents.graph import Orchestrator  # noqa: E402


def test_harness(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("mi", os.path.join(HERE, "..", "eval", "make_images.py"))
    mi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mi)
    images = str(tmp_path / "images")
    mi.make_all(images, offline=True, log=lambda *a: None)
    o = Orchestrator(MockBackends(), BackendConfig(), out_dir=str(tmp_path / "o"), trace_dir=str(tmp_path / "t"))
    scen = ev.load_cases(os.path.join(HERE, "..", "eval", "scenarios.json"), "scenarios")
    red = ev.load_cases(os.path.join(HERE, "..", "eval", "redteam.json"), "cases")
    assert len(scen) == 15 and len(red) == 10
    assert pd.Series([s["category"] for s in scen]).value_counts().eq(3).all()
    w, sw = ev.run_scenarios(o, scen, images, True, log=lambda *a: None)
    wo, _ = ev.run_scenarios(o, scen, images, False, log=lambda *a: None)
    r, _ = ev.run_redteam(o, red, images, check_images=False, log=lambda *a: None)
    assert r["attack_success"].sum() == 0
    p1, p2 = ev.human_sheets(w, wo, sw, str(tmp_path / "h"))
    h = pd.read_csv(p1, dtype=str, keep_default_na=False)
    h["rater_A"] = "pass"; h["rater_B"] = ["pass"] * (len(h) - 1) + ["fail"]; h.to_csv(p1, index=False)
    c = pd.read_csv(p2, dtype=str, keep_default_na=False); c["human_decision"] = "accept"; c.to_csv(p2, index=False)
    st = ev.human_stats(p1, p2)
    assert 0.9 < st["human_agreement"] < 1 and st["critic_agreement"] == 1.0
    t = ev.summary_table(w, wo, r, st)
    assert list(t["system"]) == ["without critique (baseline)", "with critique (final)"]
    assert len(os.listdir(o.trace_dir)) == 40
