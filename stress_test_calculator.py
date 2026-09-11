"""
Stress test calculator.py -- the students' production calculator page.

What matters: one button per country actually in play, numbers taken from the
LIVE simulation (so a shock is reflected after a re-export), and check values
that really are the engine's output, so the page's self-check means something.
The page's own arithmetic runs in a browser; the self-check embedded here is
what verifies it there.
"""
import atexit
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import traceback

from engine import (
    IPESimulation,
    PHASE1_COUNTRIES, PHASE1_GOODS,
    PHASE2_COUNTRIES, PHASE2_GOODS,
)
import calculator

PASS, FAIL = [], []
KEEP = ["Sabine", "Bosque", "Llano", "Trinity"]
# Scratch space only. Never the project's docs/ (the published page) and
# never rounds/ (the only copy of the class data).
TMP = tempfile.mkdtemp(prefix="ipe_calc_test_")
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def p2_sim(names=KEEP):
    return IPESimulation({k: PHASE2_COUNTRIES[k] for k in names},
                         PHASE2_GOODS, phase=2)


def export(sim, sub):
    path = os.path.join(TMP, sub, "index.html")
    calculator.export_calculator(sim, path, verbose=False)
    return path


def page(path):
    html = open(path, encoding="utf-8").read()
    m = re.search(r'<script type="application/json" id="sim-data">(.*?)</script>',
                  html, re.S)
    return html, (json.loads(m.group(1)) if m else None), (m.group(1) if m else "")


def test_one_button_per_country():
    print("\n[1] one country option per country in the simulation")
    for names in (KEEP[:3], KEEP, list(PHASE2_COUNTRIES)):
        sim = p2_sim(names)
        html, data, _ = page(export(sim, f"n{len(names)}"))
        check(f"  {len(names)} countries -> {len(names)} options",
              data is not None and len(data["countries"]) == len(names),
              str(None if data is None else len(data["countries"])))
        check(f"  {len(names)} countries: alphabetical",
              [c["name"] for c in data["countries"]] == sorted(names))
    check("  placeholder filled in", calculator.PLACEHOLDER not in html)
    folder = os.path.join(TMP, f"n{len(PHASE2_COUNTRIES)}")
    check("  .nojekyll written beside the page",
          os.path.exists(os.path.join(folder, ".nojekyll")))
    check("  nothing else written",
          sorted(os.listdir(folder)) == [".nojekyll", "index.html"],
          str(sorted(os.listdir(folder))))


def test_numbers_match_live_sim():
    print("\n[2] every number on the page is the simulation's own")
    sim = p2_sim()
    _, data, _ = page(export(sim, "match"))
    ok = True
    for c in data["countries"]:
        cfg = sim.countries[c["name"]]
        ok &= c["labor"] == cfg["labor"] and c["capital"] == cfg["capital"]
        for g in sim.goods:
            for key in ("tfp", "labor_share", "capital_share"):
                ok &= c["tech"][g][key] == cfg["tech"][g][key]
    check("  endowments and technology identical", ok)
    check("  goods in engine order", data["goods"] == list(sim.goods))
    check("  uses capital in Phase 2", data["uses_capital"] is True)
    check("  dated", bool(data["updated"]))


def test_reexport_follows_shocks():
    print("\n[3] a re-export after a shock shows the shocked economy")
    sim = p2_sim()
    before = sim.countries["Llano"]["tech"]["wine"]["tfp"]
    with contextlib.redirect_stdout(io.StringIO()):
        sim.inject_productivity_surge("Llano", "wine", 1.5)
        sim.inject_shock("Capital flight", {"Trinity": {"capital": 150}})
    _, data, _ = page(export(sim, "shock"))
    by = {c["name"]: c for c in data["countries"]}
    check("  Llano wine TFP x1.5",
          abs(by["Llano"]["tech"]["wine"]["tfp"] - before * 1.5) < 1e-12,
          str(by["Llano"]["tech"]["wine"]["tfp"]))
    check("  Trinity capital 150", by["Trinity"]["capital"] == 150)
    check("  others untouched",
          by["Bosque"]["tech"]["wine"]["tfp"]
          == PHASE2_COUNTRIES["Bosque"]["tech"]["wine"]["tfp"])
    llano = [k for k in data["checks"] if k["country"] == "Llano"]
    dec = {"production": {"labor": llano[0]["labor"],
                          "capital": llano[0]["capital"]}}
    fresh = sim._compute_production({n: dec for n in sim.countries})["Llano"]
    check("  check values use the shocked TFP",
          abs(llano[0]["expected"]["wine"] - fresh["wine"]) < 1e-12)


def test_check_values_are_engine_output():
    print("\n[4] the self-check compares against real engine output")
    sim = p2_sim()
    _, data, _ = page(export(sim, "checks"))
    ok, zero_cover = True, set()
    for k in data["checks"]:
        dec = {"production": {"labor": k["labor"], "capital": k["capital"]}}
        out = sim._compute_production({n: dec for n in sim.countries})[k["country"]]
        ok &= all(abs(out[g] - k["expected"][g]) < 1e-12 for g in sim.goods)
        if any(k["expected"][g] == 0 for g in sim.goods):
            zero_cover.add(k["country"])
        ok &= all(v >= 0 for v in k["labor"].values())
        ok &= all(v >= 0 for v in k["capital"].values())
    check("  every check value equals _compute_production", ok)
    check("  every country's checks include a zero-output sector",
          zero_cover == set(sim.countries), str(sorted(zero_cover)))
    per = {n: sum(k["country"] == n for k in data["checks"]) for n in sim.countries}
    check("  at least 3 checks per country", min(per.values()) >= 3, str(per))


def test_phase1_export():
    print("\n[5] Phase 1 (labor only) exports cleanly")
    sim = IPESimulation({k: PHASE1_COUNTRIES[k] for k in KEEP}, PHASE1_GOODS,
                        phase=1)
    _, data, _ = page(export(sim, "p1"))
    check("  no capital", data["uses_capital"] is False
          and all(c["capital"] is None for c in data["countries"]))
    s = next(c for c in data["countries"] if c["name"] == "Sabine")
    check("  output = labor x productivity (share 1, no capital)",
          all(s["tech"][g] == {"tfp": PHASE1_COUNTRIES["Sabine"]["productivity"][g],
                               "labor_share": 1.0, "capital_share": 0.0}
              for g in PHASE1_GOODS))
    k = next(k for k in data["checks"] if k["country"] == "Sabine")
    want = {g: k["labor"][g] * PHASE1_COUNTRIES["Sabine"]["productivity"][g]
            for g in PHASE1_GOODS}
    check("  Phase 1 check values = labor x productivity",
          all(abs(k["expected"][g] - want[g]) < 1e-12 for g in PHASE1_GOODS))


def test_data_cannot_break_the_page():
    print("\n[6] odd country names cannot close the data block early")
    odd = "A</script><b>"
    sim = IPESimulation({odd: PHASE2_COUNTRIES["Bosque"],
                         "Llano": PHASE2_COUNTRIES["Llano"]}, PHASE2_GOODS,
                        phase=2)
    html, data, raw = page(export(sim, "odd"))
    check("  no '<' inside the data block", "<" not in raw)
    check("  name survives the round trip",
          data is not None and odd in [c["name"] for c in data["countries"]])
    check("  still exactly two script blocks", html.count("</script>") == 2,
          str(html.count("</script>")))


def test_guards_and_defaults():
    print("\n[7] template guard, default location, engine delegate")
    bad = os.path.join(TMP, "no_placeholder.html")
    with open(bad, "w", encoding="utf-8") as f:
        f.write("<html>no data slot</html>")
    real, calculator.TEMPLATE = calculator.TEMPLATE, bad
    try:
        try:
            calculator.export_calculator(p2_sim(), os.path.join(TMP, "x.html"),
                                         verbose=False)
            raised = False
        except RuntimeError:
            raised = True
    finally:
        calculator.TEMPLATE = real
    check("  a template without the data slot is refused", raised)
    here = os.path.dirname(os.path.abspath(calculator.__file__))
    check("  default is docs/index.html in the project",
          os.path.normpath(calculator.DEFAULT_PATH)
          == os.path.join(here, "docs", "index.html"))
    path = os.path.join(TMP, "delegate", "index.html")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        got = p2_sim().export_calculator(path)
    check("  sim.export_calculator(path) writes there", got == path
          and os.path.exists(path))
    check("  and says how to publish", "commit and push" in out.getvalue())


def main():
    for t in [test_one_button_per_country, test_numbers_match_live_sim,
              test_reexport_follows_shocks, test_check_values_are_engine_output,
              test_phase1_export, test_data_cannot_break_the_page,
              test_guards_and_defaults]:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAIL.append((t.__name__, "exception"))
    print(f"\n{'='*60}")
    print(f"  PASSED: {len(PASS)}")
    print(f"  FAILED: {len(FAIL)}")
    if FAIL:
        for name, detail in FAIL:
            print(f"   - {name}: {detail}")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
