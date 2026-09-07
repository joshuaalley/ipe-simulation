"""
Stress test classroom.py -- the projection scoreboard and spreadsheet round I/O.

Heavy on the ways a workbook goes wrong in a live class: empty trades, blank
spacer rows, a stale template still listing dropped countries, and headers that
do not match the phase. Every one of those used to fail silently or with a
message that blamed the data instead of the headers.
"""
import atexit
import os
import shutil
import sys
import tempfile
import traceback

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from engine import (
    IPESimulation,
    PHASE1_COUNTRIES, PHASE1_GOODS,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    build_firm_roster,
)
import classroom

PASS, FAIL = [], []
KEEP = ["Sabine", "Bosque", "Llano", "Trinity"]
# Scratch space for test workbooks. Never the project directory, and never
# rounds/ -- that holds the real transcribed class data and is the only copy.
# Isolated per run, and removed on exit so repeated runs don't litter TEMP.
TMP = tempfile.mkdtemp(prefix="ipe_test_")
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def p1_sim():
    return IPESimulation({k: PHASE1_COUNTRIES[k] for k in KEEP},
                         PHASE1_GOODS, phase=1)


def p1_production():
    return [{"country": n, "cloth": PHASE1_COUNTRIES[n]["labor"] / 2,
             "wine": PHASE1_COUNTRIES[n]["labor"] / 2} for n in KEEP]


def write(path, book):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for sheet, df in book.items():
            df.to_excel(xl, sheet_name=sheet, index=False)


def prepared(sim, mutate=None, name="wb.xlsx"):
    """Template with production filled in, then optionally mangled."""
    path = os.path.join(TMP, name)
    sim.write_round_template(path)
    book = pd.read_excel(path, sheet_name=None)
    book["production"] = pd.DataFrame(p1_production())
    if mutate:
        mutate(book)
    write(path, book)
    return path


def load_error(sim, path):
    try:
        sim.load_round(path)
        return None
    except ValueError as e:
        return str(e)


# ────────────────────────────────────────────────────────────────────
# 1. the ordinary path
# ────────────────────────────────────────────────────────────────────
def test_round_trip():
    print("\n[1] template -> fill -> load -> run")
    sim = p1_sim()
    path = prepared(sim)
    kw = sim.load_round(path)
    check("  returns decisions + trades", set(kw) == {"decisions", "trades"})
    check("  all four countries parsed", len(kw["decisions"]) == 4)
    check("  autarky round has no trades", kw["trades"] == [])
    sim.run_round(**kw)
    check("  round runs", sim.round_num == 1)


def test_valid_trade_parses():
    print("\n[2] a real trade survives the round trip")
    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__("trades", pd.DataFrame([
        {"exporter": "Bosque", "importer": "Trinity", "good_out": "cloth",
         "qty_out": 20, "good_in": "wine", "qty_in": 15}])))
    kw = sim.load_round(path)
    check("  trade parsed as a 6-tuple",
          kw["trades"] == [("Bosque", "Trinity", "cloth", 20.0, "wine", 15.0)],
          str(kw["trades"]))


# ────────────────────────────────────────────────────────────────────
# 2. things that should be tolerated, not rejected
# ────────────────────────────────────────────────────────────────────
def test_empty_and_blank_trades_tolerated():
    print("\n[3] empty / blank trade sheets are fine (autarky rounds)")
    sim = p1_sim()
    kw = sim.load_round(prepared(sim))
    check("  empty trades sheet -> no trades", kw["trades"] == [])

    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__("trades", pd.DataFrame(
        [{c: None for c in b["trades"].columns}] * 3)))
    kw = sim.load_round(path)
    check("  blank spacer rows ignored", kw["trades"] == [])

    sim = p1_sim()
    path = prepared(sim, lambda b: b.pop("trades"))
    kw = sim.load_round(path)
    check("  deleted trades sheet tolerated", kw["trades"] == [])


# ────────────────────────────────────────────────────────────────────
# 3. things that must fail loudly, naming the headers
# ────────────────────────────────────────────────────────────────────
def test_renamed_trade_headers_rejected():
    print("\n[4] renamed trade headers are caught, not silently dropped")
    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__("trades", pd.DataFrame([
        {"from": "Bosque", "to": "Trinity", "good_out": "cloth",
         "qty_out": 20, "good_in": "wine", "qty_in": 15}])))
    err = load_error(sim, path)
    check("  raises rather than dropping the trade", err is not None)
    if err:
        check("  names the missing columns",
              "exporter" in err and "importer" in err, err[:120])
        check("  shows what was found instead", "from" in err, err[:120])


def test_wrong_phase_headers_rejected():
    print("\n[5] wrong-phase production headers are caught")
    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__("production", pd.DataFrame(
        [{"country": n, "labor_cloth": 50, "labor_wine": 50} for n in KEEP])))
    err = load_error(sim, path)
    check("  raises instead of loading zeros", err is not None)
    if err:
        check("  blames the headers, not the numbers",
              "missing column" in err and "cloth" in err, err[:160])


def test_dropped_country_rows_rejected():
    print("\n[6] a stale template listing dropped countries is caught")
    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__(
        "production", pd.DataFrame(p1_production()
                                   + [{"country": "Brazos", "cloth": 75,
                                       "wine": 75}])))
    err = load_error(sim, path)
    check("  raises on an unknown country", err is not None)
    if err:
        check("  names the offender", "Brazos" in err, err[:160])
        check("  lists the countries actually in play",
              "Sabine" in err and "Trinity" in err, err[:160])
        check("  says what to do about it",
              "Delete that row" in err or "regenerate" in err, err[:160])


def test_missing_countries_reported():
    print("\n[7] a country with no row is reported")
    sim = p1_sim()
    path = prepared(sim, lambda b: b.__setitem__(
        "production", pd.DataFrame(p1_production()[:2])))
    err = load_error(sim, path)
    check("  raises", err is not None)
    if err:
        check("  names every missing country",
              err.count("no row for") == 2, err[:160])


# ────────────────────────────────────────────────────────────────────
# 4. later phases through the spreadsheet
# ────────────────────────────────────────────────────────────────────
def test_phase7_round_trip():
    print("\n[8] Phase 7 round trip on a reduced country set")
    countries = {k: PHASE2_COUNTRIES[k] for k in KEEP}
    firms = build_firm_roster(KEEP, n_firms=11, verbose=False)
    sim = IPESimulation(countries, PHASE2_GOODS, phase=2)
    bal = {n: {"production": {
        "labor": {g: c["labor"] / 3 for g in PHASE2_GOODS},
        "capital": {g: c["capital"] / 3 for g in PHASE2_GOODS}}}
        for n, c in countries.items()}
    sim.run_round(bal, [])
    sim.upgrade_to_phase3(firms)
    fd = {f: {"scale": 10, "relocate_to": None, "export": False}
          for f in sim.firms}
    sim.run_round(bal, [], firm_decisions=fd)
    sim.award_reserve_currency()
    sim.upgrade_to_phase5()
    sim.run_round(bal, [], firm_decisions=fd)
    sim.upgrade_to_phase6()
    sim.run_round(bal, [], firm_decisions=fd,
                  debt_decisions={n: {"borrow": 4} for n in countries})
    sim.upgrade_to_phase7()

    path = os.path.join(TMP, "p7.xlsx")
    sim.write_round_template(path)
    book = pd.read_excel(path, sheet_name=None)
    book["production"] = pd.DataFrame([{
        "country": n,
        **{f"labor_{g}": c["labor"] / 3 for g in PHASE2_GOODS},
        **{f"capital_{g}": c["capital"] / 3 for g in PHASE2_GOODS}}
        for n, c in countries.items()])
    book["firms"] = pd.DataFrame([
        {"firm": f, "scale": 12, "relocate_to": "", "export": "yes"}
        for f in sim.firm_config])
    book["finance"] = pd.DataFrame([
        {"country": n, "fx_regime": "float", "capital_controls": "no",
         "independent_monetary": "yes", "money_supply_growth": 0.02,
         "borrow": 3, "repay": 0, "default": "no", "join_wto": "yes"}
        for n in countries])
    write(path, book)

    kw = sim.load_round(path)
    check("  every decision kind returned",
          set(kw) == {"decisions", "trades", "firm_decisions",
                      "monetary_decisions", "debt_decisions",
                      "institutional_decisions"}, str(sorted(kw)))
    check("  firm sheet honoured", len(kw["firm_decisions"]) == 11)
    check("  'yes' parses as a bool",
          kw["firm_decisions"][list(kw["firm_decisions"])[0]]["export"] is True)
    sim.run_round(**kw)
    check("  Phase 7 round runs", sim.round_num == 5)


# ────────────────────────────────────────────────────────────────────
# 5. scoreboard
# ────────────────────────────────────────────────────────────────────
def test_scoreboard():
    print("\n[9] projection scoreboard")
    sim = p1_sim()
    sim.run_round(**sim.load_round(prepared(sim)))
    html = classroom.scoreboard_html(sim, scale=1.4)
    check("  renders a table", "<table" in html and "Round 1" in html)
    check("  forces a light background", "background:#ffffff" in html)
    check("  scale drives the font size", "font-size:28px" in html)
    core = classroom.scoreboard_html(sim, columns="core")
    check("  columns='core' trims to welfare + gains",
          core.count("<th") == 3, str(core.count("<th")))
    text = classroom.scoreboard_text(sim)
    check("  text fallback works", "ROUND 1" in text and "Bosque" in text)



# ────────────────────────────────────────────────────────────────────
# 6. play_round: the one-call classroom workflow
# ────────────────────────────────────────────────────────────────────
def test_play_round():
    print("\n[10] play_round writes, then plays, then refuses to double-play")
    import shutil
    d = os.path.join(TMP, "play_round_test")
    shutil.rmtree(d, ignore_errors=True)
    wb = os.path.join(d, "round01.xlsx")
    sim = p1_sim()

    sim.play_round(wb)
    check("  1st call writes a blank workbook", os.path.exists(wb))
    check("  1st call does not advance the round", sim.round_num == 0)

    book = pd.read_excel(wb, sheet_name=None)
    book["production"] = pd.DataFrame(p1_production())
    write(wb, book)

    sim.play_round(wb)
    check("  2nd call plays the round", sim.round_num == 1)

    sim.play_round(wb)
    check("  accidental re-run does not double-play", sim.round_num == 1)

    sim.play_round(wb, replay=True)
    check("  replay=True plays it again on purpose", sim.round_num == 2)

    before = pd.read_excel(wb, sheet_name="production").to_dict()
    sim.play_round(wb)
    after = pd.read_excel(wb, sheet_name="production").to_dict()
    check("  a filled workbook is never overwritten", before == after)
    shutil.rmtree(d, ignore_errors=True)


def main():
    for t in [test_round_trip, test_valid_trade_parses,
              test_empty_and_blank_trades_tolerated,
              test_renamed_trade_headers_rejected,
              test_wrong_phase_headers_rejected,
              test_dropped_country_rows_rejected,
              test_missing_countries_reported,
              test_phase7_round_trip, test_scoreboard,
              test_play_round]:
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
