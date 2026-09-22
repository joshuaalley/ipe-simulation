"""
Stress test classroom.py -- the projection scoreboard and spreadsheet round I/O.

Heavy on the ways a workbook goes wrong in a live class: empty trades, blank
spacer rows, a stale template still listing dropped countries, and headers that
do not match the phase. Every one of those used to fail silently or with a
message that blamed the data instead of the headers.
"""
import atexit
import contextlib
import io
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
    check("  firm sheet honoured",
          len(kw["firm_decisions"]) == len(sim.firm_config),
          f"{len(kw['firm_decisions'])} of {len(sim.firm_config)}")
    check("  'yes' parses as a bool",
          kw["firm_decisions"][list(kw["firm_decisions"])[0]]["export"] is True)
    sim.run_round(**kw)
    check("  Phase 7 round runs", sim.round_num == 5)


def test_compensation_and_side_payments_round_trip():
    print("\n[8c] compensation and side payments through the workbook (Phase 2)")
    countries = {k: PHASE2_COUNTRIES[k] for k in KEEP}
    sim = IPESimulation(countries, PHASE2_GOODS, phase=2)
    path = os.path.join(TMP, "comp.xlsx")
    sim.write_round_template(path)
    book = pd.read_excel(path, sheet_name=None)
    check("  template has a side_payments sheet", "side_payments" in book,
          str(list(book)))
    check("  ...and a compensation column on production",
          "compensation_pct" in book["production"].columns,
          str(list(book["production"].columns)))
    rows = []
    for n, c in countries.items():
        r = {"country": n, "compensation_pct": 10 if n == "Llano" else 0}
        for g in PHASE2_GOODS:
            r[f"labor_{g}"] = c["labor"] / 3
            r[f"capital_{g}"] = c["capital"] / 3
        rows.append(r)
    book["production"] = pd.DataFrame(rows)
    book["side_payments"] = pd.DataFrame(
        [{"donor": "Trinity", "recipient": "Llano", "good": "wine", "qty": 5}])
    write(path, book)
    kw = sim.load_round(path)
    check("  compensation typed as 10 read as 10%",
          abs(kw["decisions"]["Llano"]["compensation"] - 0.10) < 1e-12)
    check("  countries that left it blank have none",
          "compensation" not in kw["decisions"]["Bosque"])
    check("  side payment parsed",
          kw.get("side_payments") == [("Trinity", "Llano", "wine", 5.0)],
          str(kw.get("side_payments")))
    r = sim.run_round(**kw)
    appr = r["results"]["Llano"]["approval"]
    check("  the round runs and both reach approval",
          appr["compensation_share"] == 0.10 and appr["net_side_payments"] == 5.0)
    bad = pd.read_excel(path, sheet_name=None)
    bad["side_payments"] = pd.DataFrame(
        [{"donor": "Atlantis", "recipient": "Llano", "good": "wine", "qty": 5}])
    write(path, bad)
    err = load_error(sim, path)
    check("  an unknown donor is named, not dropped",
          err is not None and "Atlantis" in err, str(err)[:90])


def filled(sim, folder, n):
    """A workbook for round n, production filled in, inside `folder`."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"round{n:02d}.xlsx")
    sim.write_round_template(path)
    book = pd.read_excel(path, sheet_name=None)
    book["production"] = pd.DataFrame(p1_production())
    write(path, book)
    return path


def test_autosave_and_resume():
    print("\n[8d] every round saves itself; resume picks up the newest")
    sim = p1_sim()
    folder = os.path.join(TMP, "auto", "rounds")
    path = filled(sim, folder, 1)
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        sim.play_round(path)
    snap = classroom.state_path(path)
    check("  a snapshot lands beside the workbook",
          os.path.exists(snap)
          and os.path.dirname(snap) == os.path.join(folder, "state"), snap)
    check("  and the round says so", "saved:" in quiet.getvalue(),
          quiet.getvalue()[:80])
    back = classroom.resume(os.path.join(folder, "state"), verbose=False)
    check("  restores the same state",
          back.round_num == sim.round_num and back.phase == sim.phase
          and len(back.history) == len(sim.history)
          and all(abs(back.countries[c]["approval"]
                      - sim.countries[c]["approval"]) < 1e-12 for c in KEEP))

    # a second round: resume takes the higher number, not the older file
    path2 = filled(sim, folder, 2)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.play_round(path2)
    newest = classroom.latest_state(os.path.join(folder, "state"))
    check("  latest_state picks the highest round", newest.endswith("round02.json"),
          str(newest))
    check("  resume lands on round 2",
          classroom.resume(os.path.join(folder, "state"), verbose=False).round_num == 2)
    check("  one specific round can be reopened",
          classroom.resume(os.path.join(folder, "state", "round01.json"),
                           verbose=False).round_num == 1)
    with open(os.path.join(folder, "state", "notes.json"), "w") as f:
        f.write("{}")
    check("  an unnumbered file doesn't hijack the pick",
          classroom.latest_state(os.path.join(folder, "state")).endswith("round02.json"))


def test_resume_without_a_snapshot():
    print("\n[8e] the first class of a term has nothing to resume")
    fresh = p1_sim()
    got = classroom.resume(os.path.join(TMP, "empty", "state"), default=fresh,
                           verbose=False)
    check("  returns the simulation from the setup cell", got is fresh)
    check("  and says nothing is there",
          classroom.latest_state(os.path.join(TMP, "empty", "state")) is None)


def test_a_failed_save_never_costs_the_round():
    print("\n[8f] a failed snapshot does not cost the round")
    sim = p1_sim()
    folder = os.path.join(TMP, "blocked", "rounds")
    path = filled(sim, folder, 1)
    with open(os.path.join(folder, "state"), "w") as f:   # a FILE where the dir goes
        f.write("in the way")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = sim.play_round(path)
    out = buf.getvalue()
    check("  the round still played", result is not None and sim.round_num == 1)
    check("  the scoreboard still showed", "ROUND 1" in out, out[:120])
    check("  and it warns instead of raising", "could not save" in out,
          out[:200])

    sim2 = p1_sim()
    path2 = filled(sim2, os.path.join(TMP, "off", "rounds"), 1)
    with contextlib.redirect_stdout(io.StringIO()):
        sim2.play_round(path2, autosave=False)
    check("  autosave=False writes nothing",
          not os.path.exists(classroom.state_path(path2)))


def test_run_all_after_resume_is_safe():
    print("\n[8g] running the notebook from the top after a resume is safe")
    sim = p1_sim()
    folder = os.path.join(TMP, "runall", "rounds")
    for n in (1, 2):
        with contextlib.redirect_stdout(io.StringIO()):
            sim.play_round(filled(sim, folder, n))
    with contextlib.redirect_stdout(io.StringIO()):
        back = classroom.resume(os.path.join(folder, "state"))
        # the notebook re-runs every round cell from the top, flags and all
        back.play_round(os.path.join(folder, "round01.xlsx"), replay=True)
        back.play_round(os.path.join(folder, "round02.xlsx"))
    check("  earlier rounds re-project instead of replaying", back.round_num == 2)
    check("  history is unchanged", len(back.history) == 2)
    path4 = filled(back, folder, 4)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        back.play_round(path4)
    check("  a round from further ahead waits its turn",
          back.round_num == 2 and "comes first" in buf.getvalue(),
          buf.getvalue()[-120:])
    with contextlib.redirect_stdout(io.StringIO()):
        back.play_round(filled(back, folder, 3))
    check("  the next round plays normally", back.round_num == 3)


def test_phase5_finance_sheet():
    print("\n[8b] Phase 5 finance sheet: three choices, typed the way people type")
    countries = {k: PHASE2_COUNTRIES[k] for k in KEEP}
    firms = build_firm_roster(KEEP, n_firms=11, verbose=False)
    sim = IPESimulation(countries, PHASE2_GOODS, phase=2)
    bal = {n: {"production": {
        "labor": {g: c["labor"] / 3 for g in PHASE2_GOODS},
        "capital": {g: c["capital"] / 3 for g in PHASE2_GOODS}}}
        for n, c in countries.items()}
    sim.run_round(bal, [])
    sim.upgrade_to_phase3(firms)
    fd = {f: {"scale": 10, "relocate_to": None, "export": False} for f in sim.firms}
    sim.run_round(bal, [], firm_decisions=fd)
    sim.award_reserve_currency()
    sim.upgrade_to_phase5()

    path = os.path.join(TMP, "p5.xlsx")
    sim.write_round_template(path)
    fin = pd.read_excel(path, sheet_name="finance")
    check("  template has no independent_monetary column",
          "independent_monetary" not in fin.columns, str(list(fin.columns)))
    check("  regimes pre-filled as float", set(fin["fx_regime"]) == {"float"})

    book = pd.read_excel(path, sheet_name=None)
    book["production"] = pd.DataFrame([{
        "country": n,
        **{f"labor_{g}": c["labor"] / 3 for g in PHASE2_GOODS},
        **{f"capital_{g}": c["capital"] / 3 for g in PHASE2_GOODS}}
        for n, c in countries.items()])
    book["firms"] = pd.DataFrame([
        {"firm": f, "scale": 10, "relocate_to": "", "export": "no"}
        for f in sim.firm_config])
    rows = [{"country": n, "fx_regime": "float", "capital_controls": "no",
             "money_supply_growth": 0} for n in countries]
    rows[0].update({"fx_regime": "Peg", "capital_controls": "yes",
                    "money_supply_growth": 5})          # typed as a percent
    book["finance"] = pd.DataFrame(rows)
    write(path, book)
    kw = sim.load_round(path)
    first = kw["monetary_decisions"][rows[0]["country"]]
    check("  'Peg' read as peg", first["fx_regime"] == "peg")
    check("  money growth typed as 5 read as 5%",
          abs(first["money_supply_growth"] - 0.05) < 1e-12)
    sim.run_round(**kw)
    check("  the round runs", sim.round_num == 3)


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
    check("  columns='core' is Country + welfare + gains + approval",
          core.count("<th") == 4, str(core.count("<th")))
    check("  approval is on the board", "Approval" in core)
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
    check("  replay=True cannot rewrite history (round01 stays round 1)",
          sim.round_num == 1)

    other = os.path.join(d, "demo.xlsx")        # not named roundNN
    sim.write_round_template(other)
    book = pd.read_excel(other, sheet_name=None)
    book["production"] = pd.DataFrame(p1_production())
    write(other, book)
    sim.play_round(other)
    sim.play_round(other, replay=True)
    check("  for other names, replay=True still plays it again", sim.round_num == 3)

    before = pd.read_excel(wb, sheet_name="production").to_dict()
    sim.play_round(wb)
    after = pd.read_excel(wb, sheet_name="production").to_dict()
    check("  a filled workbook is never overwritten", before == after)
    shutil.rmtree(d, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────
# 7. configuration drift: the simulation, not the spreadsheet, is wrong
# ────────────────────────────────────────────────────────────────────
def test_phase_mismatch_diagnosed():
    print("\n[11] a Phase-2 workbook loaded at Phase 1 blames the phase")
    # build a Phase 2 template, then try to load it from a Phase 1 sim --
    # exactly what happens when a kernel restart skips the upgrade cell
    countries = {k: PHASE2_COUNTRIES[k] for k in KEEP}
    p2 = IPESimulation(countries, PHASE2_GOODS, phase=2)
    path = os.path.join(TMP, "phase_mismatch.xlsx")
    p2.write_round_template(path)

    p1 = p1_sim()
    err = load_error(p1, path)
    check("  raises", err is not None)
    if err:
        check("  names both phases",
              "Phase 2" in err and "Phase 1" in err, err[:140])
        check("  exonerates the headers",
              "headers are fine" in err, err[:140])
        check("  says to run the upgrade cell",
              "upgrade cell" in err, err[:140])
        check("  does NOT blame missing columns",
              "missing column" not in err, err[:140])


def test_country_metadata_does_not_block_valid_rows():
    print("\n[12] a template made by a bigger sim still loads valid rows")
    # Exactly the real case: round01.xlsx was generated while the notebook
    # still built a six-country sim, then filled in for the four in play.
    six = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)
    path = os.path.join(TMP, "six_country_template.xlsx")
    six.write_round_template(path)
    book = pd.read_excel(path, sheet_name=None)
    book["production"] = pd.DataFrame(p1_production())      # four rows only
    write(path, book)

    four = p1_sim()
    err = load_error(four, path)
    check("  loads without error", err is None, (err or "")[:180])

    # ...while a genuinely wrong country in the ROWS is still caught
    book["production"] = pd.DataFrame(
        p1_production() + [{"country": "Pecos", "cloth": 25, "wine": 25}])
    write(path, book)
    err = load_error(four, path)
    check("  a dropped country typed into the rows is still rejected",
          err is not None and "Pecos" in err, (err or "")[:180])


def main():
    for t in [test_round_trip, test_valid_trade_parses,
              test_empty_and_blank_trades_tolerated,
              test_renamed_trade_headers_rejected,
              test_wrong_phase_headers_rejected,
              test_dropped_country_rows_rejected,
              test_missing_countries_reported,
              test_phase7_round_trip, test_phase5_finance_sheet,
              test_compensation_and_side_payments_round_trip,
              test_autosave_and_resume, test_resume_without_a_snapshot,
              test_a_failed_save_never_costs_the_round,
              test_run_all_after_resume_is_safe, test_scoreboard,
              test_play_round, test_phase_mismatch_diagnosed,
              test_country_metadata_does_not_block_valid_rows]:
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
