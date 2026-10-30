"""
Stress test the decision-file inbox (classroom.py, section 4) and the round
page that makes the files (calculator.py).

Students submit a round from the page: one JSON file per country, and per firm
whose owner changes anything, dropped into rounds/inbox. play_round builds the
round's workbook from them and plays it. What must hold:

  * the newest file per team wins; files from before the last round are stale
  * a trade happens only when both sides list it with the same terms
  * an absent country repeats last round; an absent firm does what it did
  * an unusable file is reported and treated as absent -- never half-applied
  * the built workbook plays through the ordinary loader, and files that
    arrive between the build and the play trigger a rebuild, not a stale play
  * paper rounds work exactly as before
  * a reserve-currency challenge reaches the finance sheet, the engine and
    the board, and never carries over to the next round
"""
import atexit
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from engine import (IPESimulation, PHASE2_COUNTRIES, PHASE2_GOODS,
                    build_firm_roster)
import calculator
import classroom

PASS, FAIL = [], []
KEEP = ["Bosque", "Llano", "Sabine", "Trinity"]
G = PHASE2_GOODS
TMP = tempfile.mkdtemp(prefix="ipe_inbox_test_")
atexit.register(shutil.rmtree, TMP, ignore_errors=True)


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*a, **k)
    return result, out.getvalue()


# ── a Phase 3 game one paper round in ───────────────────────────────────

ALLOC = {
    "Bosque":  ((40, 12, 8), (15, 5, 5)),
    "Llano":   ((10, 70, 20), (10, 50, 20)),
    "Sabine":  ((70, 20, 10), (20, 8, 7)),
    "Trinity": ((40, 20, 60), (30, 20, 150)),
}


def game(sub):
    """A four-country Phase 3 game whose round 1 was played from paper."""
    folder = os.path.join(TMP, sub, "rounds")
    os.makedirs(folder)
    sim = IPESimulation({c: PHASE2_COUNTRIES[c] for c in KEEP}, PHASE2_GOODS, phase=2)
    quiet(sim.upgrade_to_phase3, build_firm_roster(KEEP, verbose=False))
    quiet(sim.set_firm_owners, {"F1": "Sabine", "F11": "Bosque"})
    path = os.path.join(folder, "round01.xlsx")
    quiet(sim.play_round, path)                       # blank template
    book = pd.read_excel(path, sheet_name=None)
    prod = book["production"].set_index("country")
    for c, (L, K) in ALLOC.items():
        for g, l, k in zip(G, L, K):
            prod.loc[c, f"labor_{g}"] = l
            prod.loc[c, f"capital_{g}"] = k
    prod.loc["Llano", "compensation_pct"] = 10
    book["production"] = prod.reset_index()
    book["tariffs"] = pd.DataFrame([{"importer": "Sabine", "partner": "Trinity",
                                     "good": "machinery", "tariff": 20}])
    book["firms"]["scale"] = 30
    _write(path, book)
    quiet(sim.play_round, path)                       # plays round 1
    return sim, folder


def _write(path, book):
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for sheet, df in book.items():
            df.to_excel(xl, sheet_name=sheet, index=False)


def country_file(c, trades=(), tariffs=None, mnc=None, comp=0, phase=3, alloc=None, **extra):
    L, K = alloc or ALLOC[c]
    d = {"ipe": 1, "kind": "country", "country": c, "phase": phase,
         "made": "2026-09-30T14:00:00",
         "production": {"labor": dict(zip(G, L)), "capital": dict(zip(G, K))},
         "tariffs": tariffs or {}, "compensation_pct": comp, "mnc_tax_pct": mnc,
         "trades": [{"give": q1, "give_good": g1, "to": to, "get": q2, "get_good": g2}
                    for (to, q1, g1, q2, g2) in trades],
         "side_payments": []}
    d.update(extra)
    return d


def drop(folder, name, data):
    inbox = os.path.join(folder, "inbox")
    os.makedirs(inbox, exist_ok=True)
    with open(os.path.join(inbox, name), "w", encoding="utf-8") as f:
        f.write(data if isinstance(data, str) else json.dumps(data))
    time.sleep(0.02)                                   # distinct times


def all_four(folder, trades=None):
    trades = trades or {}
    for c in KEEP:
        drop(folder, f"{c}.json", country_file(c, trades.get(c, ())))


SWAP = {"Bosque": [("Trinity", 10, "cloth", 4, "machinery")],
        "Trinity": [("Bosque", 4, "machinery", 10, "cloth")]}


# ── tests ────────────────────────────────────────────────────────────────

def test_newest_wins_and_junk_is_set_aside():
    print("\n[1] the newest file per team wins; junk is set aside, never played")
    sim, folder = game("newest")
    drop(folder, "Bosque.json", country_file("Bosque", mnc=5))
    drop(folder, "Bosque (1).json", country_file("Bosque", mnc=15))
    drop(folder, "notes.json", "this is not json")
    drop(folder, "Atlantis.json", country_file("Bosque") | {"country": "Atlantis"})
    drop(folder, "F99.json", {"ipe": 1, "kind": "firm", "firm": "F99", "phase": 3,
                              "scale": 10, "relocate_to": None, "export": False})
    box = classroom.read_inbox(sim, os.path.join(folder, "inbox"))
    entry = box["teams"][("country", "Bosque")]
    check("  the later of two Bosque files is used",
          entry["file"] == "Bosque (1).json" and entry["data"]["mnc_tax_pct"] == 15)
    check("  the earlier one is recorded as replaced",
          [e["file"] for e in box["replaced"][("country", "Bosque")]] == ["Bosque.json"])
    why = dict(box["rejected"])
    check("  garbled, unknown-country and unknown-firm files are rejected",
          set(why) == {"notes.json", "Atlantis.json", "F99.json"}, str(why))


def test_stale_files_wait():
    print("\n[2] a file that reached the inbox before the last round is stale")
    sim, folder = game("stale")
    drop(folder, "Llano.json", country_file("Llano"))
    time.sleep(1.1)                        # played_at has one-second resolution
    quiet(sim.run_round, **sim.load_round(os.path.join(folder, "round01.xlsx")))
    path = os.path.join(folder, "round03.xlsx")
    box = classroom.read_inbox(sim, os.path.join(folder, "inbox"),
                               classroom._last_played_time(sim, path))
    check("  a file older than the last round is stale, not used",
          box["stale"] == ["Llano.json"] and not box["teams"], str(box["stale"]))
    _, out = quiet(sim.play_round, path)
    check("  stale files alone don't start a round from the inbox",
          "Blank workbook written" in out, out[:200])

    sim, folder = game("byhand")
    drop(folder, "Llano.json", country_file("Llano"))
    path = os.path.join(folder, "round02.xlsx")
    shutil.copy(os.path.join(folder, "round01.xlsx"), path)   # typed by hand
    _, out = quiet(sim.play_round, path)
    check("  a workbook typed by hand wins over waiting files, with a note",
          sim.round_num == 2 and "typed by hand" in out, out[:200])


def test_trades_need_both_sides():
    print("\n[3] a trade happens only when both sides list the same swap")
    legs = {
        "Bosque": [("Bosque", "Trinity", "cloth", 10.0, "machinery", 4.0),
                   ("Bosque", "Llano", "cloth", 5.0, "wine", 5.0),
                   ("Bosque", "Sabine", "wine", 2.0, "cloth", 3.0)],
        "Trinity": [("Trinity", "Bosque", "machinery", 4.0, "cloth", 10.0)],
        "Sabine": [("Sabine", "Bosque", "cloth", 3.0, "wine", 3.0)],
        "Llano": [],
    }
    confirmed, unconfirmed, mismatched = classroom.match_trades(legs)
    check("  matching terms from both sides: one trade",
          confirmed == [("Bosque", "Trinity", "cloth", 10.0, "machinery", 4.0)], str(confirmed))
    check("  listed by one side only: not confirmed",
          len(unconfirmed) == 1 and "Llano didn't list it" in unconfirmed[0], str(unconfirmed))
    check("  different terms: a mismatch, naming both versions",
          len(mismatched) == 1 and "2 wine" in mismatched[0] and "3 cloth" in mismatched[0],
          str(mismatched))
    twice = {"Bosque": [("Bosque", "Llano", "cloth", 5.0, "wine", 5.0)] * 2,
             "Llano": [("Llano", "Bosque", "wine", 5.0, "cloth", 5.0)]}
    c2, u2, _ = classroom.match_trades(twice)
    check("  the same swap twice needs two listings from each side",
          len(c2) == 1 and len(u2) == 1, f"{c2} {u2}")


def test_build_and_play():
    print("\n[4] first run builds and reports, second run plays and files away")
    sim, folder = game("play")
    all_four(folder, SWAP)
    drop(folder, "F1.json", {"ipe": 1, "kind": "firm", "firm": "F1", "phase": 3,
                             "scale": 20, "relocate_to": "Llano", "export": False})
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  first run builds the workbook and does not play",
          os.path.exists(path) and sim.round_num == 1 and "ROUND 2 INBOX" in out, out[:160])
    check("  the report counts the teams in",
          "4 of 4 countries" in out and "1 of 8 firms" in out and "1 confirmed" in out, out[:300])
    kw = sim.load_round(path)
    check("  the built workbook loads like a typed one",
          kw["trades"] == [("Bosque", "Trinity", "cloth", 10.0, "machinery", 4.0)]
          and kw["firm_decisions"]["F1"]["relocate_to"] == "Llano"
          and kw["firm_decisions"]["F3"]["scale"] == 30, str(kw["trades"]))
    drop(folder, "Sabine (1).json", country_file("Sabine", mnc=12))   # arrives late
    _, out = quiet(sim.play_round, path)
    check("  a file arriving after the build means rebuild, not play",
          sim.round_num == 1 and "ROUND 2 INBOX" in out)
    check("  ...and the rebuild uses it",
          abs(sim.load_round(path)["decisions"]["Sabine"]["mnc_tax"] - 0.12) < 1e-12)
    _, out = quiet(sim.play_round, path)
    check("  the next run plays the round", sim.round_num == 2)
    inbox = os.path.join(folder, "inbox")
    check("  the inbox is emptied into rounds/round02/",
          os.listdir(inbox) == [] and len(os.listdir(os.path.join(folder, "round02"))) == 6
          and "moved 6 decision file(s)" in out, str(os.listdir(inbox)))
    check("  the moved firm really moved", sim.firms["F1"]["host"] == "Llano")


def test_absent_teams_repeat_last_round():
    print("\n[5] an absent country repeats last round; an absent firm keeps going")
    sim, folder = game("absent")
    drop(folder, "Bosque.json", country_file("Bosque", SWAP["Bosque"]))
    drop(folder, "Llano.json", country_file("Llano"))
    drop(folder, "Sabine.json", country_file("Sabine"))
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  the report says Trinity repeats Round 1",
          "Trinity" in out and "repeating Round 1" in out, out[:400])
    kw = sim.load_round(path)
    L, K = ALLOC["Trinity"]
    check("  Trinity's production is last round's",
          kw["decisions"]["Trinity"]["production"]["labor"] == dict(zip(G, map(float, L)))
          and kw["decisions"]["Trinity"]["production"]["capital"] == dict(zip(G, map(float, K))))
    check("  Bosque's swap with absent Trinity doesn't happen",
          kw["trades"] == [] and "NOT CONFIRMED" in out)
    check("  a country that sent a file uses its own tariffs, not last round's",
          "tariffs" not in kw["decisions"]["Sabine"])
    check("  firms without files keep last round's scale",
          all(fd["scale"] == 30 and fd["relocate_to"] is None
              for fd in kw["firm_decisions"].values()))
    # a firm that moved last round keeps its scale from before the move
    sim.history[-1]["firms"]["F3"].update({"relocated": True, "scale": 0.0})
    check("  a firm that just moved goes back to full scale, not zero",
          classroom._standing_firm(sim, "F3")[0] == 40.0)


def test_unusable_files_are_never_half_applied():
    print("\n[6] an unusable file is reported and treated as absent")
    sim, folder = game("unusable")
    all_four(folder)
    drop(folder, "Llano.json", country_file("Llano", phase=1))                 # wrong era
    drop(folder, "Sabine.json", country_file("Sabine", mnc=80,                  # out of range
                                             tariffs={"Trinity": {"cloth": 40}}))
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  wrong phase: reported, and Llano repeats last round",
          "filled in for Phase 1" in out and "Llano" in out, out[:500])
    kw = sim.load_round(path)
    check("  out of range: reported, and none of that file is used",
          "MNC tax is 80" in out
          and kw["decisions"]["Sabine"].get("tariffs", {}) == {"Trinity": {"machinery": 0.2}},
          str(kw["decisions"]["Sabine"].get("tariffs")))


def test_paper_rounds_unchanged():
    print("\n[7] with an empty inbox, paper rounds work exactly as before")
    sim, folder = game("paper")
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  no files: the blank template, as always",
          "Blank workbook written" in out and not classroom._built_from_inbox(path))
    _, out = quiet(sim.play_round, path)
    check("  ...and a blank template still says so", "still blank" in out)


def test_money_debt_institutions_fields():
    print("\n[8] later phases: money, debt and institutions reach the finance sheet")
    sim, folder = game("later")
    quiet(sim.award_reserve_currency)
    quiet(sim.upgrade_to_phase5)
    quiet(sim.upgrade_to_phase6)
    quiet(sim.upgrade_to_phase7)
    for c in KEEP:
        extra = {}
        if c == "Bosque":
            extra = {"money": {"fx_regime": "peg", "capital_controls": True,
                               "money_supply_growth": 5},
                     "debt": {"borrow": 3, "repay": 0, "default": False},
                     "institutions": {"join_wto": True}}
        elif c == "Sabine":
            extra = {"money": {"fx_regime": "float", "capital_controls": False,
                               "money_supply_growth": -2}}
        elif c == "Trinity":
            extra = {"money": {"money_supply_growth": -10}}
        drop(folder, f"{c}.json", country_file(c, phase=7, **extra))
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)
    kw = sim.load_round(path)
    m = kw["monetary_decisions"]
    check("  Bosque's money choices arrive",
          m["Bosque"] == {"fx_regime": "peg", "capital_controls": True,
                          "money_supply_growth": 0.05}, str(m["Bosque"]))
    check("  tightening from the page (-2) arrives as -2%",
          m["Sabine"]["money_supply_growth"] == -0.02, str(m["Sabine"]))
    check("  the big tightening step (-10) arrives as -10%, a choice the engine takes",
          m["Trinity"]["money_supply_growth"] == -0.10
          and not sim._validate_monetary_decisions({"Trinity": m["Trinity"]}), str(m["Trinity"]))
    check("  a country that chose 'keep' keeps its current policy",
          m["Llano"]["fx_regime"] == sim._regime(sim._mon("Llano").get("fx_regime", "float"))
          and m["Llano"]["money_supply_growth"] == sim._mon("Llano").get("money_supply_growth", 0.0))
    check("  borrowing and WTO membership arrive",
          kw["debt_decisions"]["Bosque"]["borrow"] == 3
          and kw["institutional_decisions"]["Bosque"]["join_wto"] is True)


def test_report_writes_nothing():
    print("\n[9] sim.inbox_report() shows who's in and writes nothing")
    sim, folder = game("report")
    drop(folder, "Bosque.json", country_file("Bosque"))
    before = sorted(os.listdir(folder))
    _, out = quiet(sim.inbox_report, folder)
    check("  it reports", "files from 1 of 4 countries" in out, out[:120])
    check("  and writes nothing", sorted(os.listdir(folder)) == before)


def test_real_round8_through_the_inbox():
    print("\n[10] your Round 8 decisions, as files, build the same round")
    here = os.path.dirname(os.path.abspath(__file__))
    real = os.path.join(here, "rounds", "round08.xlsx")
    snap = os.path.join(here, "rounds", "state", "round07.json")
    roster = os.path.join(here, "rounds", "firms-roster.xlsx")
    if not all(os.path.exists(p) for p in (real, snap, roster)):
        print("  (class data not present -- skipped)")
        return
    folder = os.path.join(TMP, "real", "rounds")
    os.makedirs(os.path.join(folder, "state"))
    shutil.copy(snap, os.path.join(folder, "state"))
    shutil.copy(os.path.join(here, "rounds", "round07.xlsx"), folder)
    sim = IPESimulation.resume(folder=os.path.join(folder, "state"), verbose=False)
    own = pd.read_excel(roster)
    quiet(sim.upgrade_to_phase3, build_firm_roster(list(sim.countries), verbose=False))
    quiet(sim.set_firm_owners, {r["firm"]: [r["country-origin"]] for _, r in own.iterrows()})
    book = pd.read_excel(real, sheet_name=None)
    paper = sim.load_round(real)
    time.sleep(0.05)
    prod, tar, trades = book["production"].set_index("country"), book["tariffs"], book["trades"]
    for c in sim.countries:
        mine = [(t["importer"], t["qty_out"], t["good_out"], t["qty_in"], t["good_in"])
                for _, t in trades[trades["exporter"] == c].iterrows()]
        theirs = [(t["exporter"], t["qty_in"], t["good_in"], t["qty_out"], t["good_out"])
                  for _, t in trades[trades["importer"] == c].iterrows()]
        rates = {}
        for _, t in tar[tar["importer"] == c].iterrows():
            if float(t["tariff"]) > 0:
                rates.setdefault(t["partner"], {})[t["good"]] = float(t["tariff"])
        L = tuple(float(prod.loc[c, f"labor_{g}"]) for g in G)
        K = tuple(float(prod.loc[c, f"capital_{g}"]) for g in G)
        drop(folder, f"{c}.json", country_file(c, mine + theirs, rates,
                                               comp=float(prod.loc[c, "compensation_pct"]),
                                               alloc=(L, K)))
    firms = book["firms"].set_index("firm")
    for fid in ("F7", "F11"):
        rel = firms.loc[fid, "relocate_to"]
        drop(folder, f"{fid}.json", {"ipe": 1, "kind": "firm", "firm": fid, "phase": 3,
                                     "scale": float(firms.loc[fid, "scale"]),
                                     "relocate_to": None if pd.isna(rel) else rel,
                                     "export": False})
    path = os.path.join(folder, "round08.xlsx")
    quiet(sim.play_round, path)
    built = sim.load_round(path)
    nonzero = lambda t: {p: {g: r for g, r in gs.items() if r > 0}
                         for p, gs in (t or {}).items() if any(r > 0 for r in gs.values())}
    check("  production, tariffs and compensation match the paper round",
          all(built["decisions"][c]["production"] == paper["decisions"][c]["production"]
              and nonzero(built["decisions"][c].get("tariffs")) == nonzero(paper["decisions"][c].get("tariffs"))
              and built["decisions"][c].get("compensation", 0) == paper["decisions"][c].get("compensation", 0)
              for c in sim.countries))
    canon = lambda ts: sorted(classroom._canon((a, b, ga, float(qa), gb, float(qb)))
                              for a, b, ga, qa, gb, qb in ts)
    check("  the same six trades", canon(built["trades"]) == canon(paper["trades"]),
          str(built["trades"]))
    check("  the same firm decisions",
          built["firm_decisions"] == paper["firm_decisions"],
          str({f: (built["firm_decisions"][f], paper["firm_decisions"][f])
               for f in built["firm_decisions"]
               if built["firm_decisions"][f] != paper["firm_decisions"][f]}))


def test_page_data():
    print("\n[11] the round page carries every phase, the firms and the upload link")
    sim, folder = game("page")
    path = os.path.join(TMP, "page", "index.html")
    quiet(calculator.export_calculator, sim, path,
          inbox_url="https://www.dropbox.com/request/abc")
    data = calculator._page_data(path)
    check("  both production eras are on the page",
          set(data["eras"]) == {"1", "2"}
          and data["eras"]["1"]["uses_capital"] is False
          and len(data["eras"]["2"]["goods"]) == 3)
    f1 = next(f for f in data["firms"] if f["id"] == "F1")
    check("  firms carry their owners", f1["owners"] == ["Sabine"])
    check("  the rules come from the engine",
          data["rules"]["mnc_tax_max"] == 50 and data["rules"]["compensation_max"] == 25)
    stamp = os.path.getmtime(path)
    time.sleep(0.05)
    quiet(calculator.export_calculator, sim, path)
    check("  exporting again with nothing changed leaves the file alone",
          os.path.getmtime(path) == stamp)
    check("  ...and the upload link is kept",
          calculator._page_data(path)["inbox_url"] == "https://www.dropbox.com/request/abc")


def test_pass_through_listed_first():
    print("\n[12] passing on an import settles though its pair sorts first")
    # Round 12: Bosque buys machinery from Trinity and passes some to Sabine.
    # Bosque-Sabine sorts before Bosque-Trinity, so the relay comes up first.
    sim, folder = game("relay")
    relay = {"Bosque": [("Trinity", 20, "cloth", 15, "machinery"),
                        ("Sabine", 12, "machinery", 8, "wine")],
             "Trinity": [("Bosque", 15, "machinery", 20, "cloth")],
             "Sabine": [("Bosque", 8, "wine", 12, "machinery")]}
    no_machines = ((48, 12, 0), (20, 5, 0))             # and hosts no machinery firm
    for c in KEEP:
        drop(folder, f"{c}.json", country_file(
            c, relay.get(c, ()), alloc=no_machines if c == "Bosque" else None))
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)                          # builds
    order = [t[:2] for t in sim.load_round(path)["trades"]]
    check("  the relay is listed before the trade that supplies it",
          order == [("Bosque", "Sabine"), ("Bosque", "Trinity")], str(order))
    quiet(sim.play_round, path)                          # plays
    r = sim.history[-1]
    log = "\n".join(r["trade_log"])
    check("  both trades settle", sim.round_num == 2 and "FAILED" not in log
          and len(r["trades_executed"]) == 2, log)
    check("  Bosque keeps the 3 machinery it didn't pass on",
          abs(r["results"]["Bosque"]["consumption"]["machinery"] - 3.0) < 1e-9,
          str(r["results"]["Bosque"]["consumption"]))


def test_file_times_survive_excel():
    print("\n[13] Excel's rounding of a file time doesn't look like a new file")
    sim, folder = game("rounding")
    all_four(folder)
    inbox = os.path.join(folder, "inbox")
    base = int(time.time()) * 10**9
    for i, name in enumerate(sorted(os.listdir(inbox))):
        t = base + i * 10**9 + 123_499_000              # x.123499 s
        os.utime(os.path.join(inbox, name), ns=(t, t))
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)                          # builds
    book = pd.read_excel(path, sheet_name=None)
    book[classroom.INBOX_SHEET]["mtime"] += 2e-6         # x.123501: across a rounding line
    _write(path, book)
    check("  a time a hair off is the same file",
          not classroom._inbox_changed(path, inbox))
    quiet(sim.play_round, path)
    check("  ...so the next run plays the round instead of rebuilding it",
          sim.round_num == 2)


def test_reserve_challenge_through_the_inbox():
    print("\n[14] a reserve-currency challenge goes from the files to the board")
    sim, folder = game("reserve")
    quiet(sim.award_reserve_currency)
    quiet(sim.upgrade_to_phase5)
    holder = sim.reserve_currency_holder
    challenger, backer, third = [c for c in KEEP if c != holder]

    def submit(rnd, picks):
        for c in KEEP:
            extra = {"reserve_challenge": picks[c]} if c in picks else {}
            drop(folder, f"{c} r{rnd}.json", country_file(c, phase=5, **extra))
        path = os.path.join(folder, f"round{rnd:02d}.xlsx")
        _, out = quiet(sim.play_round, path)                       # builds
        return path, out

    path, out = submit(2, {challenger: challenger, backer: challenger, holder: holder})
    fin = pd.read_excel(path, sheet_name="finance").set_index("country")
    check("  the picks land in the finance sheet",
          fin.loc[challenger, "reserve_challenge"] == challenger
          and fin.loc[backer, "reserve_challenge"] == challenger
          and pd.isna(fin.loc[third, "reserve_challenge"]))
    check("  the report shows the challenge and the count before anything plays",
          f"{challenger} challenges {holder}, backed by {backer} -- 2 of 4 countries, "
          "3 needed" in out and f"{holder} already holds it" in out, out)
    check("  the loader passes the picks on",
          sim.load_round(path)["reserve_challenge"]
          == {challenger: challenger, backer: challenger, holder: holder})
    sim.challenge_rule = "welfare"
    _, lines, _ = classroom._plan_from_inbox(sim, path, os.path.join(folder, "inbox"))
    sim.challenge_rule = "majority"
    check("  under the 'welfare' rule the report says so instead of counting",
          any(f"{challenger} challenges" in l and "on cumulative welfare" in l
              for l in lines), "\n".join(lines))
    quiet(sim.play_round, path)                                    # plays
    board = classroom.scoreboard_html(sim)
    check("  two of four is not enough, and the board says so",
          sim.round_num == 2 and sim.reserve_currency_holder == holder
          and "Reserve currency" in board and "2 of 4" in board
          and f"{holder} keeps it" in board)

    path, out = submit(3, {challenger: challenger, backer: challenger, third: challenger})
    quiet(sim.play_round, path)
    check("  three of four wins; the board says from which round",
          sim.reserve_currency_holder == challenger
          and f"{challenger} holds it from Round 4" in classroom.scoreboard_text(sim),
          classroom.scoreboard_text(sim))
    back, _ = quiet(classroom.resume, os.path.join(folder, "state"), verbose=False)
    check("  the saved snapshot has the new holder", back.reserve_currency_holder == challenger)

    drop(folder, f"{backer} r4.json",
         country_file(backer, phase=5, reserve_challenge="Atlantis"))
    path = os.path.join(folder, "round04.xlsx")
    _, out = quiet(sim.play_round, path)                           # builds
    fin = pd.read_excel(path, sheet_name="finance").set_index("country")
    check("  a file naming no real country is unusable, and says why",
          "file unusable" in out and "'Atlantis' is not a country" in out, out)
    check("  picks never carry over: no file, no challenge",
          fin["reserve_challenge"].isna().all() and "reserve_challenge" not in sim.load_round(path))

    book = pd.read_excel(path, sheet_name=None)
    book["finance"]["reserve_challenge"] = book["finance"]["reserve_challenge"].astype(object)
    book["finance"].loc[0, "reserve_challenge"] = "Atlantis"
    typed = os.path.join(folder, "typed.xlsx")
    _write(typed, book)
    try:
        sim.load_round(typed)
        check("  a typed workbook naming no real country is refused", False, "no error")
    except ValueError as e:
        check("  a typed workbook naming no real country is refused",
              "reserve_challenge 'Atlantis' is not a country" in str(e), str(e))
    book["finance"] = book["finance"].drop(columns="reserve_challenge")
    _write(typed, book)
    check("  a workbook from before the column still loads, with no challenge",
          "reserve_challenge" not in sim.load_round(typed))


def phase5_game(sub):
    sim, folder = game(sub)
    quiet(sim.award_reserve_currency)
    quiet(sim.upgrade_to_phase5)
    return sim, folder


def test_bad_files_are_reported_not_fatal():
    print("\n[15] one bad file is reported and set aside; it never stops the build")
    sim, folder = game("badfiles")
    path = os.path.join(folder, "round02.xlsx")
    alloc_bad = ((-5, 45, 20), (15, 5, 5))
    cases = {
        "a negative number": (country_file("Bosque", alloc=alloc_bad), "below 0"),
        "workers that don't add up": (country_file("Bosque", alloc=((40, 12, 2), (15, 5, 5))),
                                      "workers add up to 54"),
        "capital that doesn't add up": (country_file("Bosque", alloc=((40, 12, 8), (15, 5, 1))),
                                        "capital adds up to 21"),
        "production that isn't a table": (country_file("Bosque") | {"production": 5}, "layout"),
        "tariffs that aren't a table": (country_file("Bosque") | {"tariffs": [1, 2]}, "layout"),
        "trades that aren't a list": (country_file("Bosque") | {"trades": 7}, "layout"),
        "a negative side payment": (country_file("Bosque") | {"side_payments": [
            {"qty": -5, "good": "cloth", "to": "Llano"}]}, "below 0"),
    }
    for c in KEEP[1:]:
        drop(folder, f"{c}.json", country_file(c))
    for label, (data, says) in cases.items():
        drop(folder, "Bosque.json", data)
        try:
            _, lines, _ = quiet(classroom._plan_from_inbox, sim, path,
                                os.path.join(folder, "inbox"))[0]
        except Exception as e:
            check(f"  {label}", False, f"{type(e).__name__}: {e}")
            continue
        row = next(l for l in lines if l.strip().startswith("Bosque"))
        check(f"  {label}: unusable, says why, repeats last round",
              "file unusable" in row and says in row and "repeating Round 1" in row, row)
    drop(folder, "Bosque.json", country_file("Bosque") | {"country": ["Bosque"]})
    _, lines, _ = quiet(classroom._plan_from_inbox, sim, path, os.path.join(folder, "inbox"))[0]
    check("  a file whose country is a list is set aside",
          any("Set aside Bosque.json" in l for l in lines), str(lines))

    sim, folder = game("badmove")
    all_four(folder)
    drop(folder, "F1.json", {"ipe": 1, "kind": "firm", "firm": "F1", "phase": 3,
                             "scale": 20, "relocate_to": "Sabine", "export": False})
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  a firm moving to its owner's home is reported, not left to stop the round",
          "can't move to Sabine" in out and "THIS ROUND CAN'T PLAY" not in out, out[-500:])
    _, out = quiet(sim.play_round, path)
    check("  ...and the round plays with the firm where it was",
          sim.round_num == 2 and sim.firms["F1"]["host"] != "Sabine")

    sim, folder = phase5_game("badmoney")
    sim._mon("Llano")["money_supply_growth"] = -0.10
    drop(folder, "Bosque.json", country_file("Bosque", phase=5, money={"money_supply_growth": 3}))
    drop(folder, "Llano.json", country_file("Llano", phase=5))
    drop(folder, "Sabine.json", country_file("Sabine", phase=4))
    drop(folder, "Trinity.json", country_file("Trinity", phase=5, money={"money_supply_growth": -5}))
    _, out = quiet(sim.inbox_report, folder)
    check("  a money growth that isn't a choice is reported",
          "money growth 3.0% is not one of the choices" in out, out)
    check("  a rate that repeats by default is flagged",
          "Llano sent no new rate, so last round's -10% repeats" in out, out)
    check("  ...not for a country that chose",
          "Trinity sent no new rate" not in out)
    check("  a file from the wrong phase of the same era is noted",
          "filled in for Phase 4, this is Phase 5" in out, out)


def test_dry_run_flags_what_wont_go_through():
    print("\n[16] the build report says what won't go through, while it can be fixed")
    sim, folder = game("dry")
    for c in KEEP:
        drop(folder, f"{c}.json", country_file(c))
    drop(folder, "Bosque.json", country_file("Bosque") | {"side_payments": [
        {"qty": 50, "good": "machinery", "to": "Sabine"}]})
    path = os.path.join(folder, "round02.xlsx")
    state = json.dumps(sim.get_state())
    _, out = quiet(sim.play_round, path)
    check("  a side payment that can't be covered is named before the round plays",
          "Won't go through" in out and "Bosque cannot pay 50 machinery" in out
          and sim.round_num == 1, out[-400:])
    check("  the trial leaves the game untouched", json.dumps(sim.get_state()) == state)
    check("  ...and shows no results", "elfare" not in out, out)
    _, out = quiet(sim.inbox_report, folder)
    check("  the report while teams are deciding says the same",
          "Bosque cannot pay 50 machinery" in out, out)
    drop(folder, "Bosque (1).json", country_file("Bosque"))
    _, out = quiet(sim.play_round, path)
    check("  fixed, it no longer shows", "Won't go through" not in out, out[-400:])

    sim, folder = phase5_game("dryunion")
    quiet(sim.form_monetary_union, "Bosque", "Llano", name="U")
    drop(folder, "Bosque.json", country_file("Bosque", phase=5, money={"money_supply_growth": 5}))
    drop(folder, "Llano.json", country_file("Llano", phase=5, money={"money_supply_growth": 2}))
    drop(folder, "Sabine.json", country_file("Sabine", phase=5))
    drop(folder, "Trinity.json", country_file("Trinity", phase=5))
    path = os.path.join(folder, "round02.xlsx")
    _, out = quiet(sim.play_round, path)
    check("  union members who disagree are caught at the build, not at the play",
          "THIS ROUND CAN'T PLAY YET" in out and "union members must be identical" in out, out[-500:])


def test_late_file_stays_in_the_inbox():
    print("\n[17] a file that lands while the round plays stays out of that round")
    sim, folder = game("late")
    all_four(folder)
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)
    real = sim.run_round

    def landing_mid_round(**kw):
        drop(folder, "Bosque late.json", country_file("Bosque", mnc=9))
        return real(**kw)

    sim.run_round = landing_mid_round
    _, out = quiet(sim.play_round, path)
    del sim.run_round
    inbox = os.path.join(folder, "inbox")
    check("  it stays in the inbox, and the output says so",
          sim.round_num == 2 and os.listdir(inbox) == ["Bosque late.json"]
          and "did not count: Bosque late.json" in out, out[-300:])
    check("  the files that were played moved as usual",
          len(os.listdir(os.path.join(folder, "round02"))) == 4)


def test_bindings_through_the_inbox():
    print("\n[18] WTO bindings come in on the round form")
    sim, folder = game("bind")
    quiet(sim.award_reserve_currency)
    quiet(sim.upgrade_to_phase5)
    quiet(sim.upgrade_to_phase6)
    quiet(sim.upgrade_to_phase7)
    for c in KEEP:
        extra = {}
        if c == "Bosque":
            extra = {"institutions": {"join_wto": True, "bound_tariffs": {"cloth": 10, "wine": 0}}}
        drop(folder, f"{c}.json", country_file(c, phase=7, **extra))
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)
    inst = sim.load_round(path)["institutional_decisions"]
    check("  the loader passes them on, as shares; a 0% binding is a binding",
          inst["Bosque"]["bound_tariffs"] == {"cloth": 0.1, "wine": 0.0}, str(inst))
    check("  nobody else's are invented",
          "bound_tariffs" not in inst.get("Llano", {}), str(inst.get("Llano")))
    quiet(sim.play_round, path)
    check("  the round applies them", sim.countries["Bosque"]["bound_tariffs"] == {"cloth": 0.1, "wine": 0.0})
    drop(folder, "Bosque 2.json", country_file("Bosque", phase=7, tariffs={"Llano": {"cloth": 20}}))
    for c in KEEP[1:]:
        drop(folder, f"{c} 2.json", country_file(c, phase=7))
    path = os.path.join(folder, "round03.xlsx")
    quiet(sim.play_round, path)
    fin = pd.read_excel(path, sheet_name="finance").set_index("country")
    check("  the next workbook keeps them", abs(fin.loc["Bosque", "bind_cloth"] - 0.1) < 1e-12)
    r = quiet(sim.play_round, path)[0]
    check("  ...and a tariff above one is a defection",
          sim.round_num == 3 and r["defected"] == ["Bosque"], str(r["defected"]))


def test_board_shows_side_payments():
    print("\n[19] the board shows side payments, made or failed")
    sim, folder = game("board")
    for c in KEEP:
        drop(folder, f"{c}.json", country_file(c))
    drop(folder, "Bosque.json", country_file("Bosque") | {"side_payments": [
        {"qty": 2, "good": "cloth", "to": "Sabine"}, {"qty": 50, "good": "machinery", "to": "Sabine"}]})
    path = os.path.join(folder, "round02.xlsx")
    quiet(sim.play_round, path)
    quiet(sim.play_round, path)
    board = classroom.scoreboard_html(sim)
    check("  both lines are on the board",
          "Side payments" in board and "Bosque -&gt; Sabine" not in board
          and "Bosque -> Sabine: 2 cloth" in board and "Bosque cannot pay 50 machinery" in board,
          board[-600:])
    check("  a round without any shows no heading",
          "Side payments" not in classroom.scoreboard_html(sim, round_num=1))


def main():
    for t in [test_newest_wins_and_junk_is_set_aside, test_stale_files_wait,
              test_trades_need_both_sides, test_build_and_play,
              test_absent_teams_repeat_last_round,
              test_unusable_files_are_never_half_applied, test_paper_rounds_unchanged,
              test_money_debt_institutions_fields, test_report_writes_nothing,
              test_real_round8_through_the_inbox, test_page_data,
              test_pass_through_listed_first, test_file_times_survive_excel,
              test_reserve_challenge_through_the_inbox,
              test_bad_files_are_reported_not_fatal,
              test_dry_run_flags_what_wont_go_through,
              test_late_file_stays_in_the_inbox, test_bindings_through_the_inbox,
              test_board_shows_side_payments]:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAIL.append((t.__name__, "exception"))
    print(f"\n{'=' * 60}")
    print(f"  PASSED: {len(PASS)}")
    print(f"  FAILED: {len(FAIL)}")
    if FAIL:
        for name, detail in FAIL:
            print(f"   - {name}: {detail}")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
