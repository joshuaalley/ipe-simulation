"""
Stress test Phase 5 monetary mechanics:
- upgrade requires reserve currency holder
- monetary decision validation (regimes, discrete money growth, union consistency)
- graduated trilemma stress: peg + open + PRINTING overreaches; any 2-of-3 safe;
  warning at stress=1, crisis at stress=2
- money-supply decay
- FX friction (reserve = 0, union = 0, peg-peg = 0, baseline, warning bump)
- host-currency firm profits (real = nominal * host depreciation)
- manual speculative attack, monetary shock, capital flight
- monetary unions: shared state, formation, dissolution
- save/restore; dashboard + plot render
- what each choice buys: printing's stimulus, a weak currency's import cost,
  controls' cost to foreign firms; 'managed' = float; the open-peg exposure;
  no attack without a clear target; capital flight works in Phase 5
"""
import copy, contextlib, io, sys, json, traceback
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS, PHASE3_FIRMS,
    PHASE5_MONEY_GROWTH_CHOICES,
    WARNING_DEVALUATION, CRISIS_DEVALUATION, CRISIS_WELFARE_HIT,
    BASE_FX_FRICTION, WARNING_FRICTION_BUMP,
    STIMULUS_PER_POINT, WEAK_FX_IMPORT_COST, WEAK_FX_WELFARE_DRAG,
    CONTROLS_FIRM_CUT,
)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")

BAL_DEC = {
    "Brazos":  {"production": {"labor":   {"cloth": 50, "wine": 50, "machinery": 50},
                               "capital": {"cloth": 50, "wine": 50, "machinery": 50}}},
    "Bosque":  {"production": {"labor":   {"cloth": 30, "wine": 15, "machinery": 15},
                               "capital": {"cloth": 10, "wine": 8,  "machinery": 7}}},
    "Llano":  {"production": {"labor":   {"cloth": 25, "wine": 50, "machinery": 25},
                               "capital": {"cloth": 20, "wine": 35, "machinery": 25}}},
    "Trinity":  {"production": {"labor":   {"cloth": 30, "wine": 30, "machinery": 60},
                               "capital": {"cloth": 40, "wine": 40, "machinery": 120}}},
    "Pecos":  {"production": {"labor":   {"cloth": 10, "wine": 15, "machinery": 25},
                               "capital": {"cloth": 20, "wine": 30, "machinery": 70}}},
    "Sabine": {"production": {"labor":   {"cloth": 50, "wine": 35, "machinery": 15},
                               "capital": {"cloth": 15, "wine": 12, "machinery": 8}}},
}

def base_md():
    """Default safe monetary decision (float, open, 0% = follow the anchor)."""
    return {c: {"fx_regime": "float", "capital_controls": False,
                "money_supply_growth": 0.0}
            for c in PHASE2_COUNTRIES}


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)

def fd(sim, scale=30):
    return {fid: {"scale": scale, "relocate_to": None, "export": False}
            for fid in sim.firms}

def fresh_phase5():
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))   # Phase 3 round (history)
    sim.phase = 4
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))   # Phase 4 round
    sim.award_reserve_currency()
    sim.upgrade_to_phase5()
    return sim


# ───────────────────────────────────────────────────────────────
# 1. Upgrade guard + initialization
# ───────────────────────────────────────────────────────────────
def test_upgrade():
    print("\n[1] upgrade_to_phase5 guard + init")
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    try:
        sim.upgrade_to_phase5()  # no reserve currency yet
        check("  upgrade requires reserve currency", False, "no error")
    except ValueError:
        check("  upgrade requires reserve currency", True)

    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.award_reserve_currency()
    sim.upgrade_to_phase5()
    check("  phase == 5", sim.phase == 5)
    check("  countries have currency", all(
        "currency" in sim.countries[c] for c in sim.countries))
    check("  default dep factor 1.0", all(
        sim.countries[c]["depreciation_factor"] == 1.0 for c in sim.countries))
    check("  reserve holder set", sim.reserve_currency_holder is not None)


# ───────────────────────────────────────────────────────────────
# 2. Validation
# ───────────────────────────────────────────────────────────────
def test_validation():
    print("\n[2] monetary decision validation")
    sim = fresh_phase5()
    md = base_md()
    md["Brazos"]["money_supply_growth"] = 0.07  # not in discrete set
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
        check("  bad money growth rejected", False, "no error")
    except ValueError:
        check("  bad money growth rejected", True)

    md = base_md()
    md["Brazos"]["fx_regime"] = "crawling"  # invalid
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
        check("  bad regime rejected", False, "no error")
    except ValueError:
        check("  bad regime rejected", True)
    check("  round_num not advanced on failure", sim.round_num == 2)


# ───────────────────────────────────────────────────────────────
# 3. Trilemma: 2-of-3 safe; graduated stress to crisis
# ───────────────────────────────────────────────────────────────
def test_trilemma_graduated():
    print("\n[3] graduated trilemma stress")
    sim = fresh_phase5()

    # Safe: peg + capital controls + printing (controls give up open capital)
    md = base_md()
    md["Bosque"] = {"fx_regime": "peg", "capital_controls": True,
                    "money_supply_growth": 0.05}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    check("  peg + controls + print -> no stress",
          r["results"]["Bosque"]["monetary"]["stress"] == 0)
    check("  peg + controls + print -> no warning",
          not r["results"]["Bosque"]["monetary"]["warning"])

    # Safe: peg + open capital + 0% (following the anchor gives up autonomy)
    md = base_md()
    md["Llano"] = {"fx_regime": "peg", "capital_controls": False,
                   "money_supply_growth": 0.0}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    check("  peg + open + 0% (follow the anchor) -> no stress",
          r["results"]["Llano"]["monetary"]["stress"] == 0
          and not r["results"]["Llano"]["monetary"]["warning"])

    # Overreach: peg + open capital + printing
    g = 0.05
    over = base_md()
    over["Bosque"] = {"fx_regime": "peg", "capital_controls": False,
                      "money_supply_growth": g}
    dep0 = sim._mon("Bosque")["depreciation_factor"]
    r1 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    m1 = r1["results"]["Bosque"]["monetary"]
    check("  overreach round 1 -> warning", m1["warning"] and not m1["crisis"])
    want1 = dep0 * WARNING_DEVALUATION * (1 - g)
    check("  warning devalued 10% (on top of the printing)",
          abs(m1["depreciation_factor"] - want1) < 1e-6,
          f"got {m1['depreciation_factor']}, want {want1}")
    check("  stress now 1", m1["stress"] == 1)

    # Second consecutive overreach -> full crisis
    r2 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    m2 = r2["results"]["Bosque"]["monetary"]
    check("  overreach round 2 -> crisis", m2["crisis"])
    want2 = want1 * CRISIS_DEVALUATION * (1 - g)
    check("  crisis compounds devaluation",
          abs(m2["depreciation_factor"] - want2) < 1e-6,
          f"got {m2['depreciation_factor']}, want {want2}")
    check("  stress reset after crisis", m2["stress"] == 0)
    check("  crisis welfare loss recorded", m2["crisis_welfare_loss"] > 0)

    # Back off: float, stop printing -> stress stays 0, no new crisis
    safe = base_md()
    safe["Bosque"] = {"fx_regime": "float", "capital_controls": False,
                      "money_supply_growth": 0.0}
    r3 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=safe)
    m3 = r3["results"]["Bosque"]["monetary"]
    check("  backing off -> no crisis", not m3["crisis"] and not m3["warning"])


# ───────────────────────────────────────────────────────────────
# 4. Money supply decay
# ───────────────────────────────────────────────────────────────
def test_money_decay():
    print("\n[4] money-supply growth depreciates currency")
    sim = fresh_phase5()
    md = base_md()
    md["Pecos"] = {"fx_regime": "float", "capital_controls": False,
                   "money_supply_growth": 0.10}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    dep = r["results"]["Pecos"]["monetary"]["depreciation_factor"]
    check("  10% money growth -> dep factor 0.90", abs(dep - 0.90) < 1e-6, f"got {dep}")
    # Another round compounds
    r2 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    dep2 = r2["results"]["Pecos"]["monetary"]["depreciation_factor"]
    check("  compounds to ~0.81", abs(dep2 - 0.81) < 1e-6, f"got {dep2}")


# ───────────────────────────────────────────────────────────────
# 5. FX friction
# ───────────────────────────────────────────────────────────────
def test_fx_friction():
    print("\n[5] FX friction rules")
    sim = fresh_phase5()
    rc = sim.reserve_currency_holder
    # Pick two non-reserve countries for a baseline-friction trade
    non_rc = [c for c in sim.countries if c != rc]
    a, b = non_rc[0], non_rc[1]
    f_base = sim._compute_fx_friction(a, b)
    check("  cross-currency baseline friction = 2%",
          abs(f_base - BASE_FX_FRICTION) < 1e-9, f"got {f_base}")
    # Reserve involved -> 0
    f_rc = sim._compute_fx_friction(rc, a)
    check("  reserve-involved friction = 0", f_rc == 0.0, f"got {f_rc}")
    # Same union -> 0
    sim.form_monetary_union(a, b, name="TestUnion")
    f_union = sim._compute_fx_friction(a, b)
    check("  same-union friction = 0", f_union == 0.0, f"got {f_union}")
    sim.dissolve_monetary_union("TestUnion")
    # Warning bump
    sim._mon(a)["warning_active"] = True
    f_warn = sim._compute_fx_friction(a, b)
    check("  warning bump adds friction",
          abs(f_warn - (BASE_FX_FRICTION + WARNING_FRICTION_BUMP)) < 1e-9,
          f"got {f_warn}")


# ───────────────────────────────────────────────────────────────
# 6. Host-currency firm profits
# ───────────────────────────────────────────────────────────────
def test_host_currency_profits():
    print("\n[6] firm profits accrue in host currency")
    sim = fresh_phase5()
    # Depreciate Bosque (F1's host) via money printing
    md = base_md()
    md["Bosque"] = {"fx_regime": "float", "capital_controls": False,
                    "money_supply_growth": 0.10}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    f1 = r["firms"]["F1"]
    # F1 nominal profit = 21 (HIGH cloth at scale 30); host dep = 0.90
    check("  F1 nominal profit unchanged (21)", abs(f1["profit_nominal"] - 21.0) < 0.01,
          f"got {f1['profit_nominal']}")
    check("  F1 real profit = nominal * 0.90 = 18.9",
          abs(f1["profit"] - 18.9) < 0.01, f"got {f1['profit']}")
    # A firm in a stable host (Trinity, F7) keeps full value
    f7 = r["firms"]["F7"]
    check("  F7 (stable host) real == nominal",
          abs(f7["profit"] - f7["profit_nominal"]) < 0.01,
          f"nominal={f7['profit_nominal']}, real={f7['profit']}")


# ───────────────────────────────────────────────────────────────
# 7. Manual triggers
# ───────────────────────────────────────────────────────────────
def test_manual_triggers():
    print("\n[7] manual instructor triggers")
    sim = fresh_phase5()
    dep_before = sim.countries["Llano"]["depreciation_factor"]
    sim.inject_speculative_attack("Llano")
    check("  attack devalues immediately",
          abs(sim.countries["Llano"]["depreciation_factor"]
              - dep_before * CRISIS_DEVALUATION) < 1e-6)
    # Next round should record the crisis welfare hit
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=base_md())
    check("  forced attack delivers welfare hit next round",
          r["results"]["Llano"]["monetary"]["crisis"])

    # Monetary shock
    sim.inject_monetary_shock("Sabine", 0.10)
    check("  monetary shock set growth", sim._mon("Sabine")["money_supply_growth"] == 0.10)
    try:
        sim.inject_monetary_shock("Sabine", 0.07)  # invalid
        check("  invalid money growth rejected", False, "no error")
    except ValueError:
        check("  invalid money growth rejected", True)

    # Capital flight
    dep_b = sim.countries["Pecos"]["depreciation_factor"]
    sim.inject_capital_flight("Pecos", severity=0.5)
    check("  capital flight halves dep factor",
          abs(sim.countries["Pecos"]["depreciation_factor"] - dep_b * 0.5) < 1e-6)


# ───────────────────────────────────────────────────────────────
# 8. Monetary unions
# ───────────────────────────────────────────────────────────────
def test_monetary_union():
    print("\n[8] monetary unions")
    sim = fresh_phase5()
    sim.form_monetary_union("Bosque", "Pecos", name="BP")
    check("  members tagged with union",
          sim.countries["Bosque"]["union_id"] == "BP"
          and sim.countries["Pecos"]["union_id"] == "BP")
    check("  union shares one state object",
          sim._mon("Bosque") is sim._mon("Pecos"))

    # Union members must submit identical decisions
    md = base_md()
    md["Bosque"]["money_supply_growth"] = 0.05
    md["Pecos"]["money_supply_growth"] = 0.10   # mismatch
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
        check("  inconsistent union decisions rejected", False, "no error")
    except ValueError:
        check("  inconsistent union decisions rejected", True)

    # Consistent decisions accepted; a union that pegs, stays open and
    # prints shares the stress
    md = base_md()
    md["Bosque"] = {"fx_regime": "peg", "capital_controls": False,
                    "money_supply_growth": 0.05}
    md["Pecos"] = {"fx_regime": "peg", "capital_controls": False,
                   "money_supply_growth": 0.05}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    check("  union overreach warns both members",
          r["results"]["Bosque"]["monetary"]["warning"]
          and r["results"]["Pecos"]["monetary"]["warning"])
    check("  union members share stress",
          r["results"]["Bosque"]["monetary"]["stress"]
          == r["results"]["Pecos"]["monetary"]["stress"] == 1)

    # Dissolve
    sim.dissolve_monetary_union("BP")
    check("  dissolve clears union_id",
          sim.countries["Bosque"]["union_id"] is None)
    check("  dissolve preserves dep factor",
          sim.countries["Bosque"]["depreciation_factor"] < 1.0)


# ───────────────────────────────────────────────────────────────
# 9. Save / restore
# ───────────────────────────────────────────────────────────────
def test_save_restore():
    print("\n[9] save/restore Phase 5 state")
    sim = fresh_phase5()
    sim.form_monetary_union("Bosque", "Pecos", name="BP")
    md = base_md()
    for c in ("Bosque", "Pecos"):
        md[c] = {"fx_regime": "peg", "capital_controls": False,
                 "money_supply_growth": 0.05}
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    state = json.loads(json.dumps(sim.get_state()))
    sim2 = IPESimulation.from_state(state)
    check("  phase preserved", sim2.phase == 5)
    check("  union preserved", "BP" in sim2.monetary_unions)
    check("  union members preserved",
          set(sim2.monetary_unions["BP"]["members"]) == {"Bosque", "Pecos"})
    check("  monetary state preserved",
          sim2._mon("Bosque")["money_supply_growth"] == 0.05)
    # Continue running
    sim2.run_round(BAL_DEC, [], firm_decisions=fd(sim2), monetary_decisions=md)
    check("  continues after restore", sim2.round_num == sim.round_num + 1)


# ───────────────────────────────────────────────────────────────
# 10. Display + plot
# ───────────────────────────────────────────────────────────────
def test_display():
    print("\n[10] dashboard + plot")
    sim = fresh_phase5()
    over = base_md()
    over["Bosque"] = {"fx_regime": "peg", "capital_controls": False,
                      "money_supply_growth": 0.05}
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    try:
        sim.print_results()
        check("  print_results works in Phase 5", True)
    except Exception as e:
        check("  print_results works in Phase 5", False, str(e))
    try:
        sim.print_monetary_dashboard()
        check("  print_monetary_dashboard works", True)
    except Exception as e:
        check("  print_monetary_dashboard works", False, str(e))
    try:
        sim.plot_currency_health()
        check("  plot_currency_health renders", True)
    except Exception as e:
        check("  plot_currency_health renders", False, str(e))


# ───────────────────────────────────────────────────────────────
# 11-18. What each choice buys (or costs)
# ───────────────────────────────────────────────────────────────
def non_reserve(sim):
    return [c for c in PHASE2_COUNTRIES if c != sim.reserve_currency_holder]


def test_regime_alias_and_independence():
    print("\n[11] 'managed' = float; printing IS independent policy")
    sim = fresh_phase5()
    md = base_md()
    md["Bosque"] = {"fx_regime": "managed", "capital_controls": False,
                    "independent_monetary": True, "money_supply_growth": 0.0}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    m = r["results"]["Bosque"]["monetary"]
    check("  old-style 'managed' + independent accepted, stored as float",
          m["fx_regime"] == "float")
    check("  independence derived: 0% growth -> not independent",
          m["independent_monetary"] is False)
    md = base_md()
    md["Bosque"] = {"fx_regime": "float", "capital_controls": False,
                    "independent_monetary": False, "money_supply_growth": 0.05}
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
        check("  'not independent' + printing is rejected", False, "no error")
    except ValueError as e:
        check("  'not independent' + printing is rejected", "printing money" in str(e))
    quiet(sim.form_monetary_union, "Bosque", "Pecos", name="BP")
    md = base_md()
    md["Bosque"]["fx_regime"] = "managed"
    md["Pecos"]["fx_regime"] = "float"
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
        check("  union: 'managed' and 'float' count as the same decision", True)
    except ValueError as e:
        check("  union: 'managed' and 'float' count as the same decision", False, str(e))


def test_printing_stimulus():
    print("\n[12] printing buys a stimulus, scaled by the currency's strength")
    base = fresh_phase5()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    g = 0.10
    md_b = base_md()
    md_b["Bosque"]["money_supply_growth"] = g
    ra = a.run_round(BAL_DEC, [], firm_decisions=fd(a), monetary_decisions=base_md())
    rb = b.run_round(BAL_DEC, [], firm_decisions=fd(b), monetary_decisions=md_b)
    wa = ra["results"]["Bosque"]["welfare"]
    wb = rb["results"]["Bosque"]["welfare"]
    mb = rb["results"]["Bosque"]["monetary"]
    d = mb["depreciation_factor"]
    want = (1 + STIMULUS_PER_POINT * g * d) * (1 - WEAK_FX_WELFARE_DRAG * (1 - d))
    check("  welfare x (1 + stimulus x growth x FX) x (1 - drag x (1 - FX))"
          " (no trade, so no import cost)",
          abs(wb / wa - want) < 1e-9, f"ratio {wb / wa:.6f}, want {want:.6f}")
    check("  stimulus recorded",
          abs(mb["stimulus"] - wa * STIMULUS_PER_POINT * g * d) < 1e-9)
    check("  the weaker currency's drag recorded",
          abs(mb["fx_drag"] - (wa + mb["stimulus"]) * WEAK_FX_WELFARE_DRAG * (1 - d))
          < 1e-9)
    check("  not counted as a gain from trade",
          ra["results"]["Bosque"]["gains_from_trade_pct"]
          == rb["results"]["Bosque"]["gains_from_trade_pct"])
    rb2 = b.run_round(BAL_DEC, [], firm_decisions=fd(b), monetary_decisions=md_b)
    check("  printing again stimulates less (the currency is weaker)",
          rb2["results"]["Bosque"]["monetary"]["stimulus"] < mb["stimulus"])
    check("  following the anchor: no stimulus",
          ra["results"]["Bosque"]["monetary"]["stimulus"] == 0.0)


def test_weak_currency_import_cost():
    print("\n[13] a weak currency shrinks the imports you receive")
    sim = fresh_phase5()
    rc = sim.reserve_currency_holder
    a, b = non_reserve(sim)[:2]
    sim._mon(a)["depreciation_factor"] = 0.80
    trade = [(rc, a, "machinery", 5, "cloth", 5)]      # reserve party: no friction
    r = sim.run_round(BAL_DEC, trade, firm_decisions=fd(sim), monetary_decisions=base_md())
    t = r["trades_executed"][0]
    cut = WEAK_FX_IMPORT_COST * (1 - 0.80)
    check("  weak side receives less", abs(t["qty_out_received"] - 5 * (1 - cut)) < 1e-9,
          f"got {t['qty_out_received']}")
    check("  strong side receives in full", abs(t["qty_in_received"] - 5) < 1e-9,
          f"got {t['qty_in_received']}")
    check("  recorded on the trade", abs(t["weak_fx_importer"] - cut) < 1e-12
          and t["weak_fx_exporter"] == 0.0)
    quiet(sim.form_monetary_union, a, b, name="AB")
    sim._mon(a)["depreciation_factor"] = 0.80
    r = sim.run_round(BAL_DEC, [(b, a, "wine", 4, "cloth", 4)],
                      firm_decisions=fd(sim), monetary_decisions=base_md())
    t = r["trades_executed"][0]
    check("  no cost inside a monetary union (same currency)",
          t["qty_out_received"] == 4 and t["qty_in_received"] == 4,
          f"{t['qty_out_received']}, {t['qty_in_received']}")


def test_weak_currency_welfare_drag():
    print("\n[13b] a weak currency drags welfare even without trade")
    base = fresh_phase5()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    b._mon("Llano")["depreciation_factor"] = 0.70           # e.g. after a crisis
    ra = a.run_round(BAL_DEC, [], firm_decisions=fd(a), monetary_decisions=base_md())
    rb = b.run_round(BAL_DEC, [], firm_decisions=fd(b), monetary_decisions=base_md())
    ratio = rb["results"]["Llano"]["welfare"] / ra["results"]["Llano"]["welfare"]
    want = 1 - WEAK_FX_WELFARE_DRAG * 0.30
    check("  FX 0.70 -> welfare x (1 - drag x 0.30), no printing involved",
          abs(ratio - want) < 1e-9, f"ratio {ratio:.6f}, want {want:.6f}")
    check("  a currency at par pays nothing",
          ra["results"]["Llano"]["monetary"]["fx_drag"] == 0.0)


def test_controls_cost_foreign_firms():
    print("\n[14] capital controls cut foreign firms' output")
    base = fresh_phase5()
    host = base.firms["F1"]["host"]
    other = next(f for f, st in base.firms.items() if st["host"] != host)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    md = base_md()
    md[host]["capital_controls"] = True
    ra = a.run_round(BAL_DEC, [], firm_decisions=fd(a), monetary_decisions=base_md())
    rb = b.run_round(BAL_DEC, [], firm_decisions=fd(b), monetary_decisions=md)
    ratio = rb["firms"]["F1"]["output"] / ra["firms"]["F1"]["output"]
    check(f"  a firm in {host} produces {CONTROLS_FIRM_CUT:.0%} less",
          abs(ratio - (1 - CONTROLS_FIRM_CUT)) < 1e-9, f"ratio {ratio}")
    check("  firms elsewhere unaffected",
          rb["firms"][other]["output"] == ra["firms"][other]["output"])
    check("  the host's own welfare pays for it",
          rb["results"][host]["welfare"] < ra["results"][host]["welfare"])


def test_peg_to_peg_friction():
    print("\n[15] two pegged currencies trade friction-free")
    sim = fresh_phase5()
    a, b = non_reserve(sim)[:2]
    sim._mon(a)["fx_regime"] = "peg"
    sim._mon(b)["fx_regime"] = "peg"
    check("  peg-peg friction = 0", sim._compute_fx_friction(a, b) == 0.0)
    sim._mon(b)["fx_regime"] = "float"
    check("  peg-float pays the baseline",
          abs(sim._compute_fx_friction(a, b) - BASE_FX_FRICTION) < 1e-12)


def test_open_peg_exposure():
    print("\n[16] an open peg is what speculators attack")
    sim = fresh_phase5()
    a, b, c = non_reserve(sim)[:3]
    sim._mon(a).update({"fx_regime": "peg", "capital_controls": False})
    sim._mon(b).update({"fx_regime": "peg", "capital_controls": True})
    sim._mon(c).update({"fx_regime": "float", "capital_controls": False})
    parts = {n: p for n, _s, p in sim.fx_vulnerability()}
    check("  open peg scores the peg", parts[a]["peg to defend"] == 1.0)
    check("  peg behind controls does not", parts[b]["peg to defend"] == 0.0)
    check("  a float has no peg to defend", parts[c]["peg to defend"] == 0.0)


def test_no_clear_target():
    print("\n[17] no attack without a clear weakest link")
    sim = fresh_phase5()                     # everyone: float, open, 0%, FX 1.00
    before = {c: sim._mon(c)["depreciation_factor"] for c in PHASE2_COUNTRIES}
    t = quiet(sim.trigger_speculative_attack)
    check("  dead heat -> no attack", t is None)
    check("  ...and nobody devalued",
          all(sim._mon(c)["depreciation_factor"] == before[c] for c in PHASE2_COUNTRIES))
    sim._mon("Llano")["fx_regime"] = "peg"   # the one open peg
    t = quiet(sim.trigger_speculative_attack)
    check("  a clear weakest link is attacked", t == "Llano", f"got {t}")


def test_capital_flight_in_phase5():
    print("\n[18] capital flight works in Phase 5 (currency exposure)")
    sim = fresh_phase5()
    sim._mon("Sabine")["money_supply_growth"] = 0.10    # loose money
    try:
        t = quiet(sim.trigger_capital_flight, severity=0.6)
        check("  targets the most exposed currency", t == "Sabine", f"got {t}")
        check("  currency loses 40%",
              abs(sim._mon("Sabine")["depreciation_factor"] - 0.6) < 1e-9)
    except Exception as e:
        check("  trigger_capital_flight runs before Phase 6", False, repr(e))


def main():
    for t in [test_upgrade, test_validation, test_trilemma_graduated,
              test_money_decay, test_fx_friction, test_host_currency_profits,
              test_manual_triggers, test_monetary_union, test_save_restore,
              test_display, test_regime_alias_and_independence,
              test_printing_stimulus, test_weak_currency_import_cost,
              test_weak_currency_welfare_drag,
              test_controls_cost_foreign_firms, test_peg_to_peg_friction,
              test_open_peg_exposure, test_no_clear_target,
              test_capital_flight_in_phase5]:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAIL.append((t.__name__, "exception"))
    print(f"\n{'='*60}")
    print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
    if FAIL:
        for name, detail in FAIL:
            print(f"   - {name}: {detail}")
        sys.exit(1)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
