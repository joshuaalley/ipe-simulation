"""
Stress test Phase 5 monetary mechanics:
- upgrade requires reserve currency holder
- monetary decision validation (regimes, discrete money growth, union consistency)
- graduated trilemma stress: 2-of-3 safe, warning at stress=1, crisis at stress=2
- money-supply decay
- FX friction (reserve = 0, union = 0, baseline, warning bump)
- host-currency firm profits (real = nominal * host depreciation)
- manual speculative attack, monetary shock, capital flight
- monetary unions: shared state, formation, dissolution
- save/restore; dashboard + plot render
"""
import sys, json, traceback
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS, PHASE3_FIRMS,
    PHASE5_MONEY_GROWTH_CHOICES,
    WARNING_DEVALUATION, CRISIS_DEVALUATION, CRISIS_WELFARE_HIT,
    BASE_FX_FRICTION, WARNING_FRICTION_BUMP,
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
    """Default safe monetary decision (managed float, open, independent, 0%)."""
    return {c: {"fx_regime": "managed", "capital_controls": False,
                "independent_monetary": True, "money_supply_growth": 0.0}
            for c in PHASE2_COUNTRIES}

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

    # Safe: peg + capital controls + independent (only 2 of the 3 risky knobs)
    md = base_md()
    md["Bosque"] = {"fx_regime": "peg", "capital_controls": True,
                    "independent_monetary": True, "money_supply_growth": 0.0}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=md)
    check("  2-of-3 -> no stress", r["results"]["Bosque"]["monetary"]["stress"] == 0)
    check("  2-of-3 -> no warning", not r["results"]["Bosque"]["monetary"]["warning"])

    # Overreach: peg + open capital + independent monetary
    over = base_md()
    over["Bosque"] = {"fx_regime": "peg", "capital_controls": False,
                      "independent_monetary": True, "money_supply_growth": 0.0}
    r1 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    m1 = r1["results"]["Bosque"]["monetary"]
    check("  overreach round 1 -> warning", m1["warning"] and not m1["crisis"])
    check("  warning devalued ~10%", abs(m1["depreciation_factor"] - WARNING_DEVALUATION) < 1e-6,
          f"got {m1['depreciation_factor']}")
    check("  stress now 1", m1["stress"] == 1)

    # Second consecutive overreach -> full crisis
    r2 = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=over)
    m2 = r2["results"]["Bosque"]["monetary"]
    check("  overreach round 2 -> crisis", m2["crisis"])
    # dep factor: 0.90 (warning) then *0.70 (crisis) = 0.63
    check("  crisis compounds devaluation (~0.63)",
          abs(m2["depreciation_factor"] - WARNING_DEVALUATION * CRISIS_DEVALUATION) < 1e-6,
          f"got {m2['depreciation_factor']}")
    check("  stress reset after crisis", m2["stress"] == 0)
    check("  crisis welfare loss recorded", m2["crisis_welfare_loss"] > 0)

    # Back off: 2-of-3 -> stress stays 0, no new crisis
    safe = base_md()
    safe["Bosque"] = {"fx_regime": "float", "capital_controls": False,
                      "independent_monetary": True, "money_supply_growth": 0.0}
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
                   "independent_monetary": True, "money_supply_growth": 0.10}
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
                    "independent_monetary": True, "money_supply_growth": 0.10}
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

    # Consistent decisions accepted; shared stress
    md = base_md()
    for c in ("Bosque", "Pecos"):
        md[c] = {"fx_regime": "peg", "capital_controls": False,
                 "independent_monetary": True, "money_supply_growth": 0.0}
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
                 "independent_monetary": True, "money_supply_growth": 0.05}
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
                      "independent_monetary": True, "money_supply_growth": 0.0}
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


def main():
    for t in [test_upgrade, test_validation, test_trilemma_graduated,
              test_money_decay, test_fx_friction, test_host_currency_profits,
              test_manual_triggers, test_monetary_union, test_save_restore,
              test_display]:
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
