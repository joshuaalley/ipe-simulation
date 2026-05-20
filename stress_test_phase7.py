"""
Stress test Phase 7 institutions & power:
- upgrade requires Phase 5 monetary state
- WTO membership + bound tariffs + rules dividend (friction cut)
- defection (tariff above binding) -> loses dividend, no welfare penalty
- hegemon provision: cost to hegemon, global friction cut; withhold: global friction up
- goods-based side payments
- hegemonic challenge via coalition cumulative welfare; reserve currency moves
- global crisis mitigated by hegemon provision + WTO breadth
- save/restore; dashboard renders
"""
import sys, json, traceback
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS, PHASE3_FIRMS,
    WTO_DIVIDEND, HEGEMON_PROVISION_COST,
    HEGEMON_PROVISION_BENEFIT, HEGEMON_WITHHOLD_PENALTY,
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
                               "capital": {"cloth": 50, "wine": 50, "machinery": 50}}, "tariffs": {}},
    "Bosque":  {"production": {"labor":   {"cloth": 30, "wine": 15, "machinery": 15},
                               "capital": {"cloth": 10, "wine": 8,  "machinery": 7}}, "tariffs": {}},
    "Llano":  {"production": {"labor":   {"cloth": 25, "wine": 50, "machinery": 25},
                               "capital": {"cloth": 20, "wine": 35, "machinery": 25}}, "tariffs": {}},
    "Trinity":  {"production": {"labor":   {"cloth": 30, "wine": 30, "machinery": 60},
                               "capital": {"cloth": 40, "wine": 40, "machinery": 120}}, "tariffs": {}},
    "Pecos":  {"production": {"labor":   {"cloth": 10, "wine": 15, "machinery": 25},
                               "capital": {"cloth": 20, "wine": 30, "machinery": 70}}, "tariffs": {}},
    "Sabine": {"production": {"labor":   {"cloth": 50, "wine": 35, "machinery": 15},
                               "capital": {"cloth": 15, "wine": 12, "machinery": 8}}, "tariffs": {}},
}

def safe_md():
    return {c: {"fx_regime": "managed", "capital_controls": False,
                "independent_monetary": True, "money_supply_growth": 0.0}
            for c in PHASE2_COUNTRIES}

def fd(sim, scale=30):
    return {fid: {"scale": scale, "relocate_to": None, "export": False}
            for fid in sim.firms}

def fresh_phase7():
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.phase = 4
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.award_reserve_currency()
    sim.upgrade_to_phase5()
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim), monetary_decisions=safe_md())
    sim.upgrade_to_phase6(); sim.upgrade_to_phase7()
    return sim

def run6(sim, trades=None, inst=None, sidep=None):
    return sim.run_round(BAL_DEC, trades or [], firm_decisions=fd(sim),
                         monetary_decisions=safe_md(),
                         institutional_decisions=inst, side_payments=sidep)


# ───────────────────────────────────────────────────────────────
def test_upgrade_guard():
    print("\n[1] upgrade guard + init")
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.award_reserve_currency()
    try:
        sim.upgrade_to_phase6(); sim.upgrade_to_phase7()  # phase 5 not done
        check("  requires phase 5 first", False, "no error")
    except ValueError:
        check("  requires phase 5 first", True)
    sim.upgrade_to_phase5()
    sim.upgrade_to_phase6(); sim.upgrade_to_phase7()
    check("  phase == 7", sim.phase == 7)
    check("  hegemon = reserve holder", sim.hegemon == sim.reserve_currency_holder)
    check("  hegemon provides by default", sim.hegemon_provides is True)


# ───────────────────────────────────────────────────────────────
def test_wto_dividend():
    print("\n[2] WTO dividend cuts friction between members")
    sim = fresh_phase7()
    rc = sim.hegemon
    non_rc = [c for c in sim.countries if c != rc]
    a, b = non_rc[0], non_rc[1]
    # Baseline (not members), hegemon provides: friction = base + (-benefit)
    sim.hegemon_provides = True
    d_nonmember = sim._institutional_friction_delta(a, b, set())
    check("  non-members get only hegemon benefit",
          abs(d_nonmember - (-HEGEMON_PROVISION_BENEFIT)) < 1e-9, f"got {d_nonmember}")
    # Both members, no defection
    sim.join_wto(a, b)
    d_member = sim._institutional_friction_delta(a, b, set())
    check("  members get hegemon benefit + WTO dividend",
          abs(d_member - (-HEGEMON_PROVISION_BENEFIT - WTO_DIVIDEND)) < 1e-9,
          f"got {d_member}")
    # If one defected, no dividend
    d_def = sim._institutional_friction_delta(a, b, {a})
    check("  defector loses dividend",
          abs(d_def - (-HEGEMON_PROVISION_BENEFIT)) < 1e-9, f"got {d_def}")


# ───────────────────────────────────────────────────────────────
def test_defection_flagged():
    print("\n[3] binding defection detected")
    sim = fresh_phase7()
    sim.join_wto("Bosque")
    sim.bind_tariff("Bosque", "cloth", 0.10)
    # Bosque applies a 30% tariff on cloth -> defection
    dec = {k: {kk: (dict(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items()}
           for k, v in BAL_DEC.items()}
    dec["Bosque"]["tariffs"] = {"Trinity": {"cloth": 0.30}}
    r = sim.run_round(dec, [], firm_decisions=fd(sim),
                      monetary_decisions=safe_md())
    check("  Bosque flagged as defector",
          "Bosque" in r["defected"], f"got {r['defected']}")
    check("  defection counter incremented",
          sim.countries["Bosque"]["defections"] == 1)
    # Within binding -> no defection
    dec["Bosque"]["tariffs"] = {"Trinity": {"cloth": 0.05}}
    r2 = sim.run_round(dec, [], firm_decisions=fd(sim), monetary_decisions=safe_md())
    check("  within binding -> no defection", "Bosque" not in r2["defected"])


# ───────────────────────────────────────────────────────────────
def test_hegemon_provision():
    print("\n[4] hegemon provision cost + global friction")
    sim = fresh_phase7()
    heg = sim.hegemon
    # Provide
    sim.set_hegemon_provision(True)
    r = run6(sim)
    inst = r["results"][heg]["institutions"]
    check("  hegemon pays provision cost", inst["hegemon_cost"] > 0)
    # Friction delta when providing is negative for a generic pair
    non = [c for c in sim.countries if c != heg]
    d_prov = sim._institutional_friction_delta(non[0], non[1], set())
    # Withhold
    sim.set_hegemon_provision(False)
    d_with = sim._institutional_friction_delta(non[0], non[1], set())
    check("  withholding raises friction vs providing", d_with > d_prov,
          f"prov={d_prov}, with={d_with}")
    r2 = run6(sim)
    check("  withholding hegemon pays no cost",
          r2["results"][heg]["institutions"]["hegemon_cost"] == 0.0)


# ───────────────────────────────────────────────────────────────
def test_side_payments():
    print("\n[5] goods-based side payments")
    sim = fresh_phase7()
    # Trinity ships Bosque 20 machinery
    r = run6(sim, sidep=[("Trinity", "Bosque", "machinery", 20)])
    check("  side payment logged",
          any("side payment" in line for line in r["side_payment_log"]),
          f"got {r['side_payment_log']}")
    # Bosque's machinery consumption rose
    check("  recipient machinery up",
          r["results"]["Bosque"]["consumption"]["machinery"] > 0)
    # Oversized payment fails gracefully
    r2 = run6(sim, sidep=[("Bosque", "Trinity", "machinery", 99999)])
    check("  oversize side payment fails",
          any("FAILED" in line for line in r2["side_payment_log"]))


# ───────────────────────────────────────────────────────────────
def test_hegemonic_challenge():
    print("\n[6] hegemonic challenge via coalition weight")
    sim = fresh_phase7()
    heg = sim.hegemon
    cum = sim._cumulative_welfare()
    # Build a coalition of everyone except hegemon -> must outweigh hegemon alone
    challenger = max((c for c in sim.countries if c != heg), key=lambda c: cum[c])
    backers = [c for c in sim.countries if c not in (heg, challenger)]
    ok = sim.challenge_hegemon(challenger, backers=backers)
    check("  all-but-hegemon coalition wins", ok)
    check("  hegemon transferred", sim.hegemon == challenger)
    check("  reserve currency moved", sim.reserve_currency_holder == challenger)

    # A lone weak challenger fails
    sim2 = fresh_phase7()
    cum2 = sim2._cumulative_welfare()
    weak = min(sim2.countries, key=lambda c: cum2[c])
    if weak == sim2.hegemon:
        weak = sorted(sim2.countries, key=lambda c: cum2[c])[1]
    ok2 = sim2.challenge_hegemon(weak, backers=[])
    check("  lone weak challenger fails", not ok2)
    check("  hegemon unchanged after failed challenge",
          sim2.hegemon == sim2.reserve_currency_holder)


# ───────────────────────────────────────────────────────────────
def test_global_crisis():
    print("\n[6.5] global crisis mitigation")
    # No mitigation: withhold + no WTO
    sim = fresh_phase7()
    sim.set_hegemon_provision(False)
    sim.inject_global_crisis(0.20)
    r = run6(sim)
    cf_bad = r["global_crisis_factor"]
    check("  withhold+no-WTO: full-ish pain (factor ~0.80)",
          abs(cf_bad - 0.80) < 0.01, f"got {cf_bad}")

    # Mitigated: provide + all WTO
    sim2 = fresh_phase7()
    sim2.set_hegemon_provision(True)
    sim2.join_wto(*sim2.countries)
    sim2.inject_global_crisis(0.20)
    r2 = run6(sim2)
    cf_good = r2["global_crisis_factor"]
    # 0.20 * 0.5 (provide) * (1 - 0.5*1.0 wto) = 0.05 -> factor 0.95
    check("  provide+full-WTO softens (factor ~0.95)",
          abs(cf_good - 0.95) < 0.01, f"got {cf_good}")
    check("  crisis is one-shot (cleared after round)",
          sim2._pending_global_crisis is None)


# ───────────────────────────────────────────────────────────────
def test_save_restore():
    print("\n[7] save/restore Phase 6")
    sim = fresh_phase7()
    sim.join_wto("Bosque", "Trinity")
    sim.bind_tariff("Bosque", "cloth", 0.10)
    sim.set_hegemon_provision(False)
    run6(sim)
    state = json.loads(json.dumps(sim.get_state()))
    sim2 = IPESimulation.from_state(state)
    check("  phase preserved", sim2.phase == 7)
    check("  hegemon preserved", sim2.hegemon == sim.hegemon)
    check("  provision flag preserved", sim2.hegemon_provides == sim.hegemon_provides)
    check("  WTO membership preserved", sim2.countries["Bosque"]["wto_member"])
    check("  bindings preserved",
          sim2.countries["Bosque"]["bound_tariffs"].get("cloth") == 0.10)
    run6(sim2)
    check("  continues after restore", sim2.round_num == sim.round_num + 1)


# ───────────────────────────────────────────────────────────────
def test_display():
    print("\n[8] dashboards render")
    sim = fresh_phase7()
    sim.join_wto("Bosque", "Trinity")
    sim.bind_tariff("Bosque", "cloth", 0.10)
    run6(sim)
    try:
        sim.print_results()
        check("  print_results works in Phase 6", True)
    except Exception as e:
        check("  print_results works in Phase 6", False, str(e))
    try:
        sim.print_institutions_dashboard()
        check("  institutions dashboard works", True)
    except Exception as e:
        check("  institutions dashboard works", False, str(e))


def main():
    for t in [test_upgrade_guard, test_wto_dividend, test_defection_flagged,
              test_hegemon_provision, test_side_payments,
              test_hegemonic_challenge, test_global_crisis,
              test_save_restore, test_display]:
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
