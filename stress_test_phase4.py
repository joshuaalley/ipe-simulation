"""
Stress test Phase 4 mechanics:
- Productivity surge (structural shock; neutral framing)
- Populist backlash (tariff floor + MNC tax)
- MNC tax ledger (separate from welfare)
- Tariff floor enforcement (max with declared)
- Reserve currency awarding (cumulative welfare; Phase-4 tiebreaker)
- Firm rankings printer
- Save/restore with Phase 4 state
"""
import sys, json, traceback
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    PHASE3_FIRMS,
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

def fresh_phase4():
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    # Get one Phase 3 round in for history
    sim.run_round(BAL_DEC, [], firm_decisions={
        fid: {"scale": 30, "relocate_to": None, "export": False}
        for fid in sim.firms
    })
    sim.phase = 4
    return sim

def fd(sim, scale=30, export=False):
    return {fid: {"scale": scale, "relocate_to": None, "export": export}
            for fid in sim.firms}


# ────────────────────────────────────────────────────────────────────
# 1. Productivity surge: multiply TFP, propagates to country H-O output
# ────────────────────────────────────────────────────────────────────
def test_productivity_surge():
    print("\n[1] productivity surge multiplies TFP")
    sim = fresh_phase4()
    pre = sim.countries["Pecos"]["tech"]["machinery"]["tfp"]
    sim.inject_productivity_surge("Pecos", "machinery", 2.5,
                                  description="Tech leap in machinery")
    post = sim.countries["Pecos"]["tech"]["machinery"]["tfp"]
    check("  TFP multiplied 2.5x", abs(post - pre * 2.5) < 1e-6,
          f"pre={pre}, post={post}")
    # Run a round and confirm Pecos's machinery output is much higher
    sim_baseline = fresh_phase4()
    r_base = sim_baseline.run_round(BAL_DEC, [], firm_decisions=fd(sim_baseline))
    r_surge = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    pecos_base = r_base["results"]["Pecos"]["production"]["machinery"]
    pecos_surge = r_surge["results"]["Pecos"]["production"]["machinery"]
    # Country H-O output scales 2.5x; MNC output stays flat, so total ratio
    # is ~2x. Just check we got a substantial jump.
    check("  Pecos machinery output substantially up after surge",
          pecos_surge > pecos_base * 1.5,
          f"base={pecos_base:.1f}, surge={pecos_surge:.1f}")

    # Bad industry name should raise
    try:
        sim.inject_productivity_surge("Pecos", "lentils", 2.0)
        check("  unknown industry raises", False, "no error")
    except (KeyError, ValueError):
        check("  unknown industry raises", True)


# ────────────────────────────────────────────────────────────────────
# 2. Populist backlash sets tariff_floor + mnc_tax_rate
# ────────────────────────────────────────────────────────────────────
def test_populist_backlash():
    print("\n[2] populist backlash")
    sim = fresh_phase4()
    sim.inject_populist_backlash("Bosque", tariff_floor=0.25, mnc_tax_rate=0.15)
    check("  Bosque tariff_floor = 0.25",
          sim.countries["Bosque"]["tariff_floor"] == 0.25)
    check("  Bosque mnc_tax_rate = 0.15",
          sim.countries["Bosque"]["mnc_tax_rate"] == 0.15)
    # Reverse via inject_shock
    sim.inject_shock("Reform government", {
        "Bosque": {"tariff_floor": 0.0, "mnc_tax_rate": 0.0}
    })
    check("  tariff_floor reset to 0",
          sim.countries["Bosque"]["tariff_floor"] == 0.0)
    check("  mnc_tax_rate reset to 0",
          sim.countries["Bosque"]["mnc_tax_rate"] == 0.0)


# ────────────────────────────────────────────────────────────────────
# 3. Tariff floor: applies as MAX of declared and floor
# ────────────────────────────────────────────────────────────────────
def test_tariff_floor():
    print("\n[3] tariff floor enforced")
    sim = fresh_phase4()
    sim.inject_populist_backlash("Llano", tariff_floor=0.30, mnc_tax_rate=0.0)
    # Trade with NO declared tariff into Llano; floor should still apply
    trades = [("Bosque", "Llano", "cloth", 40, "wine", 20)]
    r = sim.run_round(BAL_DEC, trades, firm_decisions=fd(sim))
    # Tariff loss = 40 * 0.30 = 12
    loss = r["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  tariff floor 30% applies despite no declared tariff",
          abs(loss - 12.0) < 0.01, f"got {loss}")

    # Now declare a HIGHER tariff than the floor; the higher one should win
    sim2 = fresh_phase4()
    sim2.inject_populist_backlash("Llano", tariff_floor=0.10, mnc_tax_rate=0.0)
    dec = {k: dict(v) for k, v in BAL_DEC.items()}
    dec["Llano"] = dict(dec["Llano"])
    dec["Llano"]["tariffs"] = {"Bosque": {"cloth": 0.50}}
    r2 = sim2.run_round(dec, trades, firm_decisions=fd(sim2))
    loss2 = r2["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  declared 50% > floor 10% -> 50% applies",
          abs(loss2 - 20.0) < 0.01, f"got {loss2}")


# ────────────────────────────────────────────────────────────────────
# 4. MNC tax: deducted from firm profit, credited to ledger (NOT welfare)
# ────────────────────────────────────────────────────────────────────
def test_mnc_tax_ledger():
    print("\n[4] MNC tax ledger separation")
    sim = fresh_phase4()
    # No tax baseline: F1 in Bosque at scale 30, HIGH=1.3, unit_cost=0.6
    # revenue = 30 * 1.3 * 1.0 = 39; cost = 18; profit = 21
    r_pre = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    pre_profit = r_pre["firms"]["F1"]["profit"]
    pre_welfare = r_pre["results"]["Bosque"]["welfare"]
    check("  F1 profit without tax = 21",
          abs(pre_profit - 21.0) < 0.01, f"got {pre_profit}")

    # Apply 20% MNC tax to Bosque
    sim.inject_populist_backlash("Bosque", tariff_floor=0.0, mnc_tax_rate=0.20)
    r_post = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    # MNC tax on F1 = 39 * 0.20 = 7.80; profit = 21 - 7.80 = 13.20
    check("  F1 MNC tax = 7.80",
          abs(r_post["firms"]["F1"]["mnc_tax"] - 7.80) < 0.01,
          f"got {r_post['firms']['F1']['mnc_tax']}")
    check("  F1 profit after tax = 13.20",
          abs(r_post["firms"]["F1"]["profit"] - 13.20) < 0.01,
          f"got {r_post['firms']['F1']['profit']}")
    # Bosque welfare must be UNCHANGED relative to pre-tax (same allocations)
    post_welfare = r_post["results"]["Bosque"]["welfare"]
    check("  Bosque welfare UNCHANGED by MNC tax (ledger separate)",
          abs(post_welfare - pre_welfare) < 0.01,
          f"pre={pre_welfare}, post={post_welfare}")
    # Ledger captured the tax
    check("  Bosque tax ledger this round = 7.80",
          abs(r_post["mnc_tax_this_round"]["Bosque"] - 7.80) < 0.01,
          f"got {r_post['mnc_tax_this_round']['Bosque']}")
    check("  cumulative ledger matches",
          abs(sim.mnc_tax_revenue["Bosque"] - 7.80) < 0.01,
          f"got {sim.mnc_tax_revenue.get('Bosque')}")

    # Run another round; cumulative should grow
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    check("  cumulative ledger doubles after 2 rounds",
          abs(sim.mnc_tax_revenue["Bosque"] - 15.60) < 0.01,
          f"got {sim.mnc_tax_revenue['Bosque']}")


# ────────────────────────────────────────────────────────────────────
# 5. MNC tax does NOT fire in Phase 3 even if rate is set
# ────────────────────────────────────────────────────────────────────
def test_mnc_tax_phase3_inactive():
    print("\n[5] MNC tax inactive in Phase 3")
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)  # phase = 3
    # Set tax rate but stay in Phase 3
    sim.countries["Bosque"]["mnc_tax_rate"] = 0.20
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    check("  F1 mnc_tax = 0 in Phase 3", r["firms"]["F1"]["mnc_tax"] == 0.0,
          f"got {r['firms']['F1']['mnc_tax']}")
    check("  no mnc_tax_this_round in Phase 3 result",
          "mnc_tax_this_round" not in r)


# ────────────────────────────────────────────────────────────────────
# 6. award_reserve_currency picks top cumulative welfare
# ────────────────────────────────────────────────────────────────────
def test_award_reserve_currency():
    print("\n[6] reserve currency awarded to top cumulative welfare")
    sim = fresh_phase4()
    # Already has 2 rounds of history (one Phase 3, one Phase 4)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    ranking = sim.award_reserve_currency()
    check("  ranking has all 6 countries", len(ranking) == 6)
    check("  reserve_currency_holder set",
          sim.reserve_currency_holder == ranking[0])
    # Check that the top country actually has the highest cumulative welfare
    cum = {n: sum(h["results"][n]["welfare"] for h in sim.history)
           for n in sim.countries}
    top_by_cum = max(cum, key=cum.get)
    check(f"  top in ranking ({ranking[0]}) is top in cumulative ({top_by_cum})",
          ranking[0] == top_by_cum)

    # Empty-history edge case
    sim_empty = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=4)
    r = sim_empty.award_reserve_currency()
    check("  empty history returns []", r == [])
    check("  empty history leaves holder None",
          sim_empty.reserve_currency_holder is None)


# ────────────────────────────────────────────────────────────────────
# 7. print_firm_rankings prints in descending profit order
# ────────────────────────────────────────────────────────────────────
def test_print_firm_rankings():
    print("\n[7] firm rankings descending by cumulative profit")
    sim = fresh_phase4()
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    # Capture stdout
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.print_firm_rankings()
    out = buf.getvalue()
    check("  ranking includes all firms", all(fid in out for fid in sim.firms))
    # Top firm should appear before bottom firm in the text
    # F7 Mach-A HIGH machinery should be #1; F9 Mach-C LOW machinery near bottom
    f7_pos = out.find("F7  ")
    f9_pos = out.find("F9  ")
    check("  F7 (HIGH machinery, top expected) appears above F9 (LOW)",
          f7_pos < f9_pos, f"F7@{f7_pos}, F9@{f9_pos}")


# ────────────────────────────────────────────────────────────────────
# 8. Combined: populist backlash + Melitz export selection
# ────────────────────────────────────────────────────────────────────
def test_populist_plus_selection():
    print("\n[8] populist backlash combined with export selection")
    sim = fresh_phase4()
    sim.inject_populist_backlash("Sabine", tariff_floor=0.30, mnc_tax_rate=0.15)
    # Now F3 (LOW prod cloth in Sabine) exporting:
    # output = 21, revenue = 21
    # op_cost = 18, fixed_export = 8, mnc_tax = 21 * 0.15 = 3.15
    # profit = 21 - 18 - 8 - 3.15 = -8.15
    fde = fd(sim, export=True)
    r = sim.run_round(BAL_DEC, [], firm_decisions=fde)
    check("  F3 (LOW + populist host + export) profit = -8.15",
          abs(r["firms"]["F3"]["profit"] - (-8.15)) < 0.01,
          f"got {r['firms']['F3']['profit']}")
    check("  Sabine ledger collected MNC tax",
          sim.mnc_tax_revenue["Sabine"] > 0)


# ────────────────────────────────────────────────────────────────────
# 9. save/restore preserves Phase 4 state
# ────────────────────────────────────────────────────────────────────
def test_save_restore_phase4():
    print("\n[9] save/restore preserves Phase 4 fields")
    sim = fresh_phase4()
    sim.inject_populist_backlash("Bosque", tariff_floor=0.25, mnc_tax_rate=0.20)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    ranking = sim.award_reserve_currency()
    # Round-trip via JSON
    state = json.loads(json.dumps(sim.get_state()))
    sim2 = IPESimulation.from_state(state)
    check("  phase preserved", sim2.phase == 4)
    check("  reserve_currency_holder preserved",
          sim2.reserve_currency_holder == sim.reserve_currency_holder)
    check("  mnc_tax_revenue preserved",
          sim2.mnc_tax_revenue == sim.mnc_tax_revenue)
    check("  Bosque tariff_floor preserved",
          sim2.countries["Bosque"]["tariff_floor"] == 0.25)
    check("  Bosque mnc_tax_rate preserved",
          sim2.countries["Bosque"]["mnc_tax_rate"] == 0.20)
    # Can continue rounds
    sim2.run_round(BAL_DEC, [], firm_decisions=fd(sim2))
    check("  continues running after restore",
          sim2.round_num == sim.round_num + 1)


# ────────────────────────────────────────────────────────────────────
# 10. print_results in Phase 4 shows tariff floor and MNC tax
# ────────────────────────────────────────────────────────────────────
def test_print_results_phase4():
    print("\n[10] print_results extensions for Phase 4")
    sim = fresh_phase4()
    sim.inject_populist_backlash("Bosque", tariff_floor=0.25, mnc_tax_rate=0.20)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.print_results()
    out = buf.getvalue()
    check("  output mentions tariff floor", "TARIFF FLOORS" in out)
    check("  output mentions MNC tax ledger", "MNC TAX LEDGER" in out)
    check("  output reminds tax is separate ledger",
          "does NOT enter welfare" in out)


# ────────────────────────────────────────────────────────────────────
# 11. inject_shock idempotent with previously-missing keys (smoke)
# ────────────────────────────────────────────────────────────────────
def test_inject_shock_missing_key_smoke():
    print("\n[11] inject_shock handles brand-new fields")
    sim = fresh_phase4()
    # tariff_floor doesn't exist on Pecos by default
    sim.inject_shock("New regulation", {"Pecos": {"tariff_floor": 0.15}})
    check("  new field set", sim.countries["Pecos"]["tariff_floor"] == 0.15)
    # Now nested new field on a fresh key
    sim.inject_shock("Sectoral policy", {
        "Pecos": {"sectoral_subsidy": {"cloth": 0.10}}
    })
    check("  nested new field set",
          sim.countries["Pecos"]["sectoral_subsidy"]["cloth"] == 0.10)


def main():
    tests = [
        test_productivity_surge,
        test_populist_backlash,
        test_tariff_floor,
        test_mnc_tax_ledger,
        test_mnc_tax_phase3_inactive,
        test_award_reserve_currency,
        test_print_firm_rankings,
        test_populist_plus_selection,
        test_save_restore_phase4,
        test_print_results_phase4,
        test_inject_shock_missing_key_smoke,
    ]
    for t in tests:
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
