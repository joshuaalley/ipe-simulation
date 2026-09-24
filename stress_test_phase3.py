"""
Stress test Phase 3 (MNCs + varieties + CES utility) and Phase 4 selection.
Covers normal flow, edge cases, validation, save/restore, plot interaction.
"""
import sys, json, copy, io, contextlib, traceback, math
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    PHASE3_FIRMS, WORLD_PRICES, VARIETY_RHO, FIRM_LOCAL_SHARE,
    build_firm_roster,
)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


# Canonical Phase 2 balanced allocation we'll reuse
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

def fresh_phase3(firm_ids=None):
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    if firm_ids is None:
        firm_ids = ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]
    firms = {fid: PHASE3_FIRMS[fid] for fid in firm_ids}
    sim.upgrade_to_phase3(firms)
    return sim

def zero_firm_dec(sim):
    return {fid: {"scale": 0, "relocate_to": None, "export": False}
            for fid in sim.firms}

def full_firm_dec(sim, scale=30):
    return {fid: {"scale": scale, "relocate_to": None, "export": False}
            for fid in sim.firms}


# ────────────────────────────────────────────────────────────────────
# 1. All-zero firms: no MNC output, varieties = country-generic only
# ────────────────────────────────────────────────────────────────────
def test_zero_firms():
    print("\n[1] all firms scale 0 -> only country-generic varieties")
    sim = fresh_phase3()
    r = sim.run_round(BAL_DEC, [], firm_decisions=zero_firm_dec(sim))
    for fid, fr in r["firms"].items():
        check(f"  {fid} produced 0", fr["output"] == 0.0)
        check(f"  {fid} profit = 0", fr["profit"] == 0.0)
    # Each country should have exactly one variety per good (its generic)
    for n in PHASE2_COUNTRIES:
        for g in PHASE2_GOODS:
            v = r["results"][n]["consumption_varieties"][g]
            check(f"  {n}.{g} has 1 generic variety", len(v) == 1,
                  f"got {list(v.keys())}")


# ────────────────────────────────────────────────────────────────────
# 2. CES variety bonus: more varieties -> higher utility (same total qty)
# ────────────────────────────────────────────────────────────────────
def test_ces_variety_bonus():
    print("\n[2] CES variety bonus")
    sim = fresh_phase3()
    # Single-variety bundle
    bundle_single = {"cloth": {"a": 30}, "wine": {"b": 30}, "machinery": {"c": 30}}
    u_single = sim._utility_with_varieties(bundle_single)
    # Same total quantity split across 3 varieties
    bundle_triple = {
        "cloth":     {"a": 10, "a2": 10, "a3": 10},
        "wine":      {"b": 10, "b2": 10, "b3": 10},
        "machinery": {"c": 10, "c2": 10, "c3": 10},
    }
    u_triple = sim._utility_with_varieties(bundle_triple)
    check("  triple-variety utility > single-variety (same total)",
          u_triple > u_single, f"single={u_single:.3f}, triple={u_triple:.3f}")
    # Empty bundle = 0
    check("  empty bundle -> 0", sim._utility_with_varieties({}) == 0.0)
    # Missing good entirely -> 0
    check("  missing good -> 0",
          sim._utility_with_varieties({"cloth": {"a": 10}}) == 0.0)


# ────────────────────────────────────────────────────────────────────
# 3. Phase 3 utility >= Phase 2 utility, same goods totals (variety bonus)
# ────────────────────────────────────────────────────────────────────
def test_phase3_welfare_gte_phase2():
    print("\n[3] phase 3 welfare with variety >= phase 2 welfare (same totals)")
    # Run a Phase 2 baseline
    sim2 = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    r2 = sim2.run_round(BAL_DEC, [])
    # Same setup in Phase 3 with all firms at scale 0 (no MNC output)
    sim3 = fresh_phase3()
    r3 = sim3.run_round(BAL_DEC, [], firm_decisions=zero_firm_dec(sim3))
    # With zero MNC output, the variety bundle is country-generic only =
    # one variety per good per country; CES with single variety just equals qty.
    # Welfare should match within 1e-3.
    for n in PHASE2_COUNTRIES:
        w2 = r2["results"][n]["welfare"]
        w3 = r3["results"][n]["welfare"]
        check(f"  {n} P2 welfare {w2:.2f} == P3 welfare {w3:.2f} (no MNCs)",
              abs(w2 - w3) < 0.01, f"got delta {w3 - w2}")
    # Now run Phase 3 with MNCs producing -> welfare must jump in host countries
    r3_firms = sim3.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim3, scale=30))
    for n in PHASE2_COUNTRIES:
        w3_with = r3_firms["results"][n]["welfare"]
        # Phase 2 baseline already included country production. Phase 3 with
        # firms adds MNC output to host country totals AND variety bonus.
        # Host countries should clearly exceed P2.
        if any(sim3.firm_config[f]["default_host"] == n for f in sim3.firm_config):
            check(f"  {n} (firm host) P3+MNCs welfare > P2",
                  w3_with > r2["results"][n]["welfare"],
                  f"P2={r2['results'][n]['welfare']:.2f}, P3+MNCs={w3_with:.2f}")


# ────────────────────────────────────────────────────────────────────
# 4. Relocation: F3 Sabine -> Bosque, output=0 in transit, then produces
# ────────────────────────────────────────────────────────────────────
def test_relocation():
    print("\n[4] firm relocation persists across rounds")
    sim = fresh_phase3()
    fd = full_firm_dec(sim, scale=30)
    fd["F3"] = {"scale": 30, "relocate_to": "Bosque", "export": False}
    r1 = sim.run_round(BAL_DEC, [], firm_decisions=fd)
    check("  F3 host updated to Bosque", sim.firms["F3"]["host"] == "Bosque")
    check("  F3 output = 0 in relocation round", r1["firms"]["F3"]["output"] == 0.0)
    check("  F3 relocated flag", r1["firms"]["F3"]["relocated"])
    check("  F3 profit = 0 in relocation round",
          r1["firms"]["F3"]["profit"] == 0.0)
    # Next round, F3 should produce in Bosque
    fd2 = full_firm_dec(sim, scale=30)
    r2 = sim.run_round(BAL_DEC, [], firm_decisions=fd2)
    check("  F3 produces in Bosque next round", r2["firms"]["F3"]["output"] > 0)
    check("  F3 host still Bosque after producing", sim.firms["F3"]["host"] == "Bosque")
    # Cloth-C variety should now be in Bosque
    check("  Cloth-C variety in Bosque",
          "Cloth-C" in r2["results"]["Bosque"]["consumption_varieties"]["cloth"])


# ────────────────────────────────────────────────────────────────────
# 5. Firm validation
# ────────────────────────────────────────────────────────────────────
def test_firm_validation():
    print("\n[5] firm-decision validation")
    sim = fresh_phase3()
    # Missing firm
    fd = full_firm_dec(sim)
    del fd["F1"]
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd)
        check("  missing firm rejected", False, "no error")
    except ValueError:
        check("  missing firm rejected", True)
    check("  round_num unchanged after firm rejection", sim.round_num == 0)
    # Scale over max
    fd = full_firm_dec(sim)
    fd["F1"] = {"scale": 999, "relocate_to": None, "export": False}
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd)
        check("  scale > max rejected", False, "no error")
    except ValueError:
        check("  scale > max rejected", True)
    # Negative scale
    fd = full_firm_dec(sim)
    fd["F1"] = {"scale": -5, "relocate_to": None, "export": False}
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd)
        check("  negative scale rejected", False, "no error")
    except ValueError:
        check("  negative scale rejected", True)
    # Bad relocate_to
    fd = full_firm_dec(sim)
    fd["F1"] = {"scale": 30, "relocate_to": "Atlantis", "export": False}
    try:
        sim.run_round(BAL_DEC, [], firm_decisions=fd)
        check("  bad relocate_to rejected", False, "no error")
    except ValueError:
        check("  bad relocate_to rejected", True)


# ────────────────────────────────────────────────────────────────────
# 6. Variety flow through trade
# ────────────────────────────────────────────────────────────────────
def test_variety_trade_flow():
    print("\n[6] varieties flow proportionally through trades")
    sim = fresh_phase3()
    # Run a setup round so MNCs produce
    sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    # Bosque hosts F1 (Cloth-A); Llano hosts F2 (Cloth-B) and F4 (Wine-A)
    # Trade: Bosque exports cloth to Llano (20), Llano exports wine (10).
    # (Sized to what Bosque has now that hosts keep only the local share of
    # a firm's output.)
    trades = [("Bosque", "Llano", "cloth", 20, "wine", 10)]
    r = sim.run_round(BAL_DEC, trades, firm_decisions=full_firm_dec(sim, scale=30))
    llano_cloth = r["results"]["Llano"]["consumption_varieties"]["cloth"]
    bosque_wine = r["results"]["Bosque"]["consumption_varieties"]["wine"]
    check("  Llano received Cloth-A from Bosque",
          "Cloth-A" in llano_cloth, f"got {list(llano_cloth.keys())}")
    check("  Llano received Bosque generic cloth too",
          "cloth-Bosque" in llano_cloth, f"got {list(llano_cloth.keys())}")
    check("  Bosque received Wine-A from Llano",
          "Wine-A" in bosque_wine, f"got {list(bosque_wine.keys())}")
    # Mass conservation (zero-tariff): sum of varieties = scalar consumption
    for n in ["Bosque", "Llano"]:
        for g in PHASE2_GOODS:
            scalar = r["results"][n]["consumption"][g]
            variety_total = sum(r["results"][n]["consumption_varieties"][g].values())
            check(f"  mass conserved: {n}.{g}",
                  abs(scalar - variety_total) < 0.01,
                  f"scalar={scalar:.3f}, varieties_total={variety_total:.3f}")


# ────────────────────────────────────────────────────────────────────
# 7. Tariff destroys variety quantity proportionally
# ────────────────────────────────────────────────────────────────────
def test_tariff_with_varieties():
    print("\n[7] tariff destroys variety qty proportionally")
    sim = fresh_phase3()
    dec_with_tariff = {k: dict(v) for k, v in BAL_DEC.items()}
    dec_with_tariff["Llano"] = dict(dec_with_tariff["Llano"])
    dec_with_tariff["Llano"]["tariffs"] = {"Bosque": {"cloth": 0.5}}
    sim.run_round(dec_with_tariff, [], firm_decisions=full_firm_dec(sim, scale=30))
    trades = [("Bosque", "Llano", "cloth", 20, "wine", 10)]
    r = sim.run_round(dec_with_tariff, trades,
                      firm_decisions=full_firm_dec(sim, scale=30))
    # 50% tariff: Llano should receive 20 * 0.5 = 10 cloth (scalar)
    # Variety totals on Llano's side should also reflect tariff destruction
    log = "\n".join(r["trade_log"])
    check("  trade log shows 50% tariff", "50%" in log, f"log: {log!r}")
    # Tariff losses recorded
    losses = r["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  tariff loss = 10 cloth (50% of 20)",
          abs(losses - 10.0) < 0.01, f"got {losses}")


# ────────────────────────────────────────────────────────────────────
# 8. Cumulative profit accumulates correctly
# ────────────────────────────────────────────────────────────────────
def test_cumulative_profit():
    print("\n[8] cumulative profit accumulates across rounds")
    sim = fresh_phase3()
    # F1 (HIGH=1.3, cloth, unit_cost=0.6) at scale 30: revenue=39, cost=18, profit=21
    for _ in range(4):
        sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    check("  F1 cumulative profit = 4 * 21 = 84",
          abs(sim.firms["F1"]["cumulative_profit"] - 84.0) < 0.01,
          f"got {sim.firms['F1']['cumulative_profit']}")
    # F3 (LOW=0.7, cloth, unit_cost=0.6) at scale 30: revenue=21, cost=18, profit=3
    check("  F3 cumulative profit = 4 * 3 = 12",
          abs(sim.firms["F3"]["cumulative_profit"] - 12.0) < 0.01,
          f"got {sim.firms['F3']['cumulative_profit']}")


# ────────────────────────────────────────────────────────────────────
# 9. Phase 4 selection: HIGH firms profit from exports, LOW firms lose
# ────────────────────────────────────────────────────────────────────
def test_phase4_selection():
    print("\n[9] Phase 4 fixed export cost gates selection (Melitz)")
    sim = fresh_phase3()
    # First settle one Phase 3 round to populate
    sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    sim.phase = 4
    fd_export = {fid: {"scale": 30, "relocate_to": None, "export": True}
                 for fid in sim.firms}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd_export)
    # Last round had no tariffs, so exporters earn the full 25% premium.
    # F1 HIGH cloth: 39 units, rev = 39 * 1.25 = 48.75, opcost 18, fixed 8
    #   -> 22.75, vs 21 staying home: exporting adds 1.75
    check("  F1 (HIGH) profit with export: 22.75 (home 21)",
          abs(r["firms"]["F1"]["profit"] - 22.75) < 1e-9,
          f"got {r['firms']['F1']['profit']}")
    # F3 LOW cloth: 21 units, rev = 26.25, opcost 18, fixed 8
    #   -> 0.25, vs 3 staying home: exporting costs it 2.75
    check("  F3 (LOW) profit with export: 0.25 (home 3) -- worse off",
          abs(r["firms"]["F3"]["profit"] - 0.25) < 1e-9,
          f"got {r['firms']['F3']['profit']}")
    # F7 HIGH machinery (price 1.5, cost 1.0, fixed 12): 39 units,
    # rev = 39 * 1.5 * 1.25 = 73.125 -> 73.125 - 30 - 12 = 31.125, vs 28.5
    check("  F7 (HIGH machinery) profit with export: 31.125 (home 28.5)",
          abs(r["firms"]["F7"]["profit"] - 31.125) < 1e-9,
          f"got {r['firms']['F7']['profit']}")


# ────────────────────────────────────────────────────────────────────
# 10. Save/restore round-trip preserves firm state
# ────────────────────────────────────────────────────────────────────
def test_save_restore():
    print("\n[10] save/restore preserves firms + history + cumulative profit")
    sim = fresh_phase3()
    sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    fd_rel = full_firm_dec(sim, scale=30)
    fd_rel["F3"] = {"scale": 30, "relocate_to": "Bosque", "export": False}
    sim.run_round(BAL_DEC, [], firm_decisions=fd_rel)
    # Snapshot
    state = sim.get_state()
    # JSON round-trip
    encoded = json.dumps(state)
    state_back = json.loads(encoded)
    sim2 = IPESimulation.from_state(state_back)
    check("  phase preserved", sim2.phase == 3)
    check("  round_num preserved", sim2.round_num == 2)
    check("  firms count preserved", len(sim2.firms) == 10)
    check("  F1 cum profit preserved",
          abs(sim2.firms["F1"]["cumulative_profit"] - 42.0) < 0.01,
          f"got {sim2.firms['F1']['cumulative_profit']}")
    check("  F3 host preserved (Bosque after relocation)",
          sim2.firms["F3"]["host"] == "Bosque")
    check("  firm_config preserved",
          sim2.firm_config["F1"]["productivity"] == 1.3)
    check("  world_prices preserved",
          sim2.world_prices["machinery"] == 1.5)
    check("  variety_rho preserved", sim2.variety_rho == VARIETY_RHO)
    # Can continue running rounds
    sim2.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim2, scale=30))
    check("  restored sim can run more rounds", sim2.round_num == 3)
    # Cumulative profit advanced
    check("  cum profit advances on restored sim",
          sim2.firms["F1"]["cumulative_profit"] > 42.0)


# ────────────────────────────────────────────────────────────────────
# 11. Backward compat: from_state works on a pre-Phase-3 save
# ────────────────────────────────────────────────────────────────────
def test_save_restore_backcompat():
    print("\n[11] from_state handles old (pre-firms) save")
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    sim.run_round(BAL_DEC, [])
    old_state = {
        "countries": sim.countries,
        "goods": sim.goods,
        "phase": sim.phase,
        "round_num": sim.round_num,
        "history": sim.history,
    }
    sim_back = IPESimulation.from_state(old_state)
    check("  pre-Phase 3 save loads", sim_back.phase == 2)
    check("  firms default to empty", sim_back.firms == {})
    check("  variety_rho falls back to default",
          sim_back.variety_rho == VARIETY_RHO)


# ────────────────────────────────────────────────────────────────────
# 12. plot_welfare across 3 phases (no crash)
# ────────────────────────────────────────────────────────────────────
def test_plot_three_phases():
    print("\n[12] plot_welfare with all three phases")
    from engine import PHASE1_COUNTRIES, PHASE1_GOODS
    sim = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)
    p1_dec = {n: {"production": {"cloth": sim.countries[n]["labor"]//2,
                                 "wine":  sim.countries[n]["labor"] - sim.countries[n]["labor"]//2}}
              for n in PHASE1_COUNTRIES}
    sim.run_round(p1_dec, [])
    sim.upgrade_to_phase2(PHASE2_COUNTRIES, PHASE2_GOODS)
    sim.run_round(BAL_DEC, [])
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)
    sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    try:
        sim.plot_welfare()
        check("  plot_welfare 3-phase split works", True)
    except Exception as e:
        check("  plot_welfare 3-phase split works", False, str(e))
    try:
        sim.plot_production()
        sim.plot_gains_from_trade()
        check("  other plots still work", True)
    except Exception as e:
        check("  other plots still work", False, str(e))


# ────────────────────────────────────────────────────────────────────
# 13. Multiple firms per host: Trinity has F7, F10 (both machinery)
# ────────────────────────────────────────────────────────────────────
def test_multi_firm_host():
    print("\n[13] multiple firms hosted in same country")
    sim = fresh_phase3()
    r = sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    # Trinity hosts F7 (Mach-A) and F10 (Mach-D), both machinery
    trinity_mach = r["results"]["Trinity"]["consumption_varieties"]["machinery"]
    check("  Trinity has both Mach-A and Mach-D",
          "Mach-A" in trinity_mach and "Mach-D" in trinity_mach,
          f"got {list(trinity_mach.keys())}")
    check("  Trinity also has its own machinery-generic variety",
          "machinery-Trinity" in trinity_mach,
          f"got {list(trinity_mach.keys())}")


# ────────────────────────────────────────────────────────────────────
# 14. Zero scale: firm produces nothing, no impact on country
# ────────────────────────────────────────────────────────────────────
def test_zero_scale():
    print("\n[14] zero-scale firm has no impact")
    sim = fresh_phase3()
    # Phase 2 reference welfare
    sim_p2 = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    r2 = sim_p2.run_round(BAL_DEC, [])
    # Phase 3 with all firms at scale 0
    r3 = sim.run_round(BAL_DEC, [], firm_decisions=zero_firm_dec(sim))
    # Bosque hosts F1 (cloth). With F1 at scale 0, Bosque cloth = country only
    p2_cloth = r2["results"]["Bosque"]["consumption"]["cloth"]
    p3_cloth = r3["results"]["Bosque"]["consumption"]["cloth"]
    check("  Bosque cloth equal P2 vs P3-zero",
          abs(p2_cloth - p3_cloth) < 0.01,
          f"P2={p2_cloth}, P3={p3_cloth}")


# ────────────────────────────────────────────────────────────────────
# 15. Max scale: firm produces at cap, scale clamped
# ────────────────────────────────────────────────────────────────────
def test_max_scale():
    print("\n[15] scale at cap")
    sim = fresh_phase3()
    fd = full_firm_dec(sim, scale=40)  # F1 max_scale = 40
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd)
    # F1: scale 40 * productivity 1.3 = 52 output
    check("  F1 at max_scale produces 52",
          abs(r["firms"]["F1"]["output"] - 52.0) < 0.01,
          f"got {r['firms']['F1']['output']}")


# ────────────────────────────────────────────────────────────────────
# 16. Empty firm_decisions defaults to all-zero (Phase 3 backward compat)
# ────────────────────────────────────────────────────────────────────
def test_default_firm_decisions():
    print("\n[16] None firm_decisions defaults to all-zero")
    sim = fresh_phase3()
    r = sim.run_round(BAL_DEC, [], firm_decisions=None)
    for fid in sim.firms:
        check(f"  {fid} defaulted to 0 output",
              r["firms"][fid]["output"] == 0.0)


# ────────────────────────────────────────────────────────────────────
# 17. Self-trade guard still works in Phase 3
# ────────────────────────────────────────────────────────────────────
def test_self_trade_phase3():
    print("\n[17] self-trade guard works with varieties")
    sim = fresh_phase3()
    sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    trades = [("Bosque", "Bosque", "cloth", 10, "wine", 5)]
    r = sim.run_round(BAL_DEC, trades, firm_decisions=full_firm_dec(sim, scale=30))
    log = "\n".join(r["trade_log"])
    check("  self-trade logged as FAILED",
          "FAILED" in log and "self-trade" in log, f"log: {log!r}")


# ────────────────────────────────────────────────────────────────────
# 18. Firm count flexibility: run with just 5 firms
# ────────────────────────────────────────────────────────────────────
def test_subset_firms():
    print("\n[18] subset of firms (5 instead of 10) works")
    sim = fresh_phase3(firm_ids=["F1","F4","F7","F8","F10"])
    check("  loaded 5 firms", len(sim.firms) == 5)
    r = sim.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(sim, scale=30))
    check("  Phase 3 round with subset runs", "firms" in r)
    check("  all 5 firms in result", len(r["firms"]) == 5)


# ────────────────────────────────────────────────────────────────────
# Reduced country sets (smaller class): host validation + roster builder
# ────────────────────────────────────────────────────────────────────
def _small_sim(keep):
    countries = {k: PHASE2_COUNTRIES[k] for k in keep}
    sim = IPESimulation(countries, PHASE2_GOODS, phase=2)
    dec = {n: {"production": {
        "labor":   {g: c["labor"] / 3 for g in PHASE2_GOODS},
        "capital": {g: c["capital"] / 3 for g in PHASE2_GOODS}}}
        for n, c in countries.items()}
    sim.run_round(dec, [])
    return sim, dec


def test_off_map_firm_hosts_rejected():
    print("\n[off-map firm hosts are rejected at upgrade, not mid-round]")
    keep = ["Sabine", "Bosque", "Llano", "Trinity"]
    sim, _ = _small_sim(keep)
    try:
        sim.upgrade_to_phase3(PHASE3_FIRMS)
        check("  full roster on 4 countries raises", False,
              "no error raised")
    except ValueError as e:
        msg = str(e)
        check("  full roster on 4 countries raises", True)
        check("  message names the off-map firms",
              "F6" in msg and "F8" in msg and "F9" in msg, msg[:90])
        check("  message names the dropped hosts",
              "Brazos" in msg and "Pecos" in msg, msg[:90])
        check("  message points at build_firm_roster",
              "build_firm_roster" in msg, msg[:90])
    # a roster confined to surviving countries is accepted
    sim2, _ = _small_sim(keep)
    ok = {f: c for f, c in PHASE3_FIRMS.items() if c["default_host"] in keep}
    sim2.upgrade_to_phase3(ok)
    check("  on-map subset still accepted", sim2.phase == 3)


def test_build_firm_roster():
    print("\n[build_firm_roster rehomes and trims with balance]")
    keep = ["Sabine", "Bosque", "Llano", "Trinity"]
    roster = build_firm_roster(keep, n_firms=11, verbose=False)
    check("  two firms per country is the cap (11 asked -> 8 built)",
          len(roster) == 8, str(len(roster)))
    per_host = {h: sum(1 for c in roster.values() if c["default_host"] == h)
                for h in keep}
    check("  ...and every country hosts exactly two", set(per_host.values()) == {2},
          str(per_host))
    bigger = build_firm_roster(keep, n_firms=11, verbose=False, max_per_host=3)
    check("  max_per_host lifts the cap", len(bigger) == 11
          and max(sum(1 for c in bigger.values() if c["default_host"] == h)
                  for h in keep) <= 3, str(len(bigger)))
    check("  n_firms under the cap is still honoured",
          len(build_firm_roster(keep, n_firms=5, verbose=False)) == 5)
    check("  every host is in play",
          all(c["default_host"] in keep for c in roster.values()),
          str({f: c["default_host"] for f, c in roster.items()}))
    check("  keeps the HIGH/LOW spread for Melitz",
          any(c["productivity"] >= 1.2 for c in roster.values())
          and any(c["productivity"] <= 0.8 for c in roster.values()))
    counts = {h: sum(1 for c in roster.values() if c["default_host"] == h)
              for h in keep}
    check("  no host is starved or swamped",
          max(counts.values()) - min(counts.values()) <= 2, str(counts))

    # the built roster actually drives a Phase 3 round
    sim, dec = _small_sim(keep)
    sim.upgrade_to_phase3(roster)
    fd = {f: {"scale": 10, "relocate_to": None, "export": False}
          for f in sim.firms}
    sim.run_round(dec, [], firm_decisions=fd)
    check("  built roster runs a Phase 3 round", sim.round_num == 2)

    # defaults and guards
    full = build_firm_roster(list(PHASE2_COUNTRIES), verbose=False)
    check("  n_firms=None fills every host to the cap",
          len(full) == min(len(PHASE3_FIRMS), 2 * len(PHASE2_COUNTRIES)), str(len(full)))
    try:
        build_firm_roster(keep, n_firms=99, verbose=False)
        check("  over-large n_firms raises", False, "no error")
    except ValueError:
        check("  over-large n_firms raises", True)
    try:
        build_firm_roster([], verbose=False)
        check("  empty country list raises", False, "no error")
    except ValueError:
        check("  empty country list raises", True)


# ────────────────────────────────────────────────────────────────────
# 21. The MNC tax: a country decision from Phase 3
# ────────────────────────────────────────────────────────────────────
def play_taxed(sim, taxes=None, scale=40, fd=None):
    dec = copy.deepcopy(BAL_DEC)
    for c, t in (taxes or {}).items():
        dec[c]["mnc_tax"] = t
    return sim.run_round(dec, [], firm_decisions=fd or full_firm_dec(sim, scale=scale))


def test_mnc_tax_decision():
    print("\n[21] the MNC tax: the host's choice, the host's revenue, the owner's cost")
    base = fresh_phase3()
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    r0, r1 = play_taxed(a), play_taxed(b, {"Trinity": 0.10})
    # Trinity hosts F7 (HIGH machinery) and F10 (MED machinery). At scale 40:
    # revenue 52 x 1.5 = 78 and 40 x 1.5 = 60, so 10% collects 7.8 + 6.0.
    t = r1["results"]["Trinity"]["mnc_tax"]
    check("  Trinity collects 10% of its firms' revenue: 13.80",
          abs(t["collected"] - 13.8) < 1e-9 and t["rate"] == 0.10, str(t))
    cons = r1["results"]["Trinity"]["consumption"]
    cap = sum(cons[g] * WORLD_PRICES[g] for g in PHASE2_GOODS)
    check("  the host keeps it: welfare x (1 + T/C), exactly",
          abs(r1["results"]["Trinity"]["welfare"]
              / r0["results"]["Trinity"]["welfare"] - (1 + 13.8 / cap)) < 1e-12)
    check("  the owners pay it: F7 38 -> 30.2, F10 20 -> 14",
          abs(r1["firms"]["F7"]["profit"] - 30.2) < 1e-9
          and abs(r1["firms"]["F10"]["profit"] - 14.0) < 1e-9)
    check("  nobody else's welfare moves",
          all(r1["results"][c]["welfare"] == r0["results"][c]["welfare"]
              for c in PHASE2_COUNTRIES if c != "Trinity"))
    check("  it is not a gain from trade (the metric ignores it)",
          r1["results"]["Trinity"]["gains_from_trade_pct"]
          == r0["results"]["Trinity"]["gains_from_trade_pct"])
    r2 = play_taxed(b)
    check("  the rate stands until changed",
          r2["results"]["Trinity"]["mnc_tax"]["rate"] == 0.10)
    r3 = play_taxed(b, {"Trinity": 0.0})
    check("  ...and changing it to 0 ends it",
          r3["results"]["Trinity"]["mnc_tax"]["collected"] == 0.0)
    fd = full_firm_dec(b, scale=40)
    fd["F7"] = {"scale": 0, "relocate_to": "Llano", "export": False}
    r4 = play_taxed(b, {"Trinity": 0.20}, fd=fd)
    check("  a firm moving out this round pays nothing",
          r4["firms"]["F7"]["mnc_tax"] == 0.0)

    # limits, and only once there are firms
    for bad, label in ((0.55, "above 50%"), (-0.05, "below 0 (subsidies off)"),
                       ("ten", "not a number")):
        s = copy.deepcopy(base)
        try:
            play_taxed(s, {"Bosque": bad})
            check(f"  {label} rejected", False, "no error")
        except ValueError as e:
            check(f"  {label} rejected", "MNC tax" in str(e) and s.round_num == 0,
                  str(e)[:120])
    p2 = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    dec = copy.deepcopy(BAL_DEC)
    dec["Bosque"]["mnc_tax"] = 0.10
    try:
        p2.run_round(dec, [])
        check("  Phase 2: no firms, no MNC tax", False, "no error")
    except ValueError as e:
        check("  Phase 2: no firms, no MNC tax", "Phase 3" in str(e), str(e)[:120])

    # a populist government's minimum holds; a higher choice wins
    pop = copy.deepcopy(base)
    pop.countries["Bosque"]["mnc_tax_rate"] = 0.15
    r5 = play_taxed(pop, {"Bosque": 0.05})
    check("  a populist 15% minimum beats a 5% choice",
          r5["results"]["Bosque"]["mnc_tax"]["rate"] == 0.15
          and abs(r5["firms"]["F1"]["mnc_tax"] - 0.15 * 52) < 1e-9)
    r6 = play_taxed(pop, {"Bosque": 0.25})
    check("  ...and a 25% choice beats the minimum",
          r6["results"]["Bosque"]["mnc_tax"]["rate"] == 0.25)

    # the standing rate survives a save
    s2 = IPESimulation.from_state(json.loads(json.dumps(b.get_state())))
    check("  a snapshot keeps the standing rate",
          s2.mnc_tax_in_force("Trinity") == 0.20)


# ────────────────────────────────────────────────────────────────────
# 22. Owners: a firm can't move to its owner's home (no re-shoring)
# ────────────────────────────────────────────────────────────────────
def test_owners_no_reshoring():
    print("\n[22] owners and the no-re-shoring rule")
    sim = fresh_phase3()
    with contextlib.redirect_stdout(io.StringIO()):
        sim.set_firm_owners({"F1": "Llano", "F7": ["Bosque", "Sabine"]})
    check("  owners recorded (a pair keeps both countries)",
          sim.firms["F1"]["owners"] == ["Llano"]
          and sim.firms["F7"]["owners"] == ["Bosque", "Sabine"])
    for fid, home in (("F1", "Llano"), ("F7", "Sabine")):
        fd = full_firm_dec(sim)
        fd[fid] = {"scale": 0, "relocate_to": home, "export": False}
        try:
            sim.run_round(BAL_DEC, [], firm_decisions=fd)
            check(f"  {fid} can't move home to {home}", False, "no error")
        except ValueError as e:
            check(f"  {fid} can't move home to {home}",
                  "no re-shoring" in str(e) and sim.round_num == 0, str(e)[:120])
    fd = full_firm_dec(sim)
    fd["F1"] = {"scale": 0, "relocate_to": "Pecos", "export": False}
    fd["F2"] = {"scale": 0, "relocate_to": "Bosque", "export": False}
    sim.run_round(BAL_DEC, [], firm_decisions=fd)
    check("  anywhere else is fine, and firms without owners move freely",
          sim.firms["F1"]["host"] == "Pecos" and sim.firms["F2"]["host"] == "Bosque")
    for owners, label in (({"F3": "Sabine"}, "an owner from the host country"),
                          ({"F99": "Llano"}, "an unknown firm"),
                          ({"F4": "Atlantis"}, "an unknown country")):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.set_firm_owners(owners)
            check(f"  {label} is refused", False, "no error")
        except ValueError:
            check(f"  {label} is refused", True)
    fd = full_firm_dec(sim, scale=30)
    fd["F4"] = {"scale": 30, "relocate_to": sim.firms["F4"]["host"], "export": False}
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd)
    check("  'relocate to' your current host just means stay",
          r["firms"]["F4"]["output"] > 0 and not r["firms"]["F4"]["relocated"])
    back = IPESimulation.from_state(json.loads(json.dumps(sim.get_state())))
    fd = full_firm_dec(back)
    fd["F1"] = {"scale": 0, "relocate_to": "Llano", "export": False}
    try:
        back.run_round(BAL_DEC, [], firm_decisions=fd)
        check("  owners survive a snapshot, rule and all", False, "no error")
    except ValueError:
        check("  owners survive a snapshot, rule and all", True)


# ────────────────────────────────────────────────────────────────────
# 23. Calibration: hosts get the local share; hosting matters but can't
#     swallow an economy
# ────────────────────────────────────────────────────────────────────
def test_firm_calibration():
    print("\n[23] firm calibration: local share, bounded hosting gains")
    a, b = fresh_phase3(), fresh_phase3()
    r0 = a.run_round(BAL_DEC, [], firm_decisions=zero_firm_dec(a))
    r1 = b.run_round(BAL_DEC, [], firm_decisions=full_firm_dec(b, scale=40))
    # F1 (HIGH cloth, Bosque) makes 52 units; Bosque's economy gets its local share
    added = (r1["results"]["Bosque"]["production"]["cloth"]
             - r0["results"]["Bosque"]["production"]["cloth"])
    check(f"  host production gains the local share ({FIRM_LOCAL_SHARE:.0%} of 52)",
          abs(added - 52 * FIRM_LOCAL_SHARE) < 1e-9, f"added {added:.3f}")
    check("  ...while the owner's revenue counts every unit (profit 28)",
          abs(r1["firms"]["F1"]["profit"] - 28.0) < 1e-9
          and abs(r1["firms"]["F1"]["revenue"] - 52.0) < 1e-9)
    gain = {c: r1["results"][c]["welfare"] / r0["results"][c]["welfare"]
            for c in PHASE2_COUNTRIES}
    hosts = {b.firms[f]["host"] for f in b.firms}
    check("  hosting at full scale never doubles a country",
          max(gain.values()) < 2.0, str({c: round(g, 2) for c, g in gain.items()}))
    check("  ...but every host gains at least 5%",
          all(gain[c] > 1.05 for c in hosts),
          str({c: round(gain[c], 2) for c in hosts}))

    # a save from before Phase 3 may carry an older love-of-variety setting
    old = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    old.variety_rho = 0.6
    with contextlib.redirect_stdout(io.StringIO()):
        old.upgrade_to_phase3({f: PHASE3_FIRMS[f] for f in ("F1", "F7")})
    check("  the Phase 3 upgrade applies today's VARIETY_RHO",
          old.variety_rho == VARIETY_RHO)


def main():
    tests = [
        test_zero_firms,
        test_ces_variety_bonus,
        test_phase3_welfare_gte_phase2,
        test_relocation,
        test_firm_validation,
        test_variety_trade_flow,
        test_tariff_with_varieties,
        test_cumulative_profit,
        test_phase4_selection,
        test_save_restore,
        test_save_restore_backcompat,
        test_plot_three_phases,
        test_multi_firm_host,
        test_zero_scale,
        test_max_scale,
        test_default_firm_decisions,
        test_self_trade_phase3,
        test_subset_firms,
        test_off_map_firm_hosts_rejected,
        test_build_firm_roster,
        test_mnc_tax_decision,
        test_owners_no_reshoring,
        test_firm_calibration,
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
