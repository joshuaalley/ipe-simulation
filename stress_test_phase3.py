"""
Stress test Phase 3 (MNCs + varieties + CES utility) and Phase 4 selection.
Covers normal flow, edge cases, validation, save/restore, plot interaction.
"""
import sys, json, traceback, math
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    PHASE3_FIRMS, WORLD_PRICES, VARIETY_RHO,
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
    # Trade: Bosque exports cloth to Llano (40), Llano exports wine (20)
    trades = [("Bosque", "Llano", "cloth", 40, "wine", 20)]
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
    trades = [("Bosque", "Llano", "cloth", 40, "wine", 20)]
    r = sim.run_round(dec_with_tariff, trades,
                      firm_decisions=full_firm_dec(sim, scale=30))
    # 50% tariff: Llano should receive 40 * 0.5 = 20 cloth (scalar)
    # Variety totals on Llano's side should also reflect tariff destruction
    log = "\n".join(r["trade_log"])
    check("  trade log shows 50% tariff", "50%" in log, f"log: {log!r}")
    # Tariff losses recorded
    losses = r["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  tariff loss = 20 cloth (50% of 40)",
          abs(losses - 20.0) < 0.01, f"got {losses}")


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
    # F1 HIGH cloth (cost=0.6, fixed=8): rev=39, opcost=18, fixed=8 -> profit=13
    check("  F1 (HIGH) profit with export: 13",
          abs(r["firms"]["F1"]["profit"] - 13.0) < 0.01,
          f"got {r['firms']['F1']['profit']}")
    # F3 LOW cloth: rev=21, opcost=18, fixed=8 -> profit=-5 (loses money!)
    check("  F3 (LOW) profit with export: -5",
          abs(r["firms"]["F3"]["profit"] - (-5.0)) < 0.01,
          f"got {r['firms']['F3']['profit']}")
    # F7 HIGH machinery (price=1.5, cost=1.0, fixed=12): rev=58.5, opcost=30, fixed=12
    # output = 30 * 1.3 = 39, rev = 39 * 1.5 = 58.5
    # profit = 58.5 - 30 - 12 = 16.5
    check("  F7 (HIGH machinery) profit with export: 16.5",
          abs(r["firms"]["F7"]["profit"] - 16.5) < 0.01,
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
    check("  honours n_firms", len(roster) == 11, str(len(roster)))
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
    check("  n_firms=None keeps the whole base roster",
          len(full) == len(PHASE3_FIRMS))
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
