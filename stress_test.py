"""
Stress test the IPE simulation engine across normal, edge, and weird inputs.
Each TEST function prints PASS / FAIL.
Run:  python stress_test.py
"""
import sys, traceback, math
import matplotlib
matplotlib.use("Agg")  # headless

from engine import (
    IPESimulation,
    PHASE1_COUNTRIES, PHASE1_GOODS,
    PHASE2_COUNTRIES, PHASE2_GOODS,
)

PASS, FAIL = [], []

def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")

def fresh_phase1():
    return IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)

def fresh_phase2():
    return IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)


# ────────────────────────────────────────────────────────────────────
# 1. Baseline autarky balanced
# ────────────────────────────────────────────────────────────────────
def test_phase1_balanced_autarky():
    print("\n[1] phase 1 balanced autarky")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 30, "wine": 30}},
        "Llano":  {"production": {"cloth": 50, "wine": 50}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 50, "wine": 50}},
    }
    r = sim.run_round(dec, [])
    for name in PHASE1_COUNTRIES:
        w = r["results"][name]["welfare"]
        check(f"  {name} welfare > 0", w > 0, f"got {w}")
        check(f"  {name} gains_pct == 0 in autarky",
              r["results"][name]["gains_from_trade_pct"] == 0.0,
              f"got {r['results'][name]['gains_from_trade_pct']}")


# ────────────────────────────────────────────────────────────────────
# 2. Full specialization in autarky -> welfare 0 (Cobb-Douglas)
# ────────────────────────────────────────────────────────────────────
def test_phase1_full_specialization_autarky():
    print("\n[2] phase 1 FULL specialization, no trade -> welfare = 0")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 150, "wine": 0}},
        "Bosque":  {"production": {"cloth": 60,  "wine": 0}},
        "Llano":  {"production": {"cloth": 0,   "wine": 100}},
        "Trinity":  {"production": {"cloth": 120, "wine": 0}},
        "Pecos":  {"production": {"cloth": 50,  "wine": 0}},
        "Sabine": {"production": {"cloth": 0,   "wine": 100}},
    }
    r = sim.run_round(dec, [])
    for name in PHASE1_COUNTRIES:
        w = r["results"][name]["welfare"]
        check(f"  {name} welfare = 0 (missing one good)", w == 0.0, f"got {w}")
        # gains_pct should be 0 (welfare = 0 AND no_trade = 0)
        check(f"  {name} gains_pct = 0 (both zero)",
              r["results"][name]["gains_from_trade_pct"] == 0.0,
              f"got {r['results'][name]['gains_from_trade_pct']}")


# ────────────────────────────────────────────────────────────────────
# 3. Specialization + trade -> gains_pct = inf when no_trade=0 but trade>0
# ────────────────────────────────────────────────────────────────────
def test_phase1_specialization_with_trade():
    print("\n[3] phase 1 specialization + trade -> inf gains for specializers")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75,  "wine": 75}},  # balanced
        "Bosque":  {"production": {"cloth": 60,  "wine": 0}},   # full spec
        "Llano":  {"production": {"cloth": 0,   "wine": 100}}, # full spec
        "Trinity":  {"production": {"cloth": 80,  "wine": 40}},
        "Pecos":  {"production": {"cloth": 25,  "wine": 25}},
        "Sabine": {"production": {"cloth": 30,  "wine": 70}},
    }
    trades = [
        ("Bosque", "Llano", "cloth", 100, "wine", 80),
    ]
    r = sim.run_round(dec, trades)
    bosque_gain = r["results"]["Bosque"]["gains_from_trade_pct"]
    llano_gain = r["results"]["Llano"]["gains_from_trade_pct"]
    check("  Bosque gains = inf (autarky welfare was 0)",
          bosque_gain == float("inf"), f"got {bosque_gain}")
    check("  Llano gains = inf (autarky welfare was 0)",
          llano_gain == float("inf"), f"got {llano_gain}")
    brazos_gain = r["results"]["Brazos"]["gains_from_trade_pct"]
    check("  Brazos (didn't trade) gains_pct = 0", brazos_gain == 0.0,
          f"got {brazos_gain}")
    # print_results should not crash on inf
    try:
        sim.print_results()
        check("  print_results handles inf gains", True)
    except Exception as e:
        check("  print_results handles inf gains", False, str(e))


# ────────────────────────────────────────────────────────────────────
# 4. Validation failures
# ────────────────────────────────────────────────────────────────────
def test_validation_failures():
    print("\n[4] validation rejects bad inputs")
    sim = fresh_phase1()

    # Over-allocation
    dec = {n: {"production": {"cloth": 0, "wine": 0}} for n in PHASE1_COUNTRIES}
    dec["Brazos"]["production"]["cloth"] = 200  # too much
    try:
        sim.run_round(dec, [])
        check("  over-allocation rejected", False, "no error raised")
    except ValueError:
        check("  over-allocation rejected", True)

    # Negative
    sim2 = fresh_phase1()
    dec = {n: {"production": {"cloth": 100, "wine": 50}} for n in PHASE1_COUNTRIES}
    dec["Brazos"]["production"] = {"cloth": 200, "wine": -50}  # sums to 150 but negative
    try:
        sim2.run_round(dec, [])
        check("  negative labor rejected", False, "no error raised")
    except ValueError:
        check("  negative labor rejected", True)

    # Missing country
    sim3 = fresh_phase1()
    dec = {n: {"production": {"cloth": 50, "wine": 50}} for n in PHASE1_COUNTRIES if n != "Pecos"}
    try:
        sim3.run_round(dec, [])
        check("  missing country rejected", False, "no error raised")
    except (ValueError, KeyError):
        check("  missing country rejected", True)

    # Bad tariff rate
    sim4 = fresh_phase1()
    dec = {n: {"production": {"cloth": 25 * (sim4.countries[n]['labor']//50),
                              "wine":  sim4.countries[n]['labor'] - 25 * (sim4.countries[n]['labor']//50)}}
           for n in PHASE1_COUNTRIES}
    # Fix Brazos (150)
    dec["Brazos"]["production"] = {"cloth": 75, "wine": 75}
    dec["Brazos"]["tariffs"] = {"Bosque": {"cloth": 1.5}}  # >100%
    try:
        sim4.run_round(dec, [])
        check("  tariff > 1.0 rejected", False, "no error raised")
    except ValueError:
        check("  tariff > 1.0 rejected", True)

    # Round counter should not have incremented after failures
    check("  round_num still 0 after failures", sim4.round_num == 0,
          f"got {sim4.round_num}")


# ────────────────────────────────────────────────────────────────────
# 5. Floating-point tolerance on allocation sums
# ────────────────────────────────────────────────────────────────────
def test_allocation_tolerance():
    print("\n[5] allocation off by tolerance")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 74.7, "wine": 75.0}},  # 149.7, off by 0.3 < 0.5
        "Bosque":  {"production": {"cloth": 30,  "wine": 30}},
        "Llano":  {"production": {"cloth": 50,  "wine": 50}},
        "Trinity":  {"production": {"cloth": 60,  "wine": 60}},
        "Pecos":  {"production": {"cloth": 25,  "wine": 25}},
        "Sabine": {"production": {"cloth": 50,  "wine": 50}},
    }
    try:
        sim.run_round(dec, [])
        check("  tolerance accepts 0.3 deviation", True)
    except Exception as e:
        check("  tolerance accepts 0.3 deviation", False, str(e))

    # Above tolerance
    sim2 = fresh_phase1()
    dec["Brazos"]["production"] = {"cloth": 74.0, "wine": 75.0}  # 149.0, off by 1.0
    try:
        sim2.run_round(dec, [])
        check("  tolerance rejects 1.0 deviation", False, "accepted")
    except ValueError:
        check("  tolerance rejects 1.0 deviation", True)


# ────────────────────────────────────────────────────────────────────
# 6. Trade order dependence
# ────────────────────────────────────────────────────────────────────
def test_trade_order_dependence():
    print("\n[6] trade order dependence (over-export)")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 60, "wine": 0}},  # 240 cloth
        "Llano":  {"production": {"cloth": 0,  "wine": 100}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 0,  "wine": 100}},
    }
    # Bosque has 240 cloth. Try two trades each demanding 150 cloth -> second fails
    trades = [
        ("Bosque", "Llano",  "cloth", 150, "wine", 60),  # ok (90 left)
        ("Bosque", "Sabine", "cloth", 150, "wine", 50),  # fail
    ]
    r = sim.run_round(dec, trades)
    log = "\n".join(r["trade_log"])
    check("  second trade fails (FAILED in log)", "FAILED" in log,
          f"log:\n{log}")
    # Reverse order: same trades, swapped -> still second fails
    sim2 = fresh_phase1()
    trades2 = [
        ("Bosque", "Sabine", "cloth", 150, "wine", 50),
        ("Bosque", "Llano",  "cloth", 150, "wine", 60),
    ]
    r2 = sim2.run_round(dec, trades2)
    log2 = "\n".join(r2["trade_log"])
    check("  reversed order: second still fails", "FAILED" in log2,
          f"log:\n{log2}")


# ────────────────────────────────────────────────────────────────────
# 7. Insane tariffs (100% destruction)
# ────────────────────────────────────────────────────────────────────
def test_full_tariff():
    print("\n[7] 100% tariff destroys all imports")
    sim = fresh_phase1()
    dec = {n: {"production": {"cloth": 30 if n != "Brazos" else 75,
                              "wine":  30 if n != "Brazos" else 75}}
           for n in PHASE1_COUNTRIES}
    # Match endowments
    dec["Brazos"]["production"]  = {"cloth": 75,  "wine": 75}
    dec["Bosque"]["production"]  = {"cloth": 50,  "wine": 10}
    dec["Llano"]["production"]  = {"cloth": 0,   "wine": 100}
    dec["Trinity"]["production"]  = {"cloth": 60,  "wine": 60}
    dec["Pecos"]["production"]  = {"cloth": 25,  "wine": 25}
    dec["Sabine"]["production"] = {"cloth": 50,  "wine": 50}
    dec["Llano"]["tariffs"] = {"Bosque": {"cloth": 1.0}}  # 100%
    trades = [("Bosque", "Llano", "cloth", 100, "wine", 40)]
    r = sim.run_round(dec, trades)
    cal_cloth = r["results"]["Llano"]["consumption"]["cloth"]
    # Llano produced 0 cloth, received 100 but all destroyed -> still 0
    check("  100% tariff: Llano receives 0 cloth", cal_cloth == 0.0,
          f"got {cal_cloth}")
    losses = r["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  tariff_losses = 100 cloth", losses == 100.0, f"got {losses}")


# ────────────────────────────────────────────────────────────────────
# 8. Trade quantity exceeding inventory
# ────────────────────────────────────────────────────────────────────
def test_trade_exceeds_inventory():
    print("\n[8] trade quantity > production -> rejected")
    sim = fresh_phase1()
    dec = {n: {"production": {"cloth": 0, "wine": 0}} for n in PHASE1_COUNTRIES}
    dec["Brazos"]["production"]  = {"cloth": 75,  "wine": 75}
    dec["Bosque"]["production"]  = {"cloth": 60,  "wine": 0}  # 240 cloth
    dec["Llano"]["production"]  = {"cloth": 0,   "wine": 100}  # 300 wine
    dec["Trinity"]["production"]  = {"cloth": 60,  "wine": 60}
    dec["Pecos"]["production"]  = {"cloth": 25,  "wine": 25}
    dec["Sabine"]["production"] = {"cloth": 50,  "wine": 50}
    trades = [("Bosque", "Llano", "cloth", 10000, "wine", 50)]
    r = sim.run_round(dec, trades)
    check("  oversize trade logged FAILED",
          any("FAILED" in line for line in r["trade_log"]),
          f"log:\n{r['trade_log']}")


# ────────────────────────────────────────────────────────────────────
# 9. Phase 2: labor with no capital partner -> zero output
# ────────────────────────────────────────────────────────────────────
def test_phase2_stranded_labor():
    print("\n[9] phase 2: labor with no capital -> q=0")
    sim = fresh_phase2()
    dec = {n: {"production": {
        "labor":   {"cloth": 0, "wine": 0, "machinery": 0},
        "capital": {"cloth": 0, "wine": 0, "machinery": 0},
    }} for n in PHASE2_COUNTRIES}
    # Bosque: all labor in cloth, all capital in machinery
    dec["Bosque"]["production"]["labor"]   = {"cloth": 60, "wine": 0,  "machinery": 0}
    dec["Bosque"]["production"]["capital"] = {"cloth": 0,  "wine": 0,  "machinery": 25}
    # Others: balanced enough to not crash
    for n in ("Brazos", "Llano", "Trinity", "Pecos", "Sabine"):
        L = PHASE2_COUNTRIES[n]["labor"]
        K = PHASE2_COUNTRIES[n]["capital"]
        dec[n]["production"]["labor"]   = {"cloth": L/3, "wine": L/3, "machinery": L - 2*(L/3)}
        dec[n]["production"]["capital"] = {"cloth": K/3, "wine": K/3, "machinery": K - 2*(K/3)}
    r = sim.run_round(dec, [])
    prod = r["results"]["Bosque"]["production"]
    check("  Bosque cloth=0 (no K)", prod["cloth"] == 0.0, f"got {prod['cloth']}")
    check("  Bosque machinery=0 (no L)", prod["machinery"] == 0.0, f"got {prod['machinery']}")
    # Avg wage should still be a number (not NaN)
    avg_wage = r["results"]["Bosque"]["factor_prices"]["avg_wage"]
    check("  Bosque avg_wage is finite", math.isfinite(avg_wage), f"got {avg_wage}")


# ────────────────────────────────────────────────────────────────────
# 10. Shock injection (1-level and 2-level)
# ────────────────────────────────────────────────────────────────────
def test_shocks():
    print("\n[10] inject_shock")
    sim = fresh_phase1()
    sim.inject_shock("test", {"Llano": {"productivity": {"wine": 1.5}}})
    check("  phase 1 shock applied",
          sim.countries["Llano"]["productivity"]["wine"] == 1.5,
          f"got {sim.countries['Llano']['productivity']['wine']}")
    sim.inject_shock("test", {"Brazos": {"labor": 200}})
    check("  phase 1 scalar shock applied",
          sim.countries["Brazos"]["labor"] == 200)

    sim2 = fresh_phase2()
    sim2.inject_shock("tfp shock", {
        "Bosque": {"tech": {"machinery": {"tfp": 1.5, "labor_share": 0.25, "capital_share": 0.75}}}
    })
    check("  phase 2 nested shock applied",
          sim2.countries["Bosque"]["tech"]["machinery"]["tfp"] == 1.5)


# ────────────────────────────────────────────────────────────────────
# 11. Save/restore round-trip
# ────────────────────────────────────────────────────────────────────
def test_state_roundtrip():
    print("\n[11] save/restore")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 30, "wine": 30}},
        "Llano":  {"production": {"cloth": 50, "wine": 50}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 50, "wine": 50}},
    }
    sim.run_round(dec, [])
    state = sim.get_state()
    sim2 = IPESimulation.from_state(state)
    check("  round_num preserved", sim2.round_num == 1)
    check("  history preserved", len(sim2.history) == 1)
    check("  phase preserved", sim2.phase == 1)
    # Can continue running
    sim2.run_round(dec, [])
    check("  can continue after restore", sim2.round_num == 2)


# ────────────────────────────────────────────────────────────────────
# 12. Phase upgrade mid-simulation; plots don't crash
# ────────────────────────────────────────────────────────────────────
def test_phase_upgrade_and_plots():
    print("\n[12] phase upgrade, then plots")
    sim = fresh_phase1()
    dec1 = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 30, "wine": 30}},
        "Llano":  {"production": {"cloth": 50, "wine": 50}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 50, "wine": 50}},
    }
    sim.run_round(dec1, [])
    sim.upgrade_to_phase2(PHASE2_COUNTRIES, PHASE2_GOODS)
    dec2 = {n: {"production": {
        "labor":   {"cloth": PHASE2_COUNTRIES[n]["labor"]/3,
                    "wine":  PHASE2_COUNTRIES[n]["labor"]/3,
                    "machinery": PHASE2_COUNTRIES[n]["labor"] - 2*(PHASE2_COUNTRIES[n]["labor"]/3)},
        "capital": {"cloth": PHASE2_COUNTRIES[n]["capital"]/3,
                    "wine":  PHASE2_COUNTRIES[n]["capital"]/3,
                    "machinery": PHASE2_COUNTRIES[n]["capital"] - 2*(PHASE2_COUNTRIES[n]["capital"]/3)},
    }} for n in PHASE2_COUNTRIES}
    sim.run_round(dec2, [])
    try:
        sim.plot_welfare()  # should now produce 2 subplots
        check("  plot_welfare with two phases", True)
    except Exception as e:
        check("  plot_welfare with two phases", False, str(e))
    try:
        sim.plot_production()
        check("  plot_production after upgrade", True)
    except Exception as e:
        check("  plot_production after upgrade", False, str(e))
    try:
        sim.plot_gains_from_trade()
        check("  plot_gains_from_trade", True)
    except Exception as e:
        check("  plot_gains_from_trade", False, str(e))


# ────────────────────────────────────────────────────────────────────
# 13. Crazy tariffs on both sides
# ────────────────────────────────────────────────────────────────────
def test_both_sides_tariff():
    print("\n[13] both sides apply tariffs (50% / 50%)")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 60, "wine": 0},
                    "tariffs": {"Llano": {"wine": 0.5}}},
        "Llano":  {"production": {"cloth": 0,  "wine": 100},
                    "tariffs": {"Bosque": {"cloth": 0.5}}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 50, "wine": 50}},
    }
    trades = [("Bosque", "Llano", "cloth", 100, "wine", 80)]
    r = sim.run_round(dec, trades)
    # Bosque: keeps 240-100=140 cloth, gets 80*(1-0.5)=40 wine
    # Llano: keeps 300-80=220 wine, gets 100*(1-0.5)=50 cloth
    bosque_c = r["results"]["Bosque"]["consumption"]
    llano_c = r["results"]["Llano"]["consumption"]
    check("  Bosque cloth=140", abs(bosque_c["cloth"] - 140) < 0.01, f"got {bosque_c['cloth']}")
    check("  Bosque wine=40 (50% tariff on imports)",
          abs(bosque_c["wine"] - 40) < 0.01, f"got {bosque_c['wine']}")
    check("  Llano cloth=50 (50% tariff on imports)",
          abs(llano_c["cloth"] - 50) < 0.01, f"got {llano_c['cloth']}")
    check("  Llano wine=220", abs(llano_c["wine"] - 220) < 0.01, f"got {llano_c['wine']}")


# ────────────────────────────────────────────────────────────────────
# 14. Self-trade (country trading with itself) — engine allows it
# ────────────────────────────────────────────────────────────────────
def test_self_trade():
    print("\n[14] self-trade rejected by guard")
    sim = fresh_phase1()
    dec = {n: {"production": {"cloth": PHASE1_COUNTRIES[n]["labor"]//2,
                              "wine":  PHASE1_COUNTRIES[n]["labor"] - PHASE1_COUNTRIES[n]["labor"]//2}}
           for n in PHASE1_COUNTRIES}
    trades = [("Bosque", "Bosque", "cloth", 50, "wine", 20)]
    r = sim.run_round(dec, trades)
    log = "\n".join(r["trade_log"])
    check("  self-trade logged as FAILED", "FAILED" in log and "self-trade" in log,
          f"log: {log!r}")
    # Consumption unchanged
    c = r["results"]["Bosque"]["consumption"]
    check("  Bosque cloth unchanged", abs(c["cloth"] - 120) < 0.01, f"got {c['cloth']}")
    check("  Bosque wine unchanged",  abs(c["wine"]  - 30)  < 0.01, f"got {c['wine']}")


# ────────────────────────────────────────────────────────────────────
# 15. Terms-of-trade and tariff log strings
# ────────────────────────────────────────────────────────────────────
def test_trade_log_format():
    print("\n[15] trade log contains ToT and tariff annotations")
    sim = fresh_phase1()
    dec = {n: {"production": {"cloth": PHASE1_COUNTRIES[n]["labor"]//2,
                              "wine":  PHASE1_COUNTRIES[n]["labor"] - PHASE1_COUNTRIES[n]["labor"]//2}}
           for n in PHASE1_COUNTRIES}
    dec["Llano"]["tariffs"] = {"Bosque": {"cloth": 0.15}}
    trades = [("Bosque", "Llano", "cloth", 40, "wine", 20)]
    r = sim.run_round(dec, trades)
    log = "\n".join(r["trade_log"])
    check("  log contains ToT", "ToT:" in log, f"log:\n{log}")
    check("  log contains tariff", "tariffs:" in log, f"log:\n{log}")
    check("  ToT shows ratio (0.50 wine/cloth)", "0.50 wine" in log, f"log:\n{log}")


# ────────────────────────────────────────────────────────────────────
# 16. Empty trades, empty tariffs
# ────────────────────────────────────────────────────────────────────
def test_empty_inputs():
    print("\n[16] empty trades list")
    sim = fresh_phase1()
    dec = {
        "Brazos":  {"production": {"cloth": 75, "wine": 75}},
        "Bosque":  {"production": {"cloth": 30, "wine": 30}},
        "Llano":  {"production": {"cloth": 50, "wine": 50}},
        "Trinity":  {"production": {"cloth": 60, "wine": 60}},
        "Pecos":  {"production": {"cloth": 25, "wine": 25}},
        "Sabine": {"production": {"cloth": 50, "wine": 50}},
    }
    r = sim.run_round(dec, [])
    check("  empty trades list works", r["trade_log"] == [])


# ────────────────────────────────────────────────────────────────────
# 17. All-or-nothing crazy values
# ────────────────────────────────────────────────────────────────────
def test_extreme_specialization():
    print("\n[17] extreme specialization with profitable trade")
    sim = fresh_phase1()
    # Everyone goes 100% to their CA good
    dec = {
        "Brazos":  {"production": {"cloth": 150, "wine": 0}},
        "Bosque":  {"production": {"cloth": 60,  "wine": 0}},   # 240 cloth
        "Llano":  {"production": {"cloth": 0,   "wine": 100}}, # 300 wine
        "Trinity":  {"production": {"cloth": 120, "wine": 0}},   # 360 cloth
        "Pecos":  {"production": {"cloth": 25,  "wine": 25}},  # split (only producer of both)
        "Sabine": {"production": {"cloth": 0,   "wine": 100}}, # 150 wine
    }
    trades = [
        ("Bosque", "Llano", "cloth", 100, "wine", 80),
        ("Trinity", "Sabine", "cloth", 80, "wine", 60),
        ("Brazos", "Llano", "cloth", 60, "wine", 50),
    ]
    r = sim.run_round(dec, trades)
    for n in PHASE1_COUNTRIES:
        w = r["results"][n]["welfare"]
        check(f"  {n} welfare > 0 (got some of both goods)", w > 0,
              f"got {w}, cons={r['results'][n]['consumption']}")


def main():
    tests = [
        test_phase1_balanced_autarky,
        test_phase1_full_specialization_autarky,
        test_phase1_specialization_with_trade,
        test_validation_failures,
        test_allocation_tolerance,
        test_trade_order_dependence,
        test_full_tariff,
        test_trade_exceeds_inventory,
        test_phase2_stranded_labor,
        test_shocks,
        test_state_roundtrip,
        test_phase_upgrade_and_plots,
        test_both_sides_tariff,
        test_self_trade,
        test_trade_log_format,
        test_empty_inputs,
        test_extreme_specialization,
    ]
    for t in tests:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAIL.append((t.__name__, "exception"))

    print(f"\n\n{'='*60}")
    print(f"  PASSED: {len(PASS)}")
    print(f"  FAILED: {len(FAIL)}")
    if FAIL:
        for name, detail in FAIL:
            print(f"   - {name}: {detail}")
        sys.exit(1)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
