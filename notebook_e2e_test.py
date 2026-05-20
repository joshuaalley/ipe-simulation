"""
End-to-end notebook execution test: walk through every Phase 1-3 round,
filling in the empty-seed rounds with sensible values. Verifies that the
notebook's cell sequence works once an instructor has entered real decisions.
"""
import matplotlib
matplotlib.use("Agg")
import sys

from engine import (
    IPESimulation,
    PHASE1_COUNTRIES, PHASE1_GOODS,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    PHASE3_FIRMS,
)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  -- {detail}" if not cond else ""))

# ── Phase 1 ──
sim = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)

# R1 — fill in real values for the empty Phase 1 autarky seed
r1_dec = {
    "Brazos":  {"production": {"cloth": 75, "wine": 75}},
    "Bosque":  {"production": {"cloth": 30, "wine": 30}},
    "Llano":  {"production": {"cloth": 50, "wine": 50}},
    "Trinity":  {"production": {"cloth": 60, "wine": 60}},
    "Pecos":  {"production": {"cloth": 25, "wine": 25}},
    "Sabine": {"production": {"cloth": 50, "wine": 50}},
}
sim.run_round(r1_dec, [])
check("Phase 1 R1 ran", sim.round_num == 1)

# R2 — from notebook
r2_dec = {
    "Brazos":  {"production": {"cloth": 75, "wine": 75}},
    "Bosque":  {"production": {"cloth": 50, "wine": 10}},
    "Llano":  {"production": {"cloth": 20, "wine": 80}},
    "Trinity":  {"production": {"cloth": 80, "wine": 40}},
    "Pecos":  {"production": {"cloth": 25, "wine": 25}},
    "Sabine": {"production": {"cloth": 30, "wine": 70}},
}
r2_trades = [
    ("Bosque", "Llano", "cloth", 80, "wine", 50),
    ("Trinity", "Sabine", "cloth", 40, "wine", 30),
]
sim.run_round(r2_dec, r2_trades)
check("Phase 1 R2 ran with trades", sim.round_num == 2)

# R3 — from notebook (with tariffs)
r3_dec = {
    "Brazos":  {"production": {"cloth": 75, "wine": 75}, "tariffs": {}},
    "Bosque":  {"production": {"cloth": 55, "wine": 5},  "tariffs": {}},
    "Llano":  {"production": {"cloth": 10, "wine": 90},
                "tariffs": {"Bosque": {"cloth": 0.15}}},
    "Trinity":  {"production": {"cloth": 90, "wine": 30}, "tariffs": {}},
    "Pecos":  {"production": {"cloth": 30, "wine": 20}, "tariffs": {}},
    "Sabine": {"production": {"cloth": 20, "wine": 80},
                "tariffs": {"Trinity": {"cloth": 0.20}}},
}
r3_trades = [
    ("Bosque", "Llano", "cloth", 100, "wine", 60),
    ("Trinity", "Sabine", "cloth", 50, "wine", 40),
    ("Pecos", "Brazos", "cloth", 30, "wine", 25),
]
sim.run_round(r3_dec, r3_trades)
check("Phase 1 R3 ran with tariffs", sim.round_num == 3)

# Upgrade to Phase 2
sim.upgrade_to_phase2(PHASE2_COUNTRIES, PHASE2_GOODS)
check("Upgraded to Phase 2", sim.phase == 2)

# R4 — Phase 2 autarky (fill in empty seeds)
r4_dec = {
    "Brazos": {"production": {"labor":   {"cloth": 50, "wine": 50, "machinery": 50},
                              "capital": {"cloth": 50, "wine": 50, "machinery": 50}}},
    "Bosque": {"production": {"labor":   {"cloth": 30, "wine": 15, "machinery": 15},
                              "capital": {"cloth": 10, "wine": 8,  "machinery": 7}}},
    "Llano": {"production": {"labor":   {"cloth": 25, "wine": 50, "machinery": 25},
                              "capital": {"cloth": 20, "wine": 35, "machinery": 25}}},
    "Trinity": {"production": {"labor":   {"cloth": 30, "wine": 30, "machinery": 60},
                              "capital": {"cloth": 40, "wine": 40, "machinery": 120}}},
    "Pecos": {"production": {"labor":   {"cloth": 10, "wine": 15, "machinery": 25},
                              "capital": {"cloth": 20, "wine": 30, "machinery": 70}}},
    "Sabine":{"production": {"labor":   {"cloth": 50, "wine": 35, "machinery": 15},
                              "capital": {"cloth": 15, "wine": 12, "machinery": 8}}},
}
sim.run_round(r4_dec, [])
check("Phase 2 R4 ran (autarky)", sim.round_num == 4)

# Upgrade to Phase 3 (notebook cell with F1-F10)
firms_for_class = {fid: PHASE3_FIRMS[fid] for fid in
                   ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
sim.upgrade_to_phase3(firms_for_class)
check("Upgraded to Phase 3", sim.phase == 3)
check("10 firms loaded in notebook flow", len(sim.firms) == 10)

# print_firm_roster
sim.print_firm_roster()
check("print_firm_roster works", True)

# print_decision_forms + print_firm_decision_forms (just verify no crash)
try:
    sim.print_decision_forms()
    sim.print_firm_decision_forms()
    check("Phase 3 decision forms print", True)
except Exception as e:
    check("Phase 3 decision forms print", False, str(e))

# Round 8 — same country layout as R4 but with firm_decisions
r8_dec = r4_dec  # same allocations work
r8_firm_dec = {
    "F1":  {"scale": 30, "relocate_to": None, "export": False},
    "F2":  {"scale": 30, "relocate_to": None, "export": False},
    "F3":  {"scale": 30, "relocate_to": None, "export": False},
    "F4":  {"scale": 30, "relocate_to": None, "export": False},
    "F5":  {"scale": 30, "relocate_to": None, "export": False},
    "F6":  {"scale": 30, "relocate_to": None, "export": False},
    "F7":  {"scale": 30, "relocate_to": None, "export": False},
    "F8":  {"scale": 30, "relocate_to": None, "export": False},
    "F9":  {"scale": 30, "relocate_to": None, "export": False},
    "F10": {"scale": 30, "relocate_to": None, "export": False},
}
r8 = sim.run_round(r8_dec, [], firm_decisions=r8_firm_dec)
check("Round 8 (first Phase 3 round) ran", sim.round_num == 5)
check("firms in result", "firms" in r8)
check("firm profits computed", all(
    "profit" in r8["firms"][fid] for fid in sim.firms
))

# print_results in Phase 3 (includes firms table)
try:
    sim.print_results()
    check("print_results works after Phase 3 round", True)
except Exception as e:
    check("print_results works after Phase 3 round", False, str(e))

# plot_welfare with phase split
try:
    sim.plot_welfare()
    check("plot_welfare splits across all 3 phases", True)
except Exception as e:
    check("plot_welfare splits across all 3 phases", False, str(e))

# plot_firm_locations (Phase 3+)
try:
    sim.plot_firm_locations()
    check("plot_firm_locations renders", True)
except Exception as e:
    check("plot_firm_locations renders", False, str(e))

# plot_trade_network — autarky round (no trades): should not crash
try:
    sim.plot_trade_network()
    check("plot_trade_network handles no-trades round", True)
except Exception as e:
    check("plot_trade_network handles no-trades round", False, str(e))

# ── Phase 4 walk-through ──
sim.phase = 4
r9_firm_dec = {fid: {"scale": 30, "relocate_to": None, "export": True}
               for fid in sim.firms}

# Round 12: first Phase 4 round (export selection live)
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec)
check("Phase 4 R12 (first Melitz round) ran", sim.phase == 4)
# Low-prod F3 should be in the red this round (profit = 3 - 8 = -5)
last = sim.history[-1]["firms"]["F3"]["profit"]
check("R12: F3 (LOW) export loses money", last < 0,
      f"F3 profit = {last}")

# Round 14: structural shock
sim.inject_productivity_surge("Pecos", "machinery", 2.5,
    description="Automation breakthrough in Pecos machinery")
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec)
check("R14: productivity surge applied", sim.countries["Pecos"]["tech"]["machinery"]["tfp"] > 2.0)

# Round 15: populist backlash
sim.inject_populist_backlash("Sabine", tariff_floor=0.25, mnc_tax_rate=0.10)
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec)
check("R15: tariff floor set", sim.countries["Sabine"]["tariff_floor"] == 0.25)
check("R15: MNC tax rate set", sim.countries["Sabine"]["mnc_tax_rate"] == 0.10)
check("R15: MNC tax ledger growing for Sabine",
      sim.mnc_tax_revenue.get("Sabine", 0) > 0)

# Round 16: ceremony
ranking = sim.award_reserve_currency()
check("R16: reserve currency awarded", sim.reserve_currency_holder is not None)
check("R16: ranking returned", len(ranking) == 6)
try:
    sim.print_firm_rankings()
    check("R16: firm rankings prints", True)
except Exception as e:
    check("R16: firm rankings prints", False, str(e))

# print_results still works in Phase 4 (with all extras)
try:
    sim.print_results()
    check("print_results works in Phase 4 with all extras", True)
except Exception as e:
    check("print_results works in Phase 4 with all extras", False, str(e))

# ── Phase 5 walk-through ──
sim.upgrade_to_phase5()
check("Phase 5 active", sim.phase == 5)

def safe_md():
    return {c: {"fx_regime": "managed", "capital_controls": False,
                "independent_monetary": True, "money_supply_growth": 0.0}
            for c in PHASE2_COUNTRIES}

# R17: everyone safe
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md())
check("R17 ran (no crisis)", not sim.history[-1]["results"]["Brazos"]["monetary"]["crisis"])

# R18: force Sabine to print
sim.inject_monetary_shock("Sabine", 0.10)
md18 = safe_md()
md18["Sabine"] = {"fx_regime": "float", "capital_controls": False,
                  "independent_monetary": True, "money_supply_growth": 0.10}
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=md18)
check("R18 Sabine currency depreciated",
      sim.history[-1]["results"]["Sabine"]["monetary"]["depreciation_factor"] < 1.0)

# R19: drive Pecos into a crisis over two overreach rounds
over = safe_md()
over["Pecos"] = {"fx_regime": "peg", "capital_controls": False,
                 "independent_monetary": True, "money_supply_growth": 0.0}
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=over)
check("R19a Pecos warned", sim.history[-1]["results"]["Pecos"]["monetary"]["warning"])
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=over)
check("R19b Pecos crisis fired", sim.history[-1]["results"]["Pecos"]["monetary"]["crisis"])

# R20: monetary union
sim.form_monetary_union("Bosque", "Llano", name="BL")
mdU = safe_md()
for c in ("Bosque", "Llano"):
    mdU[c] = {"fx_regime": "managed", "capital_controls": False,
              "independent_monetary": True, "money_supply_growth": 0.0}
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=mdU)
check("R20 union members share state",
      sim._mon("Bosque") is sim._mon("Llano"))

# R22: manual attack + capital flight
sim.inject_speculative_attack("Trinity")
sim.inject_capital_flight("Sabine", severity=0.6)
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md())
check("R22 forced attack delivered crisis on Trinity",
      sim.history[-1]["results"]["Trinity"]["monetary"]["crisis"])

# Phase 5 displays
try:
    sim.print_results()
    sim.print_monetary_dashboard()
    sim.plot_currency_health()
    check("Phase 5 displays render", True)
except Exception as e:
    check("Phase 5 displays render", False, str(e))

# Save/restore through Phase 5
import json as _json
state = _json.loads(_json.dumps(sim.get_state()))
sim_r = IPESimulation.from_state(state)
check("save/restore preserves Phase 5 + union",
      sim_r.phase == 5 and "BL" in sim_r.monetary_unions)

# ── Phase 6 walk-through (Sovereign Debt) ──
sim.upgrade_to_phase6()
check("Phase 6 (debt) active", sim.phase == 6)
check("debt stock initialized", all(sim.countries[c]["debt_stock"] == 0.0 for c in sim.countries))

# R21: borrowing introduced
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md(),
              debt_decisions={"Bosque": {"borrow": 30}})
check("R21 Bosque borrowed (stock 30)", abs(sim.countries["Bosque"]["debt_stock"] - 30) < 0.01)

# R22: debt crisis -> default
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md(),
              debt_decisions={"Bosque": {"default": True}})
check("R22 Bosque defaulted", "Bosque" in sim.history[-1]["debt_defaults"])
check("R22 default wiped stock", sim.countries["Bosque"]["debt_stock"] == 0.0)
check("R22 default set ban", sim._is_debt_banned("Bosque"))

# ── Phase 7 walk-through (Institutions) ──
sim.upgrade_to_phase7()
check("Phase 7 (institutions) active", sim.phase == 7)
check("hegemon seeded from reserve holder", sim.hegemon == sim.reserve_currency_holder)

# R23: WTO round
inst23 = {
    "hegemon_provides": True,
    "Trinity": {"join_wto": True, "bound_tariffs": {"machinery": 0.10}},
    "Llano":   {"join_wto": True},
    "Pecos":   {"join_wto": True},
}
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec,
              monetary_decisions=safe_md(), institutional_decisions=inst23)
check("R23 WTO members recorded",
      sim.history[-1]["results"]["Trinity"]["institutions"]["wto_member"])
check("R23 hegemon pays provision cost",
      sim.history[-1]["results"][sim.hegemon]["institutions"]["hegemon_cost"] > 0)

# IMF bailout path (alternative to default) — give Pecos some debt then bail it out
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md(),
              debt_decisions={"Pecos": {"borrow": 40}},
              institutional_decisions={"hegemon_provides": True})
before_imf = sim.countries["Pecos"]["debt_stock"]
sim.request_imf_bailout("Pecos")
check("IMF halved Pecos debt", sim.countries["Pecos"]["debt_stock"] < before_imf)

# R24: side payment + hegemonic challenge
cum = sim._cumulative_welfare()
challenger = max((c for c in sim.countries if c != sim.hegemon), key=lambda c: cum[c])
backers = [c for c in sim.countries if c not in (sim.hegemon, challenger)]
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md(),
              institutional_decisions={"hegemon_provides": True},
              side_payments=[(challenger, backers[0], "machinery", 5)])
ok = sim.challenge_hegemon(challenger, backers=backers)
check("R24 all-but-hegemon coalition wins", ok)
check("R24 reserve currency moved with hegemony",
      sim.reserve_currency_holder == challenger)

# R25: global crisis (hegemon withholds, partial WTO)
sim.set_hegemon_provision(False)
sim.inject_global_crisis(0.20)
sim.run_round(r4_dec, [], firm_decisions=r9_firm_dec, monetary_decisions=safe_md(),
              institutional_decisions={"hegemon_provides": False})
check("R25 global crisis scaled welfare",
      sim.history[-1]["global_crisis_factor"] < 1.0)
check("R25 crisis is one-shot", sim._pending_global_crisis is None)

# Phase 6+7 displays + save/restore
try:
    sim.print_results()
    sim.print_debt_dashboard()
    sim.print_institutions_dashboard()
    check("Phase 6/7 displays render", True)
except Exception as e:
    check("Phase 6/7 displays render", False, str(e))

state7 = _json.loads(_json.dumps(sim.get_state()))
sim7 = IPESimulation.from_state(state7)
check("save/restore preserves Phase 7 (hegemon + WTO + debt)",
      sim7.phase == 7 and sim7.hegemon == sim.hegemon
      and "debt_stock" in sim7.countries["Pecos"])

print(f"\nPASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    for name, detail in FAIL:
        print(f"  - {name}: {detail}")
    sys.exit(1)
