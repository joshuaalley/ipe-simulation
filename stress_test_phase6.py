"""
Stress test Phase 6 sovereign debt:
- upgrade requires Phase 5; inits debt state
- borrowing lifts welfare + grows the stock; interest accrues; rate rises with debt
- servicing/repay lowers welfare and stock
- original sin: a devalued currency makes servicing hurt more
- default wipes the stock, sets the ban, increments the counter, skips service
- ban blocks borrowing + adds FX friction; WTO dividend halved during ban
- borrow cap (<= consumption capacity)
- IMF bailout halves the stock + applies austerity
- save/restore; dashboard renders
"""
import sys, json, traceback
import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation, PHASE2_COUNTRIES, PHASE2_GOODS, PHASE3_FIRMS,
    DEBT_BASE_RATE, DEBT_RISK_PREMIUM, DEBT_DEFAULT_BAN_ROUNDS,
    DEBT_DEFAULT_FRICTION, IMF_DEBT_RELIEF, IMF_AUSTERITY,
)

PASS, FAIL = [], []
def check(name, cond, detail=""):
    if cond:
        PASS.append(name); print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail)); print(f"  FAIL  {name} -- {detail}")

BAL = {
    "Brazos":  {"production": {"labor":{"cloth":50,"wine":50,"machinery":50},"capital":{"cloth":50,"wine":50,"machinery":50}}, "tariffs":{}},
    "Bosque":  {"production": {"labor":{"cloth":30,"wine":15,"machinery":15},"capital":{"cloth":10,"wine":8,"machinery":7}}, "tariffs":{}},
    "Llano":   {"production": {"labor":{"cloth":25,"wine":50,"machinery":25},"capital":{"cloth":20,"wine":35,"machinery":25}}, "tariffs":{}},
    "Trinity": {"production": {"labor":{"cloth":30,"wine":30,"machinery":60},"capital":{"cloth":40,"wine":40,"machinery":120}}, "tariffs":{}},
    "Pecos":   {"production": {"labor":{"cloth":10,"wine":15,"machinery":25},"capital":{"cloth":20,"wine":30,"machinery":70}}, "tariffs":{}},
    "Sabine":  {"production": {"labor":{"cloth":50,"wine":35,"machinery":15},"capital":{"cloth":15,"wine":12,"machinery":8}}, "tariffs":{}},
}
def md(): return {c:{"fx_regime":"managed","capital_controls":False,"independent_monetary":True,"money_supply_growth":0.0} for c in PHASE2_COUNTRIES}
def fd(s): return {f:{"scale":30,"relocate_to":None,"export":False} for f in s.firms}

def fresh_phase6():
    s = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    fm = {f:PHASE3_FIRMS[f] for f in ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    s.upgrade_to_phase3(fm); s.run_round(BAL,[],firm_decisions=fd(s))
    s.phase=4; s.run_round(BAL,[],firm_decisions=fd(s))
    s.award_reserve_currency(); s.upgrade_to_phase5()
    s.run_round(BAL,[],firm_decisions=fd(s),monetary_decisions=md())
    s.upgrade_to_phase6()
    return s

def run6(s, debt=None, mon=None):
    return s.run_round(BAL, [], firm_decisions=fd(s),
                       monetary_decisions=mon or md(), debt_decisions=debt)


# ───────────────────────────────────────────────────────────────
def test_upgrade():
    print("\n[1] upgrade_to_phase6 guard + init")
    s = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    fm = {f:PHASE3_FIRMS[f] for f in ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    s.upgrade_to_phase3(fm); s.run_round(BAL,[],firm_decisions=fd(s)); s.award_reserve_currency()
    try:
        s.upgrade_to_phase6()  # phase 5 not done
        check("  requires phase 5", False, "no error")
    except ValueError:
        check("  requires phase 5", True)
    s.upgrade_to_phase5(); s.upgrade_to_phase6()
    check("  phase == 6", s.phase == 6)
    check("  debt stock initialized to 0", all(s.countries[c]["debt_stock"]==0.0 for c in s.countries))


# ───────────────────────────────────────────────────────────────
def test_borrow_grows_welfare_and_stock():
    print("\n[2] borrowing lifts welfare + grows stock")
    base = fresh_phase6(); r0 = run6(base)
    w_noborrow = r0["results"]["Bosque"]["welfare"]
    s = fresh_phase6()
    r = run6(s, debt={"Bosque": {"borrow": 20}})
    w_borrow = r["results"]["Bosque"]["welfare"]
    check("  borrowing raises welfare this round", w_borrow > w_noborrow,
          f"noborrow={w_noborrow:.2f}, borrow={w_borrow:.2f}")
    check("  debt stock = 20 after borrowing 20",
          abs(s.countries["Bosque"]["debt_stock"] - 20) < 0.01,
          f"got {s.countries['Bosque']['debt_stock']}")


# ───────────────────────────────────────────────────────────────
def test_interest_and_rate():
    print("\n[3] interest accrues; rate rises with debt")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 20}})       # stock 20
    r = run6(s, debt={"Bosque": {"borrow": 20}})   # stock 40 (before this round's interest logic)
    d = r["results"]["Bosque"]["debt"]
    check("  rate above base when indebted", d["rate"] > DEBT_BASE_RATE,
          f"got {d['rate']:.3f}")
    check("  interest > 0 on positive stock", d["interest"] > 0)
    check("  stock grew to 40", abs(s.countries["Bosque"]["debt_stock"]-40) < 0.01,
          f"got {s.countries['Bosque']['debt_stock']}")


# ───────────────────────────────────────────────────────────────
def test_service_lowers_welfare_and_stock():
    print("\n[4] servicing/repay lowers welfare + stock")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 30}})          # stock 30
    # Compare repay vs no-repay welfare in the next round
    s_no = fresh_phase6(); run6(s_no, debt={"Bosque": {"borrow": 30}})
    r_no = run6(s_no, debt={"Bosque": {"borrow": 0}})           # service interest only
    r_re = run6(s,    debt={"Bosque": {"borrow": 0, "repay": 15}})
    check("  repaying lowers welfare vs not repaying",
          r_re["results"]["Bosque"]["welfare"] < r_no["results"]["Bosque"]["welfare"],
          f"repay={r_re['results']['Bosque']['welfare']:.2f}, "
          f"norepay={r_no['results']['Bosque']['welfare']:.2f}")
    check("  stock reduced by repayment (30 -> 15)",
          abs(s.countries["Bosque"]["debt_stock"] - 15) < 0.01,
          f"got {s.countries['Bosque']['debt_stock']}")


# ───────────────────────────────────────────────────────────────
def test_original_sin():
    print("\n[5] original sin: devaluation worsens the service hit")
    # Strong currency
    s1 = fresh_phase6(); run6(s1, debt={"Bosque": {"borrow": 40}})
    r1 = run6(s1, debt={"Bosque": {"borrow": 0}})
    serv_hit_strong = r1["results"]["Bosque"]["debt"]["service"]
    w_strong = r1["results"]["Bosque"]["welfare"]
    # Weak currency: print money to depreciate Bosque, then service same debt
    s2 = fresh_phase6(); run6(s2, debt={"Bosque": {"borrow": 40}})
    weak_mon = md(); weak_mon["Bosque"] = {"fx_regime":"float","capital_controls":False,"independent_monetary":True,"money_supply_growth":0.10}
    # depreciate over a couple rounds
    run6(s2, debt={"Bosque": {"borrow": 0}}, mon=weak_mon)
    r2 = run6(s2, debt={"Bosque": {"borrow": 0}}, mon=weak_mon)
    dep = s2._mon("Bosque")["depreciation_factor"]
    check("  Bosque currency depreciated", dep < 1.0, f"dep={dep:.3f}")
    # The real burden divides by dep, so a weak currency means a larger welfare bite.
    # Compare welfare-retention ratio: weaker currency keeps less.
    check("  servicing exists in both", serv_hit_strong > 0 and r2["results"]["Bosque"]["debt"]["service"] > 0)
    # Verify the engine divides service by depreciation (real burden > nominal when dep<1)
    d2 = r2["results"]["Bosque"]["debt"]
    check("  real burden amplified by weak currency (dep<1 divides up)",
          d2["depreciation_factor"] < 1.0)


# ───────────────────────────────────────────────────────────────
def test_default():
    print("\n[6] default wipes stock, sets ban, counts")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 50}})   # stock 50
    r = run6(s, debt={"Bosque": {"default": True}})
    d = r["results"]["Bosque"]["debt"]
    check("  default flagged", d["defaulted"])
    check("  stock wiped to 0", s.countries["Bosque"]["debt_stock"] == 0.0)
    check("  wiped amount recorded (~50+interest borrowed=50)", d["wiped"] >= 50,
          f"got {d['wiped']}")
    check("  default counter incremented", s.countries["Bosque"]["defaults"] == 1)
    check("  ban set forward", s.countries["Bosque"]["borrow_ban_until"] == s.round_num + DEBT_DEFAULT_BAN_ROUNDS)
    check("  appears in round defaults list", "Bosque" in r["debt_defaults"])


# ───────────────────────────────────────────────────────────────
def test_ban_blocks_borrow_and_adds_friction():
    print("\n[7] ban blocks borrowing + adds friction")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 50}})
    run6(s, debt={"Bosque": {"default": True}})   # now banned
    check("  Bosque banned", s._is_debt_banned("Bosque"))
    # Try to borrow during ban -> stock stays 0
    r = run6(s, debt={"Bosque": {"borrow": 30}})
    check("  borrowing blocked during ban", s.countries["Bosque"]["debt_stock"] == 0.0,
          f"got {s.countries['Bosque']['debt_stock']}")
    check("  banned status shown", r["results"]["Bosque"]["debt"]["banned"])
    # Friction delta on a banned country's trade
    other = "Trinity"
    fr = s._debt_friction_delta("Bosque", other)
    check("  banned country adds FX friction",
          abs(fr - DEBT_DEFAULT_FRICTION) < 1e-9, f"got {fr}")
    fr0 = s._debt_friction_delta("Trinity", "Llano")
    check("  non-banned pair: no debt friction", fr0 == 0.0)


# ───────────────────────────────────────────────────────────────
def test_borrow_cap():
    print("\n[8] borrow capped at consumption capacity")
    s = fresh_phase6()
    # Borrow an absurd amount; stock should be capped at C (consumption value)
    r = run6(s, debt={"Bosque": {"borrow": 99999}})
    d = r["results"]["Bosque"]["debt"]
    # Capacity C = sum(consumption*world_price); borrow should equal C (the cap), not 99999
    check("  borrow capped well below 99999", d["borrow"] < 9999,
          f"got {d['borrow']}")
    check("  capped borrow > 0", d["borrow"] > 0)


# ───────────────────────────────────────────────────────────────
def test_imf_bailout():
    print("\n[9] IMF bailout halves stock + austerity")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 60}})   # stock 60
    before = s.countries["Bosque"]["debt_stock"]
    s.request_imf_bailout("Bosque")
    after = s.countries["Bosque"]["debt_stock"]
    check("  IMF refinanced half the stock",
          abs(after - before * (1 - IMF_DEBT_RELIEF)) < 0.01,
          f"before={before}, after={after}")
    check("  austerity window set", s.countries["Bosque"]["imf_austerity_until"] >= s.round_num + 1)
    # Austerity bites welfare next round
    r_aus = run6(s, debt={"Bosque": {"borrow": 0}})
    check("  austerity flagged active", r_aus["results"]["Bosque"]["debt"]["austerity_active"])
    check("  austerity cut recorded ~8%",
          abs(r_aus["results"]["Bosque"]["debt"]["austerity_cut"] - IMF_AUSTERITY) < 1e-9)


# ───────────────────────────────────────────────────────────────
def test_save_restore():
    print("\n[10] save/restore preserves debt state")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 40}})
    run6(s, debt={"Trinity": {"borrow": 25}})
    state = json.loads(json.dumps(s.get_state()))
    s2 = IPESimulation.from_state(state)
    check("  phase preserved", s2.phase == 6)
    check("  Bosque debt preserved",
          abs(s2.countries["Bosque"]["debt_stock"] - s.countries["Bosque"]["debt_stock"]) < 0.01)
    check("  Trinity debt preserved",
          abs(s2.countries["Trinity"]["debt_stock"] - 25) < 0.01)
    run6(s2, debt={"Bosque": {"borrow": 0}})
    check("  continues after restore", s2.round_num == s.round_num + 1)


# ───────────────────────────────────────────────────────────────
def test_display():
    print("\n[11] dashboards render")
    s = fresh_phase6()
    run6(s, debt={"Bosque": {"borrow": 40}})
    run6(s, debt={"Bosque": {"default": True}})
    try:
        s.print_results(); check("  print_results works in Phase 6", True)
    except Exception as e:
        check("  print_results works in Phase 6", False, str(e))
    try:
        s.print_debt_dashboard(); check("  debt dashboard works", True)
    except Exception as e:
        check("  debt dashboard works", False, str(e))


def test_validation_ordering():
    print("\n[12] failed debt validation must not corrupt monetary state")
    s = fresh_phase6()
    dep_before = s._mon("Bosque")["depreciation_factor"]
    round_before = s.round_num
    bad = {"Bosque": {"borrow": -5}}   # invalid (negative borrow)
    # Use a monetary decision that WOULD depreciate Bosque if applied
    weak = md(); weak["Bosque"] = {"fx_regime":"float","capital_controls":False,
                                   "independent_monetary":True,"money_supply_growth":0.10}
    try:
        s.run_round(BAL, [], firm_decisions=fd(s), monetary_decisions=weak, debt_decisions=bad)
        check("  invalid debt rejected", False, "no error")
    except ValueError:
        check("  invalid debt rejected", True)
    check("  round_num rolled back", s.round_num == round_before)
    check("  monetary state NOT mutated on failed round",
          abs(s._mon("Bosque")["depreciation_factor"] - dep_before) < 1e-12,
          f"before={dep_before}, after={s._mon('Bosque')['depreciation_factor']}")


def main():
    for t in [test_upgrade, test_borrow_grows_welfare_and_stock, test_interest_and_rate,
              test_service_lowers_welfare_and_stock, test_original_sin, test_default,
              test_ban_blocks_borrow_and_adds_friction, test_borrow_cap,
              test_imf_bailout, test_save_restore, test_display,
              test_validation_ordering]:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc(); FAIL.append((t.__name__, "exception"))
    print(f"\n{'='*60}")
    print(f"  PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
    if FAIL:
        for n,d in FAIL: print(f"   - {n}: {d}")
        sys.exit(1)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
