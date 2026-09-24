"""
Stress test Phase 4 mechanics:
- Productivity surge (structural shock; neutral framing)
- Populist backlash (tariff floor + MNC tax)
- MNC tax ledger (separate from welfare)
- Tariff floor enforcement (max with declared)
- Reserve currency awarding (cumulative welfare; Phase-4 tiebreaker)
- Firm rankings printer
- Save/restore with Phase 4 state
- Export premium: selection at the right tier, the tariff gate, no free lunch
"""
import sys, json, copy, io, contextlib, traceback
import matplotlib
matplotlib.use("Agg")

import engine
from engine import (
    IPESimulation,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    PHASE3_FIRMS, WORLD_PRICES, EXPORT_MARKET_GAIN,
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
    trades = [("Bosque", "Llano", "cloth", 20, "wine", 10)]
    r = sim.run_round(BAL_DEC, trades, firm_decisions=fd(sim))
    # Tariff loss = 20 * 0.30 = 6
    loss = r["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  tariff floor 30% applies despite no declared tariff",
          abs(loss - 6.0) < 0.01, f"got {loss}")

    # Now declare a HIGHER tariff than the floor; the higher one should win
    sim2 = fresh_phase4()
    sim2.inject_populist_backlash("Llano", tariff_floor=0.10, mnc_tax_rate=0.0)
    dec = {k: dict(v) for k, v in BAL_DEC.items()}
    dec["Llano"] = dict(dec["Llano"])
    dec["Llano"]["tariffs"] = {"Bosque": {"cloth": 0.50}}
    r2 = sim2.run_round(dec, trades, firm_decisions=fd(sim2))
    loss2 = r2["results"]["Llano"]["tariff_losses"]["cloth"]
    check("  declared 50% > floor 10% -> 50% applies",
          abs(loss2 - 10.0) < 0.01, f"got {loss2}")


# ────────────────────────────────────────────────────────────────────
# 4. MNC tax: deducted from firm profit, collected by the host
# ────────────────────────────────────────────────────────────────────
def test_mnc_tax_ledger():
    print("\n[4] MNC tax: collected by the host, paid by the owner")
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
    # The host keeps what it collects: same allocations, so Bosque's welfare
    # rises by exactly the tax as a share of its consumption at world prices.
    post_welfare = r_post["results"]["Bosque"]["welfare"]
    cons = r_post["results"]["Bosque"]["consumption"]
    cap = sum(cons[g] * WORLD_PRICES[g] for g in PHASE2_GOODS)
    check("  Bosque welfare x (1 + 7.80 / C): the host keeps the tax",
          abs(post_welfare / pre_welfare - (1 + 7.80 / cap)) < 1e-12,
          f"pre={pre_welfare}, post={post_welfare}, C={cap}")
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
def test_mnc_tax_starts_with_firms():
    print("\n[5] the MNC tax starts with the firms (Phase 3), not before")
    sim = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    r2 = sim.run_round(BAL_DEC, [])
    check("  Phase 2: no MNC tax anywhere in the result",
          "mnc_tax_this_round" not in r2
          and all("mnc_tax" not in v for v in r2["results"].values()))
    firms = {fid: PHASE3_FIRMS[fid] for fid in
             ["F1","F2","F3","F4","F5","F6","F7","F8","F9","F10"]}
    sim.upgrade_to_phase3(firms)  # phase = 3
    # A populist minimum set by a backlash bites as soon as firms arrive:
    # F1 (HIGH cloth, Bosque) at scale 30 earns 39, so 20% is 7.8.
    sim.countries["Bosque"]["mnc_tax_rate"] = 0.20
    r = sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    check("  Phase 3: F1 pays the 20% minimum (7.80)",
          abs(r["firms"]["F1"]["mnc_tax"] - 7.8) < 1e-9,
          f"got {r['firms']['F1']['mnc_tax']}")
    check("  Phase 3: the round records what each host collected",
          abs(r["mnc_tax_this_round"]["Bosque"] - 7.8) < 1e-9)


# ────────────────────────────────────────────────────────────────────
# 6. award_reserve_currency picks top average gains from trade
# ────────────────────────────────────────────────────────────────────
def test_award_reserve_currency():
    print("\n[6] reserve currency awarded to top average gains from trade")
    sim = fresh_phase4()
    # Already has 2 rounds of history (one Phase 3, one Phase 4)
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    sim.run_round(BAL_DEC, [], firm_decisions=fd(sim))
    ranking = sim.award_reserve_currency()
    check("  ranking has all 6 countries", len(ranking) == 6)
    check("  reserve_currency_holder set",
          sim.reserve_currency_holder == ranking[0])
    # Top country should have the highest AVERAGE gains-from-trade %
    avg_gains = {
        n: sum(h["results"][n]["gains_from_trade_pct"] for h in sim.history)
           / len(sim.history)
        for n in sim.countries
    }
    top_by_gains = max(avg_gains, key=avg_gains.get)
    check(f"  top in ranking ({ranking[0]}) is top in avg gains ({top_by_gains})",
          ranking[0] == top_by_gains)

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
    # Now F3 (LOW prod cloth in Sabine) exporting. Last round had no
    # tariffs, so the premium is the full 25%:
    # output = 21, revenue = 21 * 1.25 = 26.25
    # op_cost = 18, fixed_export = 8, mnc_tax = 26.25 * 0.15 = 3.9375
    # profit = 26.25 - 18 - 8 - 3.9375 = -3.6875
    # Staying home: 21 - 18 - 21 * 0.15 = -0.15
    home = copy.deepcopy(sim)
    fde = fd(sim, export=True)
    r = sim.run_round(BAL_DEC, [], firm_decisions=fde)
    check("  F3 (LOW + populist host + export) profit = -3.6875",
          abs(r["firms"]["F3"]["profit"] - (-3.6875)) < 1e-9,
          f"got {r['firms']['F3']['profit']}")
    check("  populist tax is levied on the premium too",
          abs(r["firms"]["F3"]["mnc_tax"] - 3.9375) < 1e-9,
          f"got {r['firms']['F3']['mnc_tax']}")
    rh = home.run_round(BAL_DEC, [], firm_decisions=fd(home))
    check("  ...and exporting still costs F3 money vs staying home (-0.15)",
          abs(rh["firms"]["F3"]["profit"] - (-0.15)) < 1e-9
          and r["firms"]["F3"]["profit"] < rh["firms"]["F3"]["profit"],
          f"home {rh['firms']['F3']['profit']}")
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
    check("  output shows the MNC tax collected", "MNC TAX COLLECTED" in out)
    check("  output says the host keeps it",
          "counts toward its welfare" in out)


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


# ────────────────────────────────────────────────────────────────────
# 12. Export premium: Melitz selection, the tariff gate, no free lunch
# ────────────────────────────────────────────────────────────────────
def after_tariffs(tariff=None):
    """Phase 4 sim whose LAST round applied tariff(importer, partner, good)."""
    sim = fresh_phase4()
    dec = copy.deepcopy(BAL_DEC)
    if tariff:
        for imp in dec:
            dec[imp]["tariffs"] = {p: {g: tariff(imp, p, g) for g in PHASE2_GOODS}
                                   for p in dec if p != imp}
    sim.run_round(dec, [], firm_decisions=fd(sim))
    return sim


def export_gain(sim, scale=40):
    """Per firm: profit exporting minus profit staying home, next round."""
    a, b = copy.deepcopy(sim), copy.deepcopy(sim)
    home = a.run_round(BAL_DEC, [], firm_decisions=fd(a, scale=scale))["firms"]
    out = b.run_round(BAL_DEC, [], firm_decisions=fd(b, scale=scale,
                                                     export=True))["firms"]
    return {f: out[f]["profit"] - home[f]["profit"] for f in sim.firms}, out


def hand_gain(sim, fid, premium, scale=40):
    cfg = sim.firm_config[fid]
    units = scale * cfg["productivity"]
    return units * WORLD_PRICES[cfg["industry"]] * premium - cfg["fixed_export_cost"]


def test_export_premium():
    print("\n[12] export premium: selection, tariff gate, no free lunch")
    tier = lambda sim, f: {1.3: "HIGH", 1.0: "MED", 0.7: "LOW"}[
        sim.firm_config[f]["productivity"]]

    # Phase 3: the export box is inert -- no premium, no fixed cost
    p3 = IPESimulation(PHASE2_COUNTRIES, PHASE2_GOODS, phase=2)
    p3.upgrade_to_phase3({f: PHASE3_FIRMS[f] for f in ("F1", "F3", "F7")})
    r = p3.run_round(BAL_DEC, [], firm_decisions={
        f: {"scale": 40, "relocate_to": None, "export": True} for f in p3.firms})
    f1 = r["firms"]["F1"]
    check("  Phase 3: export box is inert (no premium, no fixed cost)",
          f1["export_premium"] == 0.0 and f1["fixed_cost"] == 0.0
          and abs(f1["revenue"] - 52.0) < 1e-9, str(f1["revenue"]))

    # open world: the full premium, and the cutoff falls between MED and LOW
    sim = after_tariffs()
    gain, out = export_gain(sim)
    check("  open world: every exporter earns the full 25%",
          all(abs(out[f]["export_premium"] - EXPORT_MARKET_GAIN) < 1e-12
              for f in sim.firms))
    check("  gain = units x price x premium - fixed cost, for every firm",
          all(abs(gain[f] - hand_gain(sim, f, 0.25)) < 1e-9 for f in sim.firms),
          str({f: round(g, 3) for f, g in gain.items()}))
    check("  HIGH and MED firms profit from exporting at full scale",
          all(gain[f] > 0 for f in sim.firms if tier(sim, f) in ("HIGH", "MED")))
    check("  LOW firms lose by exporting at full scale",
          all(gain[f] < 0 for f in sim.firms if tier(sim, f) == "LOW"))
    check("  break-even is 32 units for every firm (f_x / price = 8)",
          all(abs(sim.export_breakeven(f) - 32.0) < 1e-9 for f in sim.firms))

    # no free lunch: at no scale does a LOW firm gain by exporting
    worst = max(export_gain(sim, scale=k)[0][f]
                for k in range(0, 41, 2)
                for f in sim.firms if tier(sim, f) == "LOW")
    check("  no free lunch: LOW never gains by exporting at any scale",
          worst < 0, f"best LOW gain {worst:+.3f}")

    # the gate: 20% everywhere leaves MED dead even; 50% shuts everyone out
    s20 = after_tariffs(lambda i, p, g: 0.20)
    g20, o20 = export_gain(s20)
    check("  20% world tariffs: premium falls to 20%",
          all(abs(o20[f]["export_premium"] - 0.20) < 1e-12 for f in s20.firms))
    check("  20% world tariffs: MED firms exactly break even",
          all(abs(g20[f]) < 1e-9 for f in s20.firms if tier(s20, f) == "MED"),
          str({f: g20[f] for f in s20.firms if tier(s20, f) == "MED"}))
    s50 = after_tariffs(lambda i, p, g: 0.50)
    g50, _ = export_gain(s50)
    check("  50% world tariffs: no firm's exports pay",
          all(v < 0 for v in g50.values()), str(max(g50.values())))

    # tariffs are bilateral: one country's tariff on one host's good
    sb = after_tariffs(lambda i, p, g: 0.60 if (i, p, g) == ("Bosque", "Sabine",
                                                            "cloth") else 0.0)
    others = len(sb.countries) - 1
    check("  bilateral: only that host's good is hit, by 60% / "
          f"{others} others",
          abs(sb.export_premium("Sabine", "cloth")
              - 0.25 * (1 - 0.60 / others)) < 1e-12
          and sb.export_premium("Sabine", "wine") == 0.25
          and sb.export_premium("Llano", "cloth") == 0.25)

    # a populist floor raises the wall on everything the country imports
    sp = fresh_phase4()
    sp.inject_populist_backlash("Bosque", tariff_floor=0.30, mnc_tax_rate=0.0)
    sp.run_round(BAL_DEC, [], firm_decisions=fd(sp))
    wall = sp.history[-1]["tariff_wall"]["Bosque"]
    check("  populist floor counts in the wall, on every partner and good",
          all(wall[p][g] == 0.30 for p in sp.countries if p != "Bosque"
              for g in PHASE2_GOODS))

    # the flag turns the gate off
    engine.EXPORT_TARIFF_GATE = False
    try:
        check("  EXPORT_TARIFF_GATE=False: flat 25% whatever the tariffs",
              s50.export_premium("Llano", "cloth") == EXPORT_MARKET_GAIN)
    finally:
        engine.EXPORT_TARIFF_GATE = True

    # snapshots: the wall survives a save, and an old save still loads
    s2 = IPESimulation.from_state(json.loads(json.dumps(s20.get_state())))
    check("  snapshot round-trip keeps next round's premium",
          s2.export_premium("Llano", "wine") == s20.export_premium("Llano", "wine"))
    del s2.history[-1]["tariff_wall"]
    check("  a snapshot from before this rule gives the full premium",
          s2.export_premium("Llano", "wine") == EXPORT_MARKET_GAIN)

    # the projection: each firm's premium, and "never pays" at a 100% wall
    s100 = after_tariffs(lambda i, p, g: 1.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        s100.print_export_premiums()
    text = buf.getvalue()
    check("  print_export_premiums lists every firm; 100% wall = never pays",
          all(f in text for f in s100.firms) and "never pays" in text
          and s100.export_breakeven("F1") is None, text[:200])


def main():
    tests = [
        test_productivity_surge,
        test_populist_backlash,
        test_tariff_floor,
        test_mnc_tax_ledger,
        test_mnc_tax_starts_with_firms,
        test_award_reserve_currency,
        test_print_firm_rankings,
        test_populist_plus_selection,
        test_save_restore_phase4,
        test_print_results_phase4,
        test_inject_shock_missing_key_smoke,
        test_export_premium,
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
