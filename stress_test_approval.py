"""
Stress test the political-approval ledger.

The point of approval is that protection has a political payoff, so the
Block 4 thesis -- "protection can be terrible for the country and still win
the politics" -- is playable rather than merely asserted. These checks pin
down both halves: protection must still COST welfare, and must still BUY
approval, and a country that ignores its losers must eventually lose office.
"""
import contextlib
import io
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

from engine import (
    IPESimulation,
    PHASE1_COUNTRIES, PHASE1_GOODS,
    PHASE2_COUNTRIES, PHASE2_GOODS,
    APPROVAL_START, APPROVAL_CRISIS_FLOOR, APPROVAL_CRISIS_ROUNDS,
    APPROVAL_COMPENSATION, COMPENSATION_MAX_SHARE, COMPENSATION_DEADWEIGHT,
)

PASS, FAIL = [], []
C = ["Bosque", "Llano", "Sabine", "Trinity"]
PROD = {"Bosque": {"cloth": 50, "wine": 10}, "Llano": {"cloth": 20, "wine": 80},
        "Sabine": {"cloth": 30, "wine": 70}, "Trinity": {"cloth": 90, "wine": 30}}
SWAPS = [("Bosque", "Llano", "cloth", 25, "wine", 30),
         ("Trinity", "Sabine", "cloth", 30, "wine", 35)]


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def play(tariff, rounds=20, side_payments=None, quiet=True, compensation=None):
    """Everyone runs the same flat tariff for `rounds` rounds."""
    sim = IPESimulation({c: PHASE1_COUNTRIES[c] for c in C},
                        PHASE1_GOODS, phase=1)
    fell = []
    buf = io.StringIO()
    with (contextlib.redirect_stdout(buf) if quiet
          else contextlib.nullcontext()):
        for _ in range(rounds):
            tar = {c: {p: {g: tariff for g in PHASE1_GOODS}
                       for p in C if p != c} for c in C}
            dec = {c: {"production": PROD[c], "tariffs": tar[c]} for c in C}
            for c, share in (compensation or {}).items():
                dec[c]["compensation"] = share
            sim.run_round(dec, SWAPS, side_payments=side_payments or [])
            fell += sim.history[-1].get("governments_fallen", [])
    return sim, fell


def final(sim):
    r = sim.history[-1]["results"]
    return ({n: r[n]["approval"]["approval"] for n in C},
            sum(r[n]["welfare"] for n in C))


def test_seeded_and_separate():
    print("\n[1] approval exists, and never touches welfare")
    sim = IPESimulation({c: PHASE1_COUNTRIES[c] for c in C},
                        PHASE1_GOODS, phase=1)
    check("  every country seeded at the start value",
          all(sim.countries[c]["approval"] == APPROVAL_START for c in C))

    # identical rounds, one with tariffs one without: welfare must differ in
    # the direction economics predicts, and approval must move independently
    open_sim, _ = play(0.0, rounds=1)
    prot_sim, _ = play(0.30, rounds=1)
    _, w_open = final(open_sim)
    _, w_prot = final(prot_sim)
    check("  protection still destroys welfare", w_prot < w_open,
          f"open {w_open:.1f} vs protected {w_prot:.1f}")
    r = open_sim.history[-1]["results"]["Bosque"]
    check("  approval is reported as its own ledger",
          isinstance(r.get("approval"), dict) and "protection" in r["approval"])
    check("  welfare is not the approval score",
          r["welfare"] != r["approval"]["approval"])


def test_protection_buys_approval():
    print("\n[2] protection costs welfare and buys approval")
    rows = []
    for t in (0.0, 0.10, 0.20, 0.40):
        sim, _ = play(t)
        appr, w = final(sim)
        rows.append((t, w, sum(appr.values()) / len(C)))
    welfares = [w for _, w, _ in rows]
    approvals = [a for _, _, a in rows]
    check("  welfare falls monotonically as tariffs rise",
          all(welfares[i] > welfares[i + 1] for i in range(len(welfares) - 1)),
          str([round(w, 1) for w in welfares]))
    check("  approval rises monotonically as tariffs rise",
          all(approvals[i] < approvals[i + 1] for i in range(len(approvals) - 1)),
          str([round(a, 1) for a in approvals]))


def test_openness_topples_a_government():
    print("\n[3] unmanaged openness eventually costs a government")
    _, fell_open = play(0.0)
    check("  free trade topples someone over a full term", bool(fell_open),
          "nobody fell under 0% tariffs")
    _, fell_prot = play(0.20)
    check("  moderate protection prevents it", not fell_prot,
          f"fell anyway: {fell_prot}")


def test_backlash_is_endogenous():
    print("\n[4] the fall imposes protection, without the instructor")
    sim, fell = play(0.0)
    check("  a government fell", bool(fell))
    if fell:
        victim = fell[0]
        check("  the fall is recorded on the round",
              any(victim in rd.get("governments_fallen", [])
                  for rd in sim.history))
        check("  a tariff floor was imposed on the victim",
              sim.countries[victim].get("tariff_floor", 0) > 0,
              str(sim.countries[victim].get("tariff_floor")))
        check("  the new government starts fresh",
              sim.countries[victim]["approval"] > APPROVAL_CRISIS_FLOOR)


def test_compensation_helps():
    print("\n[5] compensating the losers buys political peace")
    plain, _ = play(0.0, rounds=8)
    comp, _ = play(0.0, rounds=8,
                   side_payments=[("Trinity", "Sabine", "cloth", 10)])
    a_plain, _ = final(plain)
    a_comp, _ = final(comp)
    check("  side payments raise the recipient's approval",
          a_comp["Sabine"] > a_plain["Sabine"],
          f"plain {a_plain['Sabine']:.1f} vs compensated {a_comp['Sabine']:.1f}")


def test_paying_your_own_losers():
    print("\n[5b] a country can compensate its own losers")
    plain, _ = play(0.0, rounds=1)
    comp, _ = play(0.0, rounds=1, compensation={"Sabine": 0.10})
    a_plain, a_comp = final(plain)[0], final(comp)[0]
    got = a_comp["Sabine"] - a_plain["Sabine"]
    check("  10% compensation buys about 2.5 approval",
          abs(got - APPROVAL_COMPENSATION * 0.10) < 0.2, f"got {got:+.2f}")
    w_plain = plain.history[-1]["results"]["Sabine"]["welfare"]
    w_comp = comp.history[-1]["results"]["Sabine"]["welfare"]
    want = 1 - COMPENSATION_DEADWEIGHT * 0.10
    check("  ...and costs 2% of welfare (the deadweight on the transfer)",
          abs(w_comp / w_plain - want) < 1e-9, f"ratio {w_comp / w_plain:.4f}")
    check("  recorded on the round",
          comp.history[-1]["results"]["Sabine"]["compensation"]["share"] == 0.10)
    # ...and it is cheaper per approval point than closing the border
    tariff, _ = play(0.20, rounds=1)
    a_tar = final(tariff)[0]["Sabine"] - a_plain["Sabine"]
    w_tar = tariff.history[-1]["results"]["Sabine"]["welfare"]
    check("  compensation costs less welfare per approval point than a 20% tariff",
          (w_plain - w_comp) / max(got, 1e-9) < (w_plain - w_tar) / max(a_tar, 1e-9),
          f"compensation {got:+.2f} approval for {w_plain - w_comp:.2f} welfare; "
          f"tariff {a_tar:+.2f} for {w_plain - w_tar:.2f}")


def test_compensation_cap_is_enforced():
    print("\n[5c] the compensation cap is enforced")
    for bad in (COMPENSATION_MAX_SHARE + 0.05, -0.05):
        try:
            play(0.0, rounds=1, compensation={"Sabine": bad})
            check(f"  {bad:+.2f} is rejected", False, "no error")
        except ValueError as e:
            check(f"  {bad:+.2f} is rejected", "compensation" in str(e), str(e)[:80])


def test_side_payments_are_net_and_move_goods():
    print("\n[5d] side payments: goods move from Phase 1, and swaps buy nothing")
    plain, _ = play(0.0, rounds=1)
    swap, _ = play(0.0, rounds=1, side_payments=[("Trinity", "Sabine", "cloth", 10),
                                                 ("Sabine", "Trinity", "cloth", 10)])
    a_plain, a_swap = final(plain)[0], final(swap)[0]
    check("  swapping the same goods back and forth buys no approval",
          abs(a_swap["Sabine"] - a_plain["Sabine"]) < 1e-9
          and abs(a_swap["Trinity"] - a_plain["Trinity"]) < 1e-9,
          f"Sabine {a_swap['Sabine']:.2f} vs {a_plain['Sabine']:.2f}")
    paid, _ = play(0.0, rounds=1, side_payments=[("Trinity", "Sabine", "cloth", 10)])
    a_paid = final(paid)[0]
    check("  a one-way payment still does", a_paid["Sabine"] > a_plain["Sabine"] + 1.0)
    r = paid.history[-1]
    moved = (r["results"]["Sabine"]["consumption"]["cloth"]
             - plain.history[-1]["results"]["Sabine"]["consumption"]["cloth"])
    check("  and the goods actually move before Phase 7", abs(moved - 10) < 1e-6,
          f"cloth moved {moved:.2f}")
    check("  the transfer is logged",
          any("side payment" in l for l in r.get("side_payment_log", [])))


def test_survives_upgrade_and_restore():
    print("\n[6] approval survives phase upgrades and save/restore")
    sim, _ = play(0.30, rounds=4)
    before = {c: sim.countries[c]["approval"] for c in C}
    with contextlib.redirect_stdout(io.StringIO()):
        sim.upgrade_to_phase2({c: PHASE2_COUNTRIES[c] for c in C}, PHASE2_GOODS)
    after = {c: sim.countries[c]["approval"] for c in C}
    check("  carried across the Phase 2 upgrade", before == after,
          f"{before} -> {after}")

    state = sim.get_state()
    with contextlib.redirect_stdout(io.StringIO()):
        fresh = IPESimulation.from_state(state)
    check("  survives save/restore",
          {c: fresh.countries[c]["approval"] for c in C} == after,
          str({c: fresh.countries[c].get("approval") for c in C}))


def test_bounds():
    print("\n[7] approval stays inside 0-100")
    for t in (0.0, 0.9):
        sim, _ = play(t, rounds=25)
        vals = [sim.countries[c]["approval"] for c in C]
        check(f"  bounded at tariff {t:.0%}",
              all(0.0 <= v <= 100.0 for v in vals),
              str([round(v, 1) for v in vals]))


def main():
    for t in [test_seeded_and_separate, test_protection_buys_approval,
              test_paying_your_own_losers, test_compensation_cap_is_enforced,
              test_side_payments_are_net_and_move_goods,
              test_openness_topples_a_government, test_backlash_is_endogenous,
              test_compensation_helps, test_survives_upgrade_and_restore,
              test_bounds]:
        try:
            t()
        except Exception:
            print(f"  EXCEPTION in {t.__name__}:")
            traceback.print_exc()
            FAIL.append((t.__name__, "exception"))
    print(f"\n{'=' * 60}")
    print(f"  PASSED: {len(PASS)}")
    print(f"  FAILED: {len(FAIL)}")
    if FAIL:
        for name, detail in FAIL:
            print(f"   - {name}: {detail}")
        sys.exit(1)
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
