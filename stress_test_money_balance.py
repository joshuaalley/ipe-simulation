"""
Balance test for the money and debt rules (Phases 5-6).

The old rules had two always-wins: 'peg + capital controls + 0% growth' was
never worse than anything else (printing, open capital and floating bought
nothing), and 'borrow the maximum, then default' beat never borrowing by a
third. This plays six quite different trading worlds through the money era and
fails if either kind of always-win comes back in any of them:

  class-like     Round 5 of Fall 2026 (few, large swaps)
  heavy trade    big swaps; imports are a large share of consumption
  light trade    two small trades a round -- printing's import cost barely bites
  no trade       autarky -- only the welfare drag keeps printing honest
  specialized    everyone specializes hard and lives on imports
  six countries  all six, a denser network

The constants in engine.py were chosen from the middle of the range that
passes all six, so a real class's trade pattern shouldn't need a re-tune. If
you change STIMULUS_PER_POINT, WEAK_FX_IMPORT_COST, WEAK_FX_WELFARE_DRAG,
CONTROLS_FIRM_CUT, PEG_PEG_FRICTION, BORROW_CAP_SHARE or DEBT_DEFAULT_COST,
re-run it. Slower than the other suites (about a thousand short simulations,
~20 seconds).
"""
import contextlib
import copy
import io
import itertools
import sys
import traceback

import matplotlib
matplotlib.use("Agg")

from engine import (IPESimulation, PHASE2_COUNTRIES, PHASE2_GOODS,
                    build_firm_roster)

PASS, FAIL = [], []
G = PHASE2_GOODS


def alloc(L, K):
    return {"labor": dict(zip(G, L)), "capital": dict(zip(G, K))}


# Each country tilts toward what it's good at...
TILTED = {
    "Bosque":  alloc((40, 15, 5), (12, 7, 6)),
    "Llano":   alloc((15, 70, 15), (10, 50, 20)),
    "Sabine":  alloc((70, 20, 10), (20, 8, 7)),
    "Trinity": alloc((30, 25, 65), (30, 30, 140)),
    "Brazos":  alloc((50, 50, 50), (50, 50, 50)),
    "Pecos":   alloc((10, 15, 25), (20, 30, 70)),
}
# ...or goes all in on it.
SPECIALIZED = {
    "Bosque":  alloc((55, 3, 2), (20, 3, 2)),
    "Llano":   alloc((5, 90, 5), (5, 70, 5)),
    "Sabine":  alloc((90, 5, 5), (30, 3, 2)),
    "Trinity": alloc((10, 10, 100), (10, 10, 180)),
}
FOUR = ["Bosque", "Llano", "Sabine", "Trinity"]
SIX = ["Bosque", "Brazos", "Llano", "Pecos", "Sabine", "Trinity"]

WORLDS = {
    "class-like": (FOUR, TILTED, [
        ("Bosque", "Trinity", "cloth", 27, "machinery", 9),
        ("Trinity", "Llano", "machinery", 5, "wine", 10),
        ("Llano", "Sabine", "wine", 15, "cloth", 30),
        ("Sabine", "Trinity", "cloth", 10, "machinery", 4)]),
    "heavy trade": (FOUR, TILTED, [
        ("Bosque", "Trinity", "cloth", 25, "machinery", 9),
        ("Sabine", "Trinity", "cloth", 30, "machinery", 11),
        ("Trinity", "Llano", "machinery", 20, "wine", 40),
        ("Llano", "Sabine", "wine", 25, "cloth", 15),
        ("Bosque", "Llano", "cloth", 5, "wine", 6)]),
    "light trade": (FOUR, TILTED, [
        ("Bosque", "Trinity", "cloth", 6, "machinery", 2),
        ("Llano", "Sabine", "wine", 5, "cloth", 5)]),
    "no trade": (FOUR, TILTED, []),
    "specialized": (FOUR, SPECIALIZED, [
        ("Bosque", "Trinity", "cloth", 20, "machinery", 8),
        ("Bosque", "Llano", "cloth", 15, "wine", 15),
        ("Sabine", "Trinity", "cloth", 30, "machinery", 12),
        ("Sabine", "Llano", "cloth", 25, "wine", 25),
        ("Trinity", "Llano", "machinery", 25, "wine", 40)]),
    "six countries": (SIX, TILTED, [
        ("Bosque", "Trinity", "cloth", 15, "machinery", 5),
        ("Sabine", "Pecos", "cloth", 20, "machinery", 8),
        ("Llano", "Brazos", "wine", 20, "cloth", 15),
        ("Trinity", "Llano", "machinery", 10, "wine", 20),
        ("Pecos", "Brazos", "machinery", 10, "wine", 15),
        ("Brazos", "Sabine", "machinery", 10, "cloth", 12)]),
}
TRADING = [w for w in WORLDS if w != "no trade"]

STRATS = list(itertools.product(("peg", "float"), (False, True),
                                (0.0, 0.02, 0.05, 0.10)))
OLD_WINNER = ("peg", True, 0.0)
PRINT_FOREVER = ("float", False, 0.10)
BASELINES = (("peg", False, 0.0), ("float", False, 0.0))
ROUNDS, ATTACK_BEFORE = 12, 3             # R15-R26; the R17 attack
BIG = 1e9


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append((name, detail))
        print(f"  FAIL  {name} -- {detail}")


def label(s):
    return f"{s[0]}/{'controls' if s[1] else 'open'}/{s[2]:.0%}"


def md_of(s):
    return {"fx_regime": s[0], "capital_controls": s[1], "money_supply_growth": s[2]}


class World:
    def __init__(self, name):
        self.name = name
        self.countries, allocs, self.trades = WORLDS[name]
        self.dec = {c: {"production": allocs[c], "tariffs": {}} for c in self.countries}
        n_firms = 11 if len(self.countries) == 4 else None
        with contextlib.redirect_stdout(io.StringIO()):
            sim = IPESimulation({c: PHASE2_COUNTRIES[c] for c in self.countries},
                                PHASE2_GOODS, phase=2)
            sim.upgrade_to_phase3(build_firm_roster(self.countries, n_firms=n_firms,
                                                    verbose=False))
            self.fd = {f: {"scale": 30, "relocate_to": None, "export": False}
                       for f in sim.firms}
            sim.run_round(self.dec, self.trades, firm_decisions=self.fd)
            sim.phase = 4
            sim.run_round(self.dec, self.trades, firm_decisions=self.fd)
            sim.award_reserve_currency()
            sim.upgrade_to_phase5()
        self.p5 = sim
        self.reserve = sim.reserve_currency_holder

    def money_run(self, profile):
        """Everyone holds profile[c] for twelve rounds; the R17 attack lands on
        whoever is most exposed (if anyone clearly is)."""
        sim = copy.deepcopy(self.p5)
        cum = {c: 0.0 for c in self.countries}
        with contextlib.redirect_stdout(io.StringIO()):
            for k in range(ROUNDS):
                md = {c: md_of(profile[c]) for c in self.countries}
                if k == ATTACK_BEFORE:
                    for c in self.countries:
                        sim._mon(c).update(md[c])
                    sim.trigger_speculative_attack(show=False)
                r = sim.run_round(self.dec, self.trades, firm_decisions=self.fd,
                                  monetary_decisions=md)
                bad = [l for l in r["trade_log"] if "FAILED" in l]
                assert not bad, (self.name, bad)
                for c in self.countries:
                    cum[c] += r["results"][c]["welfare"]
        return cum

    def best_responses(self):
        """One row per (baseline, country): every strategy ranked for that
        country while everyone else holds the baseline."""
        rows = []
        for base in BASELINES:
            for c in self.countries:
                score = {s: self.money_run({x: (s if x == c else base)
                                            for x in self.countries})[c]
                         for s in STRATS}
                ranked = sorted(score, key=score.get, reverse=True)
                best_ctrl = next(s for s in ranked if s[1])
                rows.append({"baseline": base, "country": c, "best": ranked[0],
                             "print_rank": ranked.index(PRINT_FOREVER) + 1,
                             "ctrl_gap": 1 - score[best_ctrl] / score[ranked[0]]})
        return rows

    def debt_run(self, who, policy, shock_before=None, n=6):
        sim = copy.deepcopy(self.p5)
        md = {c: md_of(("float", False, 0.0)) for c in self.countries}
        total, rows = 0.0, []
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run_round(self.dec, self.trades, firm_decisions=self.fd,
                          monetary_decisions=md)
            sim.upgrade_to_phase6()
            sim.final_round = sim.round_num + n
            for k in range(n):
                if shock_before is not None and k == shock_before:
                    sim.inject_capital_flight(who, severity=0.6)
                dd = policy(k, sim)
                r = sim.run_round(self.dec, self.trades, firm_decisions=self.fd,
                                  monetary_decisions=md,
                                  debt_decisions={who: dd} if dd else {})
                total += r["results"][who]["welfare"]
                rows.append(r["results"][who]["debt"])
                # Hold approval still. Borrowing lifts welfare, which lifts the
                # prosperity term, which can stave off a populist backlash --
                # a real channel, but a political one. These checks are about
                # whether the DEBT mechanics pay for themselves.
                for c in self.countries:
                    sim.countries[c]["approval"] = 50.0
                    sim.countries[c]["low_approval_rounds"] = 0
        return total, rows


WORLD_OBJ = {}


def world(name):
    if name not in WORLD_OBJ:
        WORLD_OBJ[name] = World(name)
    return WORLD_OBJ[name]


def test_no_monetary_strategy_always_wins():
    print("\n[1] no fixed monetary strategy wins everywhere (six worlds)")
    every = []
    for name in WORLDS:
        rows = world(name).best_responses()
        every += [(name, r) for r in rows]
        bests = sorted({label(r["best"]) for r in rows})
        print(f"      {name:13s} best choices: {', '.join(bests)}")
        check(f"  {name}: printing 10% every round is never best",
              all(r["best"] != PRINT_FOREVER for r in rows),
              [r["country"] for r in rows if r["best"] == PRINT_FOREVER])
        check(f"  {name}: the old always-win (peg + controls + 0%) is never best",
              all(r["best"] != OLD_WINNER for r in rows))
        check(f"  {name}: more than one best choice across countries",
              len(bests) >= 2, str(bests))
        worst = max(r["ctrl_gap"] for r in rows)
        check(f"  {name}: capital controls stay a live option (within 12% of best)",
              worst < 0.12, f"worst gap {worst * 100:.1f}%")
    bests = [r["best"] for _, r in every]
    check("  pegging and floating are both someone's best choice",
          {s[0] for s in bests} == {"peg", "float"})
    check("  printing and following the anchor are both someone's best choice",
          any(s[2] > 0 for s in bests) and any(s[2] == 0 for s in bests))


def debtors(w):
    return [c for c in ("Bosque", "Sabine", "Trinity")
            if c in w.countries and c != w.reserve][:2]


def test_debt_has_no_free_lunch():
    print("\n[2] borrowing is a trade-off, not a money machine (five trading worlds)")
    for name in TRADING:
        w = world(name)
        for who in debtors(w):
            stock = lambda s, who=who: s.countries[who]["debt_stock"]
            never, _ = w.debt_run(who, lambda k, s: {})
            serial, _ = w.debt_run(who, lambda k, s: {"default": True}
                                   if stock(s) > 1e-9 else {"borrow": BIG})
            every, _ = w.debt_run(who, lambda k, s: {"borrow": BIG})
            last, rows = w.debt_run(who, lambda k, s: {"borrow": BIG} if k == 5 else {})
            check(f"  {name}, {who}: serial default loses to never borrowing",
                  serial < never, f"{serial:.1f} vs {never:.1f}")
            check(f"  {name}, {who}: borrowing the maximum every round loses",
                  every < never, f"{every:.1f} vs {never:.1f}")
            check(f"  {name}, {who}: borrowing in the final round is refused",
                  rows[-1]["borrow"] == 0.0 and abs(last - never) < 1e-9)


def test_default_pays_only_after_a_devaluation():
    print("\n[3] default is punished at par, and pays once the currency collapses")
    for name in TRADING:
        w = world(name)
        for who in debtors(w):
            never, _ = w.debt_run(who, lambda k, s: {})
            out = {}
            for nm, second in (("repay", {"repay": BIG}), ("default", {"default": True})):
                for shock in (None, 1):
                    out[(nm, shock)], _ = w.debt_run(
                        who, lambda k, s, second=second: {"borrow": BIG} if k == 0
                        else (second if k == 1 else {}), shock_before=shock)
            at_par = (out[("repay", None)] - out[("default", None)]) / never
            after = (out[("default", 1)] - out[("repay", 1)]) / never
            check(f"  {name}, {who}: at par, repaying clearly beats defaulting",
                  at_par > 0.005, f"margin {at_par * 100:.2f}%")
            check(f"  {name}, {who}: after a 40% devaluation, defaulting beats repaying",
                  after > 0, f"margin {after * 100:.2f}%")


def main():
    for t in [test_no_monetary_strategy_always_wins, test_debt_has_no_free_lunch,
              test_default_pays_only_after_a_devaluation]:
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
