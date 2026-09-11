"""
Student production calculator -- one static web page.

    sim.export_calculator()          # writes docs/index.html

Students pick their country from a row of buttons (one per country in play),
type where their workers and capital go, and see the output the engine will
compute. Nothing else: no welfare, no prices, no advice. Deciding what to
make and what to trade is the lesson; the arithmetic of L^0.7 * K^0.3 is not.

The numbers come from the LIVE simulation (sim.countries), not the
PHASE*_COUNTRIES constants, so re-export after any shock that changes
technology or endowments (inject_productivity_surge, inject_shock). Nothing
else in the course changes production, so there is no need to re-export
between rounds or at phase upgrades.

The page carries a few engine-computed check values and tests its own
arithmetic against them when it loads. If the page's formula and the engine's
ever drift apart, students see a "don't use these numbers" notice instead of
wrong numbers.

No server: publish docs/ with GitHub Pages (Settings -> Pages -> Deploy from
a branch -> master, /docs). Each commit + push republishes within a minute.
"""

import datetime
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "calculator_template.html")
DEFAULT_PATH = os.path.join(HERE, "docs", "index.html")
PLACEHOLDER = "__SIM_DATA__"


def _country_spec(sim, name):
    """One country's endowments and technology, in the engine's own terms."""
    cfg = sim.countries[name]
    if sim.phase == 1:
        # Ricardo: output = labor * productivity. Expressed as Cobb-Douglas
        # with a labor share of 1 and no capital, the page needs one formula.
        tech = {g: {"tfp": float(cfg["productivity"][g]),
                    "labor_share": 1.0, "capital_share": 0.0}
                for g in sim.goods}
        capital = None
    else:
        tech = {g: {"tfp": float(cfg["tech"][g]["tfp"]),
                    "labor_share": float(cfg["tech"][g]["labor_share"]),
                    "capital_share": float(cfg["tech"][g]["capital_share"])}
                for g in sim.goods}
        capital = float(cfg["capital"])
    return {"name": name, "labor": float(cfg["labor"]),
            "capital": capital, "tech": tech}


def _probe_allocations(sim, name):
    """
    Allocations the page re-computes on load. The first spreads both factors
    unevenly; the others leave a sector without one of its factors, which is
    where the engine's zero-output branch lives.
    """
    goods = sim.goods
    cfg = sim.countries[name]
    L = float(cfg["labor"])
    K = float(cfg.get("capital", 0.0)) if sim.phase != 1 else 0.0
    J = len(goods)
    w = [J - i for i in range(J)]                  # J, J-1, ..., 1
    spread_L = {g: L * w[i] / sum(w) for i, g in enumerate(goods)}
    spread_K = {g: K * w[J - 1 - i] / sum(w) for i, g in enumerate(goods)}
    first = {g: (L if i == 0 else 0.0) for i, g in enumerate(goods)}
    probes = [(spread_L, spread_K),
              (first, {g: (K if i == 0 else 0.0) for i, g in enumerate(goods)})]
    if sim.phase != 1:
        last_K = {g: (K if i == J - 1 else 0.0) for i, g in enumerate(goods)}
        probes.append((first, last_K))
    return probes


def _engine_output(sim, labor, capital):
    """What the engine's own _compute_production returns for one allocation."""
    if sim.phase == 1:
        dec = {"production": dict(labor)}
    else:
        dec = {"production": {"labor": dict(labor), "capital": dict(capital)}}
    return sim._compute_production({n: dec for n in sim.countries})


def calculator_data(sim):
    """Everything the page needs, as a JSON-ready dict."""
    names = sorted(sim.countries)          # alphabetical buttons: easy to find
    checks = []
    for name in names:
        for labor, capital in _probe_allocations(sim, name):
            out = _engine_output(sim, labor, capital)[name]
            checks.append({"country": name, "labor": labor,
                           "capital": capital,
                           "expected": {g: float(out[g]) for g in sim.goods}})
    today = datetime.date.today()
    return {
        "version": 1,
        "phase": sim.phase,
        "uses_capital": sim.phase != 1,
        "goods": list(sim.goods),
        "countries": [_country_spec(sim, n) for n in names],
        "checks": checks,
        "updated": f"{today:%B} {today.day}, {today.year}",
    }


def export_calculator(sim, path=None, verbose=True):
    """
    Write the calculator page (and a .nojekyll marker beside it, so GitHub
    Pages serves the file as-is). Returns the path written.
    """
    path = DEFAULT_PATH if path is None else path
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    if template.count(PLACEHOLDER) != 1:
        raise RuntimeError(f"{TEMPLATE} must contain {PLACEHOLDER} exactly once")

    data = calculator_data(sim)
    # "<" only ever appears inside JSON strings, where < is the same
    # character -- so nothing in the data can close the <script> block early.
    blob = json.dumps(data, indent=1).replace("<", "\\u003c")
    html = template.replace(PLACEHOLDER, blob)

    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    marker = os.path.join(folder, ".nojekyll")
    if not os.path.exists(marker):
        open(marker, "w").close()

    if verbose:
        names = ", ".join(c["name"] for c in data["countries"])
        print(f"\n  Calculator written: {path}")
        print(f"  {len(data['countries'])} countries: {names}")
        print("  Publish: commit and push docs/ -- GitHub Pages updates within "
              "a minute.")
        print("  Re-export only after a shock that changes technology or "
              "endowments.\n")
    return path
