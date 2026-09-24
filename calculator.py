"""
Student calculator and round form -- one static web page.

    sim.export_calculator()          # writes docs/index.html
    sim.export_calculator(inbox_url="https://www.dropbox.com/request/...")

Students pick the phase, then their country (or their firm). The calculator
shows what their workers and capital produce -- no welfare, no prices, no
advice: deciding what to make and trade is the lesson, the arithmetic of
L^0.7 * K^0.3 is not. Below it they fill in the rest of the round (tariffs,
trades, policies) and submit: the page saves one small decision file and
opens the class's Dropbox upload page, and play_round builds the round from
those files (classroom.py, "decision files").

The page carries every phase, so it never needs re-exporting as the term
moves on. The numbers come from the LIVE simulation where it has them
(sim.countries), so re-export after a shock that changes technology or
endowments (inject_productivity_surge, inject_shock). Re-exporting when
nothing changed leaves the file alone, and the upload link, once given, is
kept.

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
import re

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


def _era(sim):
    """One production era (Phase 1, or Phase 2 on) as the page needs it."""
    names = sorted(sim.countries)          # alphabetical buttons: easy to find
    checks = []
    for name in names:
        for labor, capital in _probe_allocations(sim, name):
            out = _engine_output(sim, labor, capital)[name]
            checks.append({"country": name, "labor": labor,
                           "capital": capital,
                           "expected": {g: float(out[g]) for g in sim.goods}})
    return {"uses_capital": sim.phase != 1, "goods": list(sim.goods),
            "countries": [_country_spec(sim, n) for n in names],
            "checks": checks}


def _other_era(sim):
    """
    The era the live simulation isn't in, built from the engine's constants
    for the same countries -- so one page serves the whole term.
    """
    import engine
    names = list(sim.countries)
    source = engine.PHASE2_COUNTRIES if sim.phase == 1 else engine.PHASE1_COUNTRIES
    if any(n not in source for n in names):
        return None                     # a custom country: only the live era
    if sim.phase == 1:
        other = engine.IPESimulation({n: engine.PHASE2_COUNTRIES[n] for n in names},
                                     engine.PHASE2_GOODS, phase=2)
    else:
        other = engine.IPESimulation({n: engine.PHASE1_COUNTRIES[n] for n in names},
                                     engine.PHASE1_GOODS, phase=1)
    return _era(other)


def _firms(sim):
    """The roster students submit for: from the game once firms exist."""
    import engine
    if sim.firms:
        config = sim.firm_config
        owners = {f: list(sim.firms[f].get("owners", [])) for f in sim.firms}
    else:
        try:
            config = engine.build_firm_roster(list(sim.countries), verbose=False)
        except ValueError:
            config = {}
        owners = {}
    return [{"id": f, "variety": c["variety"], "industry": c["industry"],
             "max_scale": float(c["max_scale"]), "owners": owners.get(f, [])}
            for f, c in config.items()]


def calculator_data(sim, inbox_url=""):
    """Everything the page needs, as a JSON-ready dict."""
    import engine
    live = _era(sim)
    other = _other_era(sim)
    eras = {("1" if sim.phase == 1 else "2"): live}
    if other is not None:
        eras["2" if sim.phase == 1 else "1"] = other
    today = datetime.date.today()
    return {
        "version": 2,
        # the live era, as before
        "uses_capital": live["uses_capital"],
        "goods": live["goods"],
        "countries": live["countries"],
        "checks": live["checks"],
        # every phase, so the page never needs re-exporting as the term moves on
        "eras": eras,
        "firms": _firms(sim),
        "rules": {
            "tariff_max": 100,
            "compensation_max": engine.COMPENSATION_MAX_SHARE * 100,
            "mnc_tax_min": engine.MNC_TAX_MIN * 100,
            "mnc_tax_max": engine.MNC_TAX_MAX * 100,
            "money_growth": [g * 100 for g in engine.PHASE5_MONEY_GROWTH_CHOICES],
        },
        "inbox_url": inbox_url or "",
        "updated": f"{today:%B} {today.day}, {today.year}",
    }


def _page_data(path):
    """The data block of an existing page, or None."""
    try:
        with open(path, encoding="utf-8") as f:
            html = f.read()
    except OSError:
        return None
    m = re.search(r'<script type="application/json" id="sim-data">(.*?)</script>',
                  html, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def export_calculator(sim, path=None, verbose=True, inbox_url=None):
    """
    Write the calculator page (and a .nojekyll marker beside it, so GitHub
    Pages serves the file as-is). Returns the path written.

    inbox_url : the Dropbox file request students upload decision files to.
                Given once, it is kept by later exports. If nothing on the
                page would change, the file is left alone -- no commit needed.
    """
    path = DEFAULT_PATH if path is None else path
    with open(TEMPLATE, encoding="utf-8") as f:
        template = f.read()
    if template.count(PLACEHOLDER) != 1:
        raise RuntimeError(f"{TEMPLATE} must contain {PLACEHOLDER} exactly once")

    old = _page_data(path)
    if inbox_url is None:
        inbox_url = (old or {}).get("inbox_url", "")
    data = calculator_data(sim, inbox_url)
    if old and old.get("updated"):
        data["updated"] = old["updated"]          # unchanged pages keep their date

    def render(d):
        # "<" only ever appears inside JSON strings, where \u003c is the same
        # character -- so nothing in the data can close the <script> block early.
        blob = json.dumps(d, indent=1).replace("<", "\\u003c")
        return template.replace(PLACEHOLDER, blob)

    html = render(data)
    try:
        with open(path, encoding="utf-8") as f:
            unchanged = f.read() == html
    except OSError:
        unchanged = False
    if unchanged:
        if verbose:
            print(f"\n  Calculator unchanged: {path} (nothing to publish)\n")
        return path
    today = datetime.date.today()
    data["updated"] = f"{today:%B} {today.day}, {today.year}"
    html = render(data)

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
        if not data["inbox_url"]:
            print("  No upload link yet: pass inbox_url= once so Submit opens "
                  "your Dropbox file request.")
        print("  Re-export only after a shock that changes technology or "
              "endowments.\n")
    return path
