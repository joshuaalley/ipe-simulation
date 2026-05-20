"""
IPE Simulation Engine v1
========================
Phase 1: Ricardian Comparative Advantage (1 factor, 2+ goods)
Phase 2: Heckscher-Ohlin (2 factors, 3+ goods, distributional politics)

Turn-based classroom simulation. Instructor enters student decisions,
engine computes outcomes. Designed to run in a Jupyter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from textwrap import dedent


# ═══════════════════════════════════════════════════════════════════
#  DEFAULT COUNTRY CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════

# Phase 1: Ricardian — each country has labor + productivity per good
# Designed for asymmetry: different sizes, different comparative advantages

PHASE1_COUNTRIES = {
    "Brazos": {
        "labor": 150,
        "productivity": {"cloth": 2.0, "wine": 2.0},
        "description": "Large, balanced economy. No clear comparative advantage.",
    },
    "Bosque": {
        "labor": 60,
        "productivity": {"cloth": 4.0, "wine": 1.0},
        "description": "Small but highly productive in cloth. Textiles powerhouse.",
    },
    "Llano": {
        "labor": 100,
        "productivity": {"cloth": 1.0, "wine": 3.0},
        "description": "Medium economy with strong wine sector.",
    },
    "Trinity": {
        "labor": 120,
        "productivity": {"cloth": 3.0, "wine": 1.5},
        "description": "Large economy, leans toward cloth but diversified.",
    },
    "Pecos": {
        "labor": 50,
        "productivity": {"cloth": 3.5, "wine": 3.5},
        "description": "Tiny but efficient. Absolute advantage in both goods, small market.",
    },
    "Sabine": {
        "labor": 100,
        "productivity": {"cloth": 1.0, "wine": 1.5},
        "description": "Medium economy, low productivity. Comparative advantage in wine.",
    },
}

PHASE1_GOODS = ["cloth", "wine"]

# Phase 2: Heckscher-Ohlin — add capital, factor intensities, third good
# Cloth: labor-intensive | Wine: moderate | Machinery: capital-intensive

PHASE2_COUNTRIES = {
    "Brazos": {
        "labor": 150, "capital": 150,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.0, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.0, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Balanced factor endowments. Can compete in any sector.",
    },
    "Bosque": {
        "labor": 60, "capital": 25,
        "tech": {
            "cloth":     {"tfp": 1.2, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 0.8, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 0.7, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Labor-abundant, capital-scarce. Strong in textiles.",
    },
    "Llano": {
        "labor": 100, "capital": 80,
        "tech": {
            "cloth":     {"tfp": 0.8, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.3, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 0.9, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Moderate endowments. TFP advantage in wine.",
    },
    "Trinity": {
        "labor": 120, "capital": 200,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 0.9, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.2, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Capital-abundant industrial power. Machinery is natural strength.",
    },
    "Pecos": {
        "labor": 50, "capital": 120,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.0, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.1, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Small, very capital-rich. High-tech potential but limited labor.",
    },
    "Sabine": {
        "labor": 100, "capital": 35,
        "tech": {
            "cloth":     {"tfp": 1.1, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.0, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 0.6, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Labor-abundant, capital-poor. Struggles with heavy industry.",
    },
}

PHASE2_GOODS = ["cloth", "wine", "machinery"]


# ═══════════════════════════════════════════════════════════════════
#  PHASE 3+ FIRMS / MNCs
# ═══════════════════════════════════════════════════════════════════
# Each firm has: industry, variety, default host, productivity multiplier,
# unit cost (per unit of output produced, paid to host as factor income),
# max_scale (production cap per round),
# fixed_export_cost (Phase 4: cost in numeraire to access foreign markets).
#
# Productivity tiers map roughly to HIGH=1.3, MED=1.0, LOW=0.7.
# Pre-set so no single country is over-rewarded or over-punished.

PHASE3_FIRMS = {
    # Cloth
    "F1":  {"industry": "cloth",     "variety": "Cloth-A", "default_host": "Bosque",
            "productivity": 1.3, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F2":  {"industry": "cloth",     "variety": "Cloth-B", "default_host": "Llano",
            "productivity": 1.0, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F3":  {"industry": "cloth",     "variety": "Cloth-C", "default_host": "Sabine",
            "productivity": 0.7, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    # Wine
    "F4":  {"industry": "wine",      "variety": "Wine-A",  "default_host": "Llano",
            "productivity": 1.3, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F5":  {"industry": "wine",      "variety": "Wine-B",  "default_host": "Sabine",
            "productivity": 1.0, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F6":  {"industry": "wine",      "variety": "Wine-C",  "default_host": "Brazos",
            "productivity": 1.0, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    # Machinery
    "F7":  {"industry": "machinery", "variety": "Mach-A",  "default_host": "Trinity",
            "productivity": 1.3, "unit_cost": 1.0, "max_scale": 40,
            "fixed_export_cost": 12.0},
    "F8":  {"industry": "machinery", "variety": "Mach-B",  "default_host": "Pecos",
            "productivity": 1.0, "unit_cost": 1.0, "max_scale": 40,
            "fixed_export_cost": 12.0},
    "F9":  {"industry": "machinery", "variety": "Mach-C",  "default_host": "Brazos",
            "productivity": 0.7, "unit_cost": 1.0, "max_scale": 40,
            "fixed_export_cost": 12.0},
    "F10": {"industry": "machinery", "variety": "Mach-D",  "default_host": "Trinity",
            "productivity": 1.0, "unit_cost": 1.0, "max_scale": 40,
            "fixed_export_cost": 12.0},
    # Slack firms (use only for class > 10 students)
    "F11": {"industry": "cloth",     "variety": "Cloth-D", "default_host": "Trinity",
            "productivity": 0.7, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F12": {"industry": "wine",      "variety": "Wine-D",  "default_host": "Bosque",
            "productivity": 0.7, "unit_cost": 0.6, "max_scale": 40,
            "fixed_export_cost": 8.0},
    "F13": {"industry": "machinery", "variety": "Mach-E",  "default_host": "Sabine",
            "productivity": 0.7, "unit_cost": 1.0, "max_scale": 40,
            "fixed_export_cost": 12.0},
}

# Reference per-unit prices firms receive (numeraire welfare units).
# Used only for firm profit accounting; inter-country goods trade is still barter.
WORLD_PRICES = {"cloth": 1.0, "wine": 1.0, "machinery": 1.5}

# CES elasticity within each industry (love-of-variety).
# rho closer to 0 = stronger variety preference; rho=1 = perfect substitutes.
VARIETY_RHO = 0.6


# ═══════════════════════════════════════════════════════════════════
#  PHASE 5 MONETARY & FX PARAMETERS
# ═══════════════════════════════════════════════════════════════════
# The trilemma: a country cannot simultaneously run a fixed exchange rate,
# open capital markets, and independent monetary policy without accumulating
# stress that eventually triggers a currency crisis.

# Discrete money-supply growth choices (only meaningful if independent_monetary).
PHASE5_MONEY_GROWTH_CHOICES = [0.0, 0.02, 0.05, 0.10]

# Graduated stress consequences:
WARNING_DEVALUATION = 0.90     # stress=1: currency drops 10%
WARNING_FRICTION_BUMP = 0.03   # warned country: +3% FX friction the following round
CRISIS_DEVALUATION = 0.70      # stress=2: currency drops 30%
CRISIS_WELFARE_HIT = 0.10      # stress=2: 10% welfare hit that round

# Cross-currency trade friction (stacks on top of tariffs):
BASE_FX_FRICTION = 0.02        # 2% baseline on cross-currency, non-union, non-reserve trade


# ═══════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class IPESimulation:
    """
    Turn-based international political economy simulation.

    Usage:
        sim = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)
        result = sim.run_round(decisions, trades)
        sim.print_results()
        sim.plot_welfare()
    """

    def __init__(self, countries: dict, goods: list, phase: int = 1):
        self.countries = deepcopy(countries)
        self.goods = list(goods)
        self.phase = phase
        self.history = []
        self.round_num = 0

        # Phase 3+ state (empty until upgrade_to_phase3 is called)
        self.firm_config = {}   # static firm parameters (immutable post-upgrade)
        self.firms = {}         # mutable per-firm state: host, cumulative_profit
        self.world_prices = dict(WORLD_PRICES)
        self.variety_rho = VARIETY_RHO

        # Phase 4+ political-economy state
        # Country attributes `tariff_floor` and `mnc_tax_rate` are looked up
        # via .get(..., 0.0) so countries don't need to pre-declare them;
        # set them via inject_populist_backlash() or inject_shock().
        self.mnc_tax_revenue = {}     # country -> cumulative MNC tax revenue
        self.reserve_currency_holder = None  # set by award_reserve_currency()

        # Phase 5+ monetary state.
        # Per-country monetary fields (currency, fx_regime, capital_controls,
        # independent_monetary, money_supply_growth, stress, depreciation_factor,
        # union_id, warning_active) are added by upgrade_to_phase5().
        # monetary_unions: name -> {"members": [...], "state": {shared monetary fields}}
        self.monetary_unions = {}

    # ── Core round logic ──────────────────────────────────────────

    def run_round(self, decisions: dict, trades: list,
                  firm_decisions: dict = None,
                  monetary_decisions: dict = None) -> dict:
        """
        Execute one round of the simulation.

        Parameters
        ----------
        decisions : dict
            Phase 1 format:
                {country: {"production": {"cloth": 60, "wine": 40},
                           "tariffs": {partner: {good: rate, ...}, ...}}}
            Phase 2+ format:
                {country: {"production": {"labor": {"cloth": 40, ...},
                                          "capital": {"cloth": 20, ...}},
                           "tariffs": {partner: {good: rate, ...}, ...}}}

            Tariffs are optional (default 0). Rates are fractions (0.1 = 10%).

        trades : list of tuples
            Each tuple: (exporter, importer, good_out, qty_out, good_in, qty_in)
            Trades are at the GOOD level. In Phase 3+, varieties move with
            their good proportionally (importer receives the exporter's
            variety mix).

        firm_decisions : dict, Phase 3+ only
            {firm_id: {"scale": int, "relocate_to": None or country,
                       "export": bool}}
            scale clamped to [0, max_scale]. export only matters in Phase 4+.

        monetary_decisions : dict, Phase 5+ only
            {country: {"fx_regime": "peg"|"managed"|"float",
                       "capital_controls": bool,
                       "independent_monetary": bool,
                       "money_supply_growth": one of PHASE5_MONEY_GROWTH_CHOICES}}
            Omitted countries keep their current monetary policy. Monetary
            decisions are applied BEFORE production/trade so this round's
            trades feel any crisis devaluation immediately.

        Returns
        -------
        dict with round results.
        """
        self.round_num += 1
        errors = self._validate_decisions(decisions)
        if errors:
            self.round_num -= 1
            raise ValueError("Decision validation failed:\n" + "\n".join(errors))

        if self.phase >= 3:
            if firm_decisions is None:
                firm_decisions = {
                    fid: {"scale": 0, "relocate_to": None, "export": False}
                    for fid in self.firms
                }
            fd_errors = self._validate_firm_decisions(firm_decisions)
            if fd_errors:
                self.round_num -= 1
                raise ValueError(
                    "Firm decision validation failed:\n" + "\n".join(fd_errors)
                )

        # Step 0: Monetary decisions + trilemma (Phase 5+), applied first so
        # crisis devaluations affect this round's trade friction and profits.
        monetary_events = {}
        if self.phase >= 5:
            if monetary_decisions is None:
                monetary_decisions = {}
            md_errors = self._validate_monetary_decisions(monetary_decisions)
            if md_errors:
                self.round_num -= 1
                raise ValueError(
                    "Monetary decision validation failed:\n" + "\n".join(md_errors)
                )
            monetary_events = self._apply_monetary_decisions(monetary_decisions)

        # Step 1: Country production (scalar totals)
        production = self._compute_production(decisions)

        # Step 1b: Firm production, added to host country's industry total
        firm_output = {}
        production_varieties = None
        if self.phase >= 3:
            firm_output = self._compute_firm_production(firm_decisions)
            for fid, out in firm_output.items():
                host = self.firms[fid]["host"]
                industry = self.firm_config[fid]["industry"]
                production[host][industry] = (
                    production[host].get(industry, 0.0) + out
                )
            # Per-variety breakdown for CES utility
            production_varieties = self._build_variety_bundles(
                production, firm_output
            )

        # Step 2: Initialize consumption (scalar totals) and varieties
        consumption = {c: dict(p) for c, p in production.items()}
        varieties = (
            deepcopy(production_varieties) if production_varieties else None
        )

        # Step 3: Execute trades. Varieties move proportionally with their good.
        trade_log, tariff_losses, trade_records = self._execute_trades(
            consumption, decisions, trades, varieties=varieties
        )

        # Step 4: Welfare. Phase 3+ uses variety-aware CES utility.
        results = {}
        for name in self.countries:
            if self.phase >= 3:
                welfare = self._utility(varieties[name])
                no_trade_welfare = self._utility(production_varieties[name])
            else:
                welfare = self._utility(consumption[name])
                no_trade_welfare = self._utility(production[name])
            # Phase 5+: a full currency crisis this round knocks welfare down.
            crisis_welfare_loss = 0.0
            if self.phase >= 5 and monetary_events.get(name, {}).get("crisis"):
                crisis_welfare_loss = welfare * CRISIS_WELFARE_HIT
                welfare -= crisis_welfare_loss

            if no_trade_welfare > 0:
                gains_pct = (welfare - no_trade_welfare) / no_trade_welfare * 100
            elif welfare > 0:
                gains_pct = float("inf")
            else:
                gains_pct = 0.0

            results[name] = {
                "production": production[name],
                "consumption": consumption[name],
                "welfare": welfare,
                "no_trade_welfare": no_trade_welfare,
                "gains_from_trade_pct": gains_pct,
                "tariff_losses": tariff_losses[name],
            }

            if self.phase >= 5:
                ev = monetary_events.get(name, {})
                results[name]["monetary"] = {
                    "currency": self.countries[name].get("currency"),
                    "fx_regime": self._mon(name).get("fx_regime"),
                    "capital_controls": self._mon(name).get("capital_controls"),
                    "independent_monetary": self._mon(name).get("independent_monetary"),
                    "money_supply_growth": self._mon(name).get("money_supply_growth"),
                    "stress": ev.get("stress", 0),
                    "depreciation_factor": ev.get("depreciation_factor", 1.0),
                    "warning": ev.get("warning", False),
                    "crisis": ev.get("crisis", False),
                    "crisis_welfare_loss": crisis_welfare_loss,
                    "union_id": self.countries[name].get("union_id"),
                }

            if self.phase >= 2:
                results[name]["factor_prices"] = self._compute_factor_prices(
                    name, decisions[name], production[name]
                )
            if self.phase >= 3:
                results[name]["consumption_varieties"] = varieties[name]
                results[name]["production_varieties"] = (
                    production_varieties[name]
                )

        # Step 5: Firm profits (Phase 3+); MNC tax (Phase 4+)
        firm_results = {}
        mnc_tax_this_round = {}
        if self.phase >= 3:
            firm_results, mnc_tax_this_round = self._compute_firm_profits(
                firm_decisions, firm_output
            )

        round_result = {
            "round": self.round_num,
            "phase": self.phase,
            "results": results,
            "trade_log": trade_log,
            "trades_executed": trade_records,
        }
        if self.phase >= 3:
            round_result["firms"] = firm_results
        if self.phase >= 4:
            round_result["mnc_tax_this_round"] = mnc_tax_this_round
            round_result["mnc_tax_cumulative"] = dict(self.mnc_tax_revenue)
        if self.phase >= 5:
            round_result["monetary_events"] = monetary_events
        self.history.append(round_result)
        return round_result

    # ── Production ────────────────────────────────────────────────

    def _compute_production(self, decisions):
        production = {}
        for name, config in self.countries.items():
            production[name] = {}
            dec = decisions[name]

            if self.phase == 1:
                for good in self.goods:
                    labor = dec["production"].get(good, 0)
                    productivity = config["productivity"][good]
                    production[name][good] = labor * productivity
            else:
                for good in self.goods:
                    L = dec["production"]["labor"].get(good, 0)
                    K = dec["production"]["capital"].get(good, 0)
                    tech = config["tech"][good]
                    if L > 0 and K > 0:
                        production[name][good] = (
                            tech["tfp"]
                            * (L ** tech["labor_share"])
                            * (K ** tech["capital_share"])
                        )
                    elif L > 0 and tech["capital_share"] == 0:
                        production[name][good] = tech["tfp"] * (L ** tech["labor_share"])
                    else:
                        production[name][good] = 0.0

        return production

    # ── Firm production (Phase 3+) ────────────────────────────────

    def _compute_firm_production(self, firm_decisions):
        """
        Return {firm_id: output_qty}. Relocating firms produce 0 this round
        and update their host. Output = scale * productivity, scale clamped
        to [0, max_scale].
        """
        output = {}
        for fid, fcfg in self.firm_config.items():
            dec = firm_decisions.get(fid, {})
            relocate_to = dec.get("relocate_to")
            if relocate_to is not None:
                # Relocation: change host, produce nothing this round
                self.firms[fid]["host"] = relocate_to
                output[fid] = 0.0
                continue
            scale = max(0.0, min(dec.get("scale", 0), fcfg["max_scale"]))
            output[fid] = scale * fcfg["productivity"]
        return output

    def _build_variety_bundles(self, production, firm_output):
        """
        Construct {country: {good: {variety_id: qty}}}.
        Each country has a 'country-generic' variety per good carrying its
        domestic (H-O) output; each hosted MNC adds its own variety.
        """
        bundles = {name: {g: {} for g in self.goods} for name in self.countries}

        # MNC varieties first
        firm_share = {}  # (host, good) -> total firm output
        for fid, fcfg in self.firm_config.items():
            host = self.firms[fid]["host"]
            good = fcfg["industry"]
            out = firm_output.get(fid, 0.0)
            if out > 0:
                bundles[host][good][fcfg["variety"]] = out
                firm_share[(host, good)] = firm_share.get((host, good), 0.0) + out

        # Country-generic = total production minus MNC contributions
        for name in self.countries:
            for good in self.goods:
                total = production[name].get(good, 0.0)
                mnc_total = firm_share.get((name, good), 0.0)
                country_only = total - mnc_total
                if country_only > 0.001:
                    bundles[name][good][f"{good}-{name}"] = country_only

        return bundles

    def _compute_firm_profits(self, firm_decisions, firm_output):
        """
        Profit per firm = revenue - operating_cost - fixed_export_cost - mnc_tax.
        revenue   = output * world_price[industry]
        op_cost   = scale * unit_cost
        fixed     = fixed_export_cost iff dec.export and Phase >= 4
        mnc_tax   = revenue * host's mnc_tax_rate (Phase 4+; ledger only,
                    does NOT enter country welfare/utility)

        Cumulative profit ledger on self.firms[fid] is updated.
        Tax collected is also added to self.mnc_tax_revenue[host] (cumulative).
        Returns (profits dict, tax_this_round dict).
        """
        profits = {}
        tax_this_round = {n: 0.0 for n in self.countries}
        for fid, fcfg in self.firm_config.items():
            dec = firm_decisions.get(fid, {})
            relocating = dec.get("relocate_to") is not None
            scale = (
                0.0 if relocating
                else max(0.0, min(dec.get("scale", 0), fcfg["max_scale"]))
            )
            output = firm_output.get(fid, 0.0)
            revenue = output * self.world_prices[fcfg["industry"]]
            op_cost = scale * fcfg["unit_cost"]
            fixed_cost = 0.0
            if self.phase >= 4 and dec.get("export", False):
                fixed_cost = fcfg["fixed_export_cost"]
            host = self.firms[fid]["host"]
            mnc_tax = 0.0
            if self.phase >= 4:
                rate = self.countries[host].get("mnc_tax_rate", 0.0)
                mnc_tax = revenue * rate
                tax_this_round[host] += mnc_tax
            profit_nominal = revenue - op_cost - fixed_cost - mnc_tax
            # Phase 5+: profits accrue in the HOST currency. Convert to real
            # (numeraire) terms via the host's depreciation factor — a firm in
            # a depreciating/crisis-hit host keeps less real value.
            dep = 1.0
            if self.phase >= 5:
                dep = self._mon(host).get("depreciation_factor", 1.0)
            profit_real = profit_nominal * dep
            self.firms[fid]["cumulative_profit"] += profit_real
            profits[fid] = {
                "host": host,
                "industry": fcfg["industry"],
                "variety": fcfg["variety"],
                "scale": scale,
                "output": output,
                "revenue": revenue,
                "operating_cost": op_cost,
                "fixed_cost": fixed_cost,
                "mnc_tax": mnc_tax,
                "profit_nominal": profit_nominal,
                "depreciation_factor": dep,
                "profit": profit_real,
                "cumulative_profit": self.firms[fid]["cumulative_profit"],
                "relocated": relocating,
                "exported": dec.get("export", False),
            }
        # Update cumulative tax ledger
        for n, t in tax_this_round.items():
            if t > 0:
                self.mnc_tax_revenue[n] = self.mnc_tax_revenue.get(n, 0.0) + t
        return profits, tax_this_round

    # ── Monetary & FX (Phase 5+) ──────────────────────────────────

    def _mon(self, country):
        """
        Return the monetary-state dict for a country: the shared union state
        if it belongs to a monetary union, else its own country config.
        """
        uid = self.countries[country].get("union_id")
        if uid and uid in self.monetary_unions:
            return self.monetary_unions[uid]["state"]
        return self.countries[country]

    def _monetary_entities(self):
        """
        List of (key, state, members) — one entry per independent monetary
        actor. A monetary union counts once; non-union countries count
        individually.
        """
        entities = []
        seen = set()
        for c in self.countries:
            uid = self.countries[c].get("union_id")
            if uid:
                if uid in seen:
                    continue
                seen.add(uid)
                entities.append(
                    (uid, self.monetary_unions[uid]["state"],
                     list(self.monetary_unions[uid]["members"]))
                )
            else:
                entities.append((c, self.countries[c], [c]))
        return entities

    def _validate_monetary_decisions(self, monetary_decisions):
        errors = []
        valid_regimes = {"peg", "managed", "float"}
        for c, md in monetary_decisions.items():
            if c not in self.countries:
                errors.append(f"Unknown country in monetary_decisions: {c}")
                continue
            if md.get("fx_regime") not in valid_regimes:
                errors.append(
                    f"{c}: fx_regime must be one of {sorted(valid_regimes)}"
                )
            if not isinstance(md.get("capital_controls"), bool):
                errors.append(f"{c}: capital_controls must be True/False")
            if not isinstance(md.get("independent_monetary"), bool):
                errors.append(f"{c}: independent_monetary must be True/False")
            g = md.get("money_supply_growth", 0.0)
            if g not in PHASE5_MONEY_GROWTH_CHOICES:
                errors.append(
                    f"{c}: money_supply_growth {g} not in "
                    f"{PHASE5_MONEY_GROWTH_CHOICES}"
                )
        # Monetary-union consistency: members must decide together, identically
        keys = ("fx_regime", "capital_controls",
                "independent_monetary", "money_supply_growth")
        for uid, union in self.monetary_unions.items():
            members = union["members"]
            present = [m for m in members if m in monetary_decisions]
            if not present:
                continue
            if len(present) != len(members):
                missing = set(members) - set(present)
                errors.append(
                    f"Union {uid}: all members must submit decisions together "
                    f"(missing: {sorted(missing)})"
                )
                continue
            ref = {k: monetary_decisions[present[0]].get(k) for k in keys}
            for m in present[1:]:
                cur = {k: monetary_decisions[m].get(k) for k in keys}
                if cur != ref:
                    errors.append(
                        f"Union {uid}: {m}'s decisions differ from "
                        f"{present[0]} (union members must be identical)"
                    )
        return errors

    def _apply_monetary_decisions(self, monetary_decisions):
        """
        Apply per-round monetary choices, then run the trilemma:
          - overreach (peg + open capital + independent monetary) raises stress
          - stress=1 -> warning crisis (10% devaluation, friction bump)
          - stress=2 -> full crisis (30% devaluation, welfare hit), reset
          - money-supply growth decays the currency each round

        Returns events: {country: {warning, crisis, stress,
                                    depreciation_factor, warning_active}}.
        """
        # 1. Write decisions into monetary state (each entity once)
        applied = set()
        for c in self.countries:
            md = monetary_decisions.get(c)
            if md is None:
                continue
            uid = self.countries[c].get("union_id")
            key = uid if uid else c
            if key in applied:
                continue
            state = self._mon(c)
            state["fx_regime"] = md["fx_regime"]
            state["capital_controls"] = md["capital_controls"]
            state["independent_monetary"] = md["independent_monetary"]
            state["money_supply_growth"] = md["money_supply_growth"]
            applied.add(key)

        # 2. Trilemma resolution per monetary entity
        events = {
            c: {"warning": False, "crisis": False} for c in self.countries
        }
        # Consume any manually-forced attacks (devaluation already applied by
        # inject_speculative_attack); deliver the welfare hit this round.
        for key, state, members in self._monetary_entities():
            if state.pop("_forced_attack", False):
                for m in members:
                    events[m]["crisis"] = True
        for key, state, members in self._monetary_entities():
            overreach = (
                state.get("fx_regime") == "peg"
                and not state.get("capital_controls", False)
                and state.get("independent_monetary", True)
            )
            if overreach:
                state["stress"] = state.get("stress", 0) + 1
            else:
                state["stress"] = 0
                state["warning_active"] = False

            if state["stress"] >= 2:
                state["depreciation_factor"] *= CRISIS_DEVALUATION
                state["stress"] = 0
                state["warning_active"] = False
                for m in members:
                    events[m]["crisis"] = True
            elif state["stress"] == 1:
                state["depreciation_factor"] *= WARNING_DEVALUATION
                state["warning_active"] = True
                for m in members:
                    events[m]["warning"] = True

            # Money-supply decay (gradual depreciation)
            if state.get("independent_monetary", True):
                state["depreciation_factor"] *= (
                    1 - state.get("money_supply_growth", 0.0)
                )

        # 3. Snapshot resolved state per country
        for c in self.countries:
            s = self._mon(c)
            events[c]["stress"] = s["stress"]
            events[c]["depreciation_factor"] = s["depreciation_factor"]
            events[c]["warning_active"] = s.get("warning_active", False)
        return events

    def _compute_fx_friction(self, exporter, importer):
        """
        Fraction of a cross-border shipment lost to currency friction
        (stacks on top of tariffs). Zero if the reserve currency holder is a
        party, or if both sides share a monetary union. Otherwise a baseline,
        plus a bump if either side is in a post-warning state.
        """
        if self.phase < 5:
            return 0.0
        rc = self.reserve_currency_holder
        if exporter == rc or importer == rc:
            return 0.0
        ex_uid = self.countries[exporter].get("union_id")
        im_uid = self.countries[importer].get("union_id")
        if ex_uid and ex_uid == im_uid:
            return 0.0
        friction = BASE_FX_FRICTION
        # Pegging to the reserve currency imports credibility: half friction
        if self._mon(exporter).get("fx_regime") == "peg" and \
           self._mon(importer).get("fx_regime") == "peg":
            friction *= 0.5
        # Warning jitters add friction if either party was just warned
        if self._mon(exporter).get("warning_active") or \
           self._mon(importer).get("warning_active"):
            friction += WARNING_FRICTION_BUMP
        return min(friction, 1.0)

    # ── Trade execution ───────────────────────────────────────────

    def _execute_trades(self, consumption, decisions, trades, varieties=None):
        """
        Execute trades. Tariffs destroy a fraction of imports (deadweight loss).
        Returns (trade_log, tariff_losses, trade_records).
          - trade_log: human-readable strings, one per executed/failed trade
          - tariff_losses: goods destroyed at the border, per (country, good)
          - trade_records: structured dicts, one per SUCCESSFUL trade
            (used by plot_trade_network and any downstream analysis)

        If `varieties` is provided (Phase 3+), per-variety inventories are
        moved proportionally to the exporter's variety mix and updated in place.
        """
        trade_log = []
        trade_records = []
        tariff_losses = {c: {g: 0.0 for g in self.goods} for c in self.countries}

        for trade in trades:
            exporter, importer, good_out, qty_out, good_in, qty_in = trade

            # Reject self-trade: net effect is zero but tariffs would silently
            # destroy goods. Catches typos in the trade list.
            if exporter == importer:
                trade_log.append(
                    f"  FAILED: self-trade -- {exporter} cannot trade with itself"
                )
                continue

            # Validate quantities available
            if consumption[exporter].get(good_out, 0) < qty_out - 0.01:
                trade_log.append(
                    f"  FAILED: {exporter} cannot export {qty_out:.1f} {good_out} "
                    f"(has {consumption[exporter][good_out]:.1f})"
                )
                continue
            if consumption[importer].get(good_in, 0) < qty_in - 0.01:
                trade_log.append(
                    f"  FAILED: {importer} cannot export {qty_in:.1f} {good_in} "
                    f"(has {consumption[importer][good_in]:.1f})"
                )
                continue

            # Look up tariffs (default 0). Phase 4+ populist regimes may
            # impose a country-wide tariff_floor (minimum applied to all
            # imports regardless of declared rate).
            t_importer = (
                decisions[importer]
                .get("tariffs", {})
                .get(exporter, {})
                .get(good_out, 0.0)
            )
            t_exporter = (
                decisions[exporter]
                .get("tariffs", {})
                .get(importer, {})
                .get(good_in, 0.0)
            )
            t_importer = max(
                t_importer,
                self.countries[importer].get("tariff_floor", 0.0),
            )
            t_exporter = max(
                t_exporter,
                self.countries[exporter].get("tariff_floor", 0.0),
            )

            # FX friction (Phase 5+) stacks multiplicatively on tariffs and
            # applies symmetrically to both legs of the barter.
            fx = self._compute_fx_friction(exporter, importer)
            loss_importer = 1 - (1 - t_importer) * (1 - fx)  # on good_out
            loss_exporter = 1 - (1 - t_exporter) * (1 - fx)  # on good_in

            received_by_importer = qty_out * (1 - loss_importer)
            received_by_exporter = qty_in * (1 - loss_exporter)
            destroyed_at_importer = qty_out * loss_importer
            destroyed_at_exporter = qty_in * loss_exporter

            # Update consumption (scalar totals)
            consumption[exporter][good_out] -= qty_out
            consumption[exporter][good_in] += received_by_exporter
            consumption[importer][good_in] -= qty_in
            consumption[importer][good_out] += received_by_importer

            # Update per-variety inventories (Phase 3+)
            if varieties is not None:
                self._transfer_varieties(
                    varieties, exporter, importer, good_out, qty_out, loss_importer
                )
                self._transfer_varieties(
                    varieties, importer, exporter, good_in, qty_in, loss_exporter
                )

            # Track losses (tariff + FX, combined)
            tariff_losses[importer][good_out] += destroyed_at_importer
            tariff_losses[exporter][good_in] += destroyed_at_exporter

            # Log: implicit terms of trade, tariffs, FX friction
            tot_str = ""
            if qty_out > 0:
                tot = qty_in / qty_out
                tot_str = f"  [ToT: 1 {good_out} = {tot:.2f} {good_in}]"
            tariff_str = ""
            if t_importer > 0 or t_exporter > 0:
                tariff_str = (
                    f" [tariffs: {t_importer:.0%} on {good_out}, "
                    f"{t_exporter:.0%} on {good_in}]"
                )
            fx_str = f" [FX friction: {fx:.0%}]" if fx > 0 else ""
            trade_log.append(
                f"  {exporter} -> {importer}: "
                f"{qty_out:.0f} {good_out} for {qty_in:.0f} {good_in}"
                f"{tot_str}{tariff_str}{fx_str}"
            )
            trade_records.append({
                "exporter": exporter,
                "importer": importer,
                "good_out": good_out,
                "qty_out_sent": qty_out,
                "qty_out_received": received_by_importer,
                "tariff_importer": t_importer,
                "good_in": good_in,
                "qty_in_sent": qty_in,
                "qty_in_received": received_by_exporter,
                "tariff_exporter": t_exporter,
                "fx_friction": fx,
                "tot": qty_in / qty_out if qty_out > 0 else 0.0,
            })

        return trade_log, tariff_losses, trade_records

    def _transfer_varieties(self, varieties, src, dst, good, qty, tariff):
        """
        Move `qty` of `good` from src to dst, distributing across varieties
        in proportion to src's current variety mix. A tariff fraction is
        destroyed at the border; the remainder lands in dst's variety pool.
        """
        src_pool = varieties[src][good]
        total_src = sum(src_pool.values())
        if total_src <= 1e-9:
            return
        frac = min(qty / total_src, 1.0)
        dst_pool = varieties[dst][good]
        for variety, src_qty in list(src_pool.items()):
            transferred = src_qty * frac
            src_pool[variety] = src_qty - transferred
            received = transferred * (1.0 - tariff)
            if received > 0:
                dst_pool[variety] = dst_pool.get(variety, 0.0) + received

    # ── Welfare ───────────────────────────────────────────────────

    def _utility(self, bundle) -> float:
        """
        Cobb-Douglas across goods with equal weights.
        Phase 1-2: bundle is {good: scalar_qty}. U = prod(c_g)^(1/J).
        Phase 3+:  bundle is {good: {variety: qty}}. CES nest within each
                   good aggregates varieties, then Cobb-Douglas across goods.
        """
        if not bundle:
            return 0.0
        # Detect variety-bundle (Phase 3+) vs scalar (Phase 1-2)
        sample = next(iter(bundle.values()))
        if isinstance(sample, dict):
            return self._utility_with_varieties(bundle)
        # Scalar (legacy) path
        J = len(self.goods)
        u = 1.0
        for good in self.goods:
            c = bundle.get(good, 0.0)
            if c > 0:
                u *= c ** (1.0 / J)
            else:
                return 0.0
        return u

    def _utility_with_varieties(self, bundle: dict) -> float:
        """CES variety nest inside Cobb-Douglas across goods."""
        J = len(self.goods)
        rho = self.variety_rho
        u = 1.0
        for good in self.goods:
            varieties = bundle.get(good, {})
            if not varieties:
                return 0.0
            ces_sum = sum(
                (qty ** rho) for qty in varieties.values() if qty > 0
            )
            if ces_sum <= 0:
                return 0.0
            c_good = ces_sum ** (1.0 / rho)
            u *= c_good ** (1.0 / J)
        return u

    # ── Factor prices (Phase 2) ───────────────────────────────────

    def _compute_factor_prices(self, country_name, decision, production):
        """Marginal products as implicit factor prices."""
        config = self.countries[country_name]
        wages = {}
        capital_returns = {}

        for good in self.goods:
            L = decision["production"]["labor"].get(good, 0)
            K = decision["production"]["capital"].get(good, 0)
            q = production[good]
            tech = config["tech"][good]

            if L > 0 and q > 0:
                wages[good] = tech["labor_share"] * q / L
            if K > 0 and q > 0:
                capital_returns[good] = tech["capital_share"] * q / K

        # Average factor prices across sectors (weighted by allocation)
        total_L = sum(
            decision["production"]["labor"].get(g, 0) for g in self.goods
        )
        total_K = sum(
            decision["production"]["capital"].get(g, 0) for g in self.goods
        )
        avg_wage = (
            sum(
                wages.get(g, 0) * decision["production"]["labor"].get(g, 0)
                for g in self.goods
            ) / total_L if total_L > 0 else 0
        )
        avg_return = (
            sum(
                capital_returns.get(g, 0)
                * decision["production"]["capital"].get(g, 0)
                for g in self.goods
            ) / total_K if total_K > 0 else 0
        )

        return {
            "wages_by_sector": wages,
            "capital_returns_by_sector": capital_returns,
            "avg_wage": avg_wage,
            "avg_capital_return": avg_return,
        }

    # ── Validation ────────────────────────────────────────────────

    def _validate_decisions(self, decisions):
        errors = []
        for name, config in self.countries.items():
            if name not in decisions:
                errors.append(f"Missing decisions for {name}")
                continue

            dec = decisions[name]

            if self.phase == 1:
                total_labor = sum(dec["production"].get(g, 0) for g in self.goods)
                if abs(total_labor - config["labor"]) > 0.5:
                    errors.append(
                        f"{name}: labor allocation ({total_labor}) "
                        f"!= endowment ({config['labor']})"
                    )
                for g in self.goods:
                    if dec["production"].get(g, 0) < 0:
                        errors.append(f"{name}: negative labor in {g}")
            else:
                total_L = sum(
                    dec["production"]["labor"].get(g, 0) for g in self.goods
                )
                total_K = sum(
                    dec["production"]["capital"].get(g, 0) for g in self.goods
                )
                if abs(total_L - config["labor"]) > 0.5:
                    errors.append(
                        f"{name}: labor allocation ({total_L}) "
                        f"!= endowment ({config['labor']})"
                    )
                if abs(total_K - config["capital"]) > 0.5:
                    errors.append(
                        f"{name}: capital allocation ({total_K}) "
                        f"!= endowment ({config['capital']})"
                    )

            # Validate tariff rates
            for partner, goods_tariffs in dec.get("tariffs", {}).items():
                for good, rate in goods_tariffs.items():
                    if not (0 <= rate <= 1.0):
                        errors.append(
                            f"{name}: tariff on {good} from {partner} "
                            f"= {rate} (must be 0-1)"
                        )
        return errors

    def _validate_firm_decisions(self, firm_decisions):
        """Phase 3+ firm decisions: scale in [0, max_scale], relocate_to valid."""
        errors = []
        for fid in self.firms:
            if fid not in firm_decisions:
                errors.append(f"Missing firm decision for {fid}")
                continue
            dec = firm_decisions[fid]
            fcfg = self.firm_config[fid]
            scale = dec.get("scale", 0)
            if not (0 <= scale <= fcfg["max_scale"] + 0.01):
                errors.append(
                    f"{fid}: scale {scale} outside [0, {fcfg['max_scale']}]"
                )
            relocate_to = dec.get("relocate_to")
            if relocate_to is not None and relocate_to not in self.countries:
                errors.append(
                    f"{fid}: relocate_to '{relocate_to}' is not a country"
                )
        return errors

    # ── Phase transition ──────────────────────────────────────────

    def upgrade_to_phase2(self, new_countries: dict, new_goods: list):
        """
        Transition from Phase 1 to Phase 2 mid-simulation.
        Keeps history intact; future rounds use new parameters.
        """
        self.countries = deepcopy(new_countries)
        self.goods = list(new_goods)
        self.phase = 2
        print(f"{'':=<55}")
        print(f"  Upgraded to Phase 2: Heckscher-Ohlin")
        print(f"{'':=<55}")
        print(f"  Goods: {', '.join(self.goods)}")
        print(f"  Countries now have labor + capital endowments.")
        print(f"  Factor prices (wages, returns to capital) tracked.\n")

    def upgrade_to_phase3(self, firms_config: dict = None):
        """
        Transition to Phase 3: introduce MNCs with differentiated varieties.

        Country endowments and tech carry over from Phase 2. Adds:
          - Firms with industry, variety, host country, productivity.
          - CES variety preference inside each industry.
          - Per-firm profit accounting in numeraire welfare units.

        firms_config defaults to PHASE3_FIRMS. Pass a subset (e.g., F1-F10)
        to match class enrollment.
        """
        if firms_config is None:
            firms_config = PHASE3_FIRMS
        self.firm_config = deepcopy(firms_config)
        self.firms = {
            fid: {"host": cfg["default_host"], "cumulative_profit": 0.0}
            for fid, cfg in firms_config.items()
        }
        self.phase = 3
        print(f"{'':=<55}")
        print(f"  Upgraded to Phase 3: MNCs and Varieties")
        print(f"{'':=<55}")
        print(f"  Firms: {len(self.firms)} MNCs across {len(self.countries)} countries")
        print(f"  Variety preference: CES with rho={self.variety_rho}")
        print(f"  World reference prices: {self.world_prices}\n")

    def upgrade_to_phase5(self):
        """
        Transition to Phase 5: money, exchange rates, and the trilemma.

        Requires award_reserve_currency() to have been run (the winner's
        currency becomes the default invoicing unit). Each country gets a
        currency and default monetary policy (managed float, open capital,
        independent monetary, 0% money growth). Firms and country endowments
        carry over from Phase 4.
        """
        if self.reserve_currency_holder is None:
            raise ValueError(
                "Run award_reserve_currency() before upgrade_to_phase5(): "
                "the reserve currency holder is needed for FX defaults."
            )
        for c in self.countries:
            cfg = self.countries[c]
            cfg.setdefault("currency", f"{c} peso")
            cfg["fx_regime"] = "managed"          # peg | managed | float
            cfg["capital_controls"] = False        # open capital by default
            cfg["independent_monetary"] = True     # runs own monetary policy
            cfg["money_supply_growth"] = 0.0
            cfg["stress"] = 0
            cfg["depreciation_factor"] = 1.0
            cfg["union_id"] = None
            cfg["warning_active"] = False
        self.monetary_unions = {}
        self.phase = 5
        print(f"{'':=<55}")
        print(f"  Upgraded to Phase 5: Money & FX (the Trilemma)")
        print(f"{'':=<55}")
        print(f"  Reserve currency: {self.countries[self.reserve_currency_holder]['currency']} "
              f"({self.reserve_currency_holder})")
        print(f"  Money-growth choices: "
              f"{[f'{g:.0%}' for g in PHASE5_MONEY_GROWTH_CHOICES]}")
        print(f"  Overreach (peg + open capital + independent money) "
              f"accumulates stress.")
        print(f"  stress=1 -> warning ({1-WARNING_DEVALUATION:.0%} devaluation); "
              f"stress=2 -> crisis ({1-CRISIS_DEVALUATION:.0%} + "
              f"{CRISIS_WELFARE_HIT:.0%} welfare).\n")

    # ── End-of-Phase-4 ceremonies ─────────────────────────────────

    def award_reserve_currency(self):
        """
        Rank countries by cumulative welfare across all completed rounds,
        with Phase 4 welfare as the tiebreaker. Set
        self.reserve_currency_holder = top country.

        Call at the end of Phase 4 (Round 16 in the default schedule).
        The winner becomes the Phase 5 reserve currency holder and the
        Phase 6 hegemon.

        Returns the full ranking list (best first).
        """
        if not self.history:
            print("No rounds played; cannot award reserve currency.")
            return []
        cum_welfare = {n: 0.0 for n in self.countries}
        p4_welfare = {n: 0.0 for n in self.countries}
        for h in self.history:
            for n in self.countries:
                if n in h["results"]:
                    w = h["results"][n]["welfare"]
                    cum_welfare[n] += w
                    if h.get("phase") == 4:
                        p4_welfare[n] += w
        # Rank descending by (cumulative, phase4 tiebreaker)
        ranking = sorted(
            self.countries.keys(),
            key=lambda n: (cum_welfare[n], p4_welfare[n]),
            reverse=True,
        )
        self.reserve_currency_holder = ranking[0]
        print(f"\n{'':=<65}")
        print(f"  RESERVE CURRENCY AWARDED")
        print(f"{'':=<65}")
        print(f"  Cumulative welfare across trade rounds "
              f"(Phase 4 as tiebreaker):\n")
        print(f"  {'Rank':5s}{'Country':14s}{'Cumulative':>12s}{'Phase 4':>12s}")
        print(f"  {'-'*43}")
        for i, n in enumerate(ranking, 1):
            mark = "*" if i == 1 else " "
            print(
                f"  {mark} {i:<3d}{n:14s}"
                f"{cum_welfare[n]:12.2f}{p4_welfare[n]:12.2f}"
            )
        print(
            f"\n  ** {self.reserve_currency_holder} ** is the reserve "
            f"currency holder."
        )
        print(
            f"  Their currency becomes the default invoicing unit in "
            f"Phase 5,"
        )
        print(f"  and they enter Phase 6 as the hegemon.\n")
        return ranking

    def print_firm_rankings(self):
        """Cumulative-profit leaderboard for MNC owners. Projectable."""
        if not self.firms:
            print("No firms loaded.")
            return
        ranked = sorted(
            self.firms.items(),
            key=lambda kv: kv[1]["cumulative_profit"],
            reverse=True,
        )
        print(f"\n{'':=<65}")
        print(f"  MNC PROFIT RANKINGS")
        print(f"{'':=<65}")
        print(
            f"  {'Rank':5s}{'ID':4s}{'Variety':10s}"
            f"{'Host':10s}{'Industry':11s}{'Cum.Profit':>12s}"
        )
        print(f"  {'-'*56}")
        for i, (fid, fstate) in enumerate(ranked, 1):
            cfg = self.firm_config[fid]
            print(
                f"  {i:5d}{fid:4s}{cfg['variety']:10s}"
                f"{fstate['host']:10s}{cfg['industry']:11s}"
                f"{fstate['cumulative_profit']:12.2f}"
            )
        print()

    def print_monetary_dashboard(self):
        """
        Projectable snapshot of current monetary state (Phase 5+): each
        country's regime, the trilemma triple, stress level, and FX index
        (depreciation factor; 1.00 = par, lower = weaker currency).
        """
        if self.phase < 5:
            print("Monetary dashboard available in Phase 5+.")
            return
        print(f"\n{'':=<72}")
        print(f"  MONETARY DASHBOARD")
        print(f"{'':=<72}")
        rc = self.reserve_currency_holder
        print(f"  Reserve currency: {self.countries[rc]['currency']} ({rc})\n")
        print(
            f"  {'Country':12s}{'Currency':16s}{'Regime':>8s}"
            f"{'Capital':>9s}{'Money':>7s}{'Stress':>8s}{'FXindex':>9s}"
        )
        print(f"  {'-'*68}")
        for c in self.countries:
            m = self._mon(c)
            cap = "controls" if m.get("capital_controls") else "open"
            mon = f"{m.get('money_supply_growth', 0):.0%}" \
                if m.get("independent_monetary") else "passive"
            union = self.countries[c].get("union_id")
            cur = m.get("currency", self.countries[c].get("currency", "?"))
            tag = f" [{union}]" if union else ""
            print(
                f"  {c:12s}{(cur + tag):16s}{m.get('fx_regime','?'):>8s}"
                f"{cap:>9s}{mon:>7s}{m.get('stress', 0):8d}"
                f"{m.get('depreciation_factor', 1.0):9.2f}"
            )
        if self.monetary_unions:
            print(f"\n  Monetary unions:")
            for name, u in self.monetary_unions.items():
                print(f"    {name}: {', '.join(u['members'])}")
        print(f"\n  (Overreach = peg + open capital + independent money. "
              f"stress>=2 -> crisis.)\n")

    # ── Instructor shock injection ────────────────────────────────

    def inject_shock(self, description: str, changes: dict):
        """
        Modify country parameters between rounds.

        Examples:
            sim.inject_shock("Oil crisis hits Llano", {
                "Llano": {"productivity": {"wine": 1.5}}  # Phase 1
            })
            sim.inject_shock("Capital flight from Trinity", {
                "Trinity": {"capital": 150}  # Phase 2
            })
        """
        print(f"\n{'':=<55}")
        print(f"  SHOCK: {description}")
        print(f"{'':=<55}")
        for country, params in changes.items():
            for key, value in params.items():
                if isinstance(value, dict):
                    if key not in self.countries[country]:
                        self.countries[country][key] = {}
                    for subkey, subval in value.items():
                        old = self.countries[country][key].get(subkey, "(unset)")
                        self.countries[country][key][subkey] = subval
                        print(f"  {country}: {key}.{subkey}: {old} -> {subval}")
                else:
                    old = self.countries[country].get(key, "(unset)")
                    self.countries[country][key] = value
                    print(f"  {country}: {key}: {old} -> {value}")
        print()

    def inject_productivity_surge(self, country: str, industry: str,
                                  multiplier: float, description: str = None):
        """
        Multiply a country's TFP in a given industry. Phase 2+ only.

        Pedagogically neutral — frame as a tech leap, an automation
        breakthrough, ML adoption, foreign-cost collapse, or whatever
        fits the lecture. The welfare and distributional effects are the
        same regardless of framing.
        """
        if "tech" not in self.countries[country]:
            raise ValueError(
                f"{country} has no 'tech' field; productivity surge requires "
                f"Phase 2+ country config."
            )
        current = self.countries[country]["tech"][industry]
        new_tfp = current["tfp"] * multiplier
        if description is None:
            description = (
                f"Productivity surge: {country} {industry} TFP "
                f"{current['tfp']:.2f} -> {new_tfp:.2f} ({multiplier}x)"
            )
        self.inject_shock(description, {
            country: {"tech": {industry: {
                "tfp": new_tfp,
                "labor_share": current["labor_share"],
                "capital_share": current["capital_share"],
            }}}
        })

    def inject_populist_backlash(self, country: str, description: str = None,
                                 tariff_floor: float = 0.20,
                                 mnc_tax_rate: float = 0.10):
        """
        Phase 4+ political shock: a country swings populist.

        Effects (until reversed):
          - tariff_floor: minimum tariff on ALL imports into this country
          - mnc_tax_rate: tax on MNC revenue, collected by host country
                          (stored in a separate ledger; does NOT affect welfare)

        Reverse later via a follow-up inject_shock setting both to 0.
        """
        if description is None:
            description = (
                f"Populist backlash in {country}: "
                f"{tariff_floor:.0%} tariff floor + "
                f"{mnc_tax_rate:.0%} MNC tax"
            )
        self.inject_shock(description, {
            country: {
                "tariff_floor": tariff_floor,
                "mnc_tax_rate": mnc_tax_rate,
            }
        })

    # ── Monetary instructor moves & unions (Phase 5+) ─────────────

    def inject_monetary_shock(self, country: str,
                              money_supply_growth: float,
                              description: str = None):
        """
        Force a country's (or its union's) money-supply growth rate. Used for
        the Round-18 'one country prints' event. Value must be one of
        PHASE5_MONEY_GROWTH_CHOICES. Takes effect the next time run_round
        applies monetary decisions (or immediately if you don't pass that
        country in the next monetary_decisions).
        """
        if money_supply_growth not in PHASE5_MONEY_GROWTH_CHOICES:
            raise ValueError(
                f"money_supply_growth must be one of "
                f"{PHASE5_MONEY_GROWTH_CHOICES}"
            )
        state = self._mon(country)
        old = state.get("money_supply_growth", 0.0)
        state["money_supply_growth"] = money_supply_growth
        state["independent_monetary"] = True  # printing requires sovereignty
        if description is None:
            description = (
                f"Monetary shock: {country} money growth "
                f"{old:.0%} -> {money_supply_growth:.0%}"
            )
        print(f"\n{'':=<55}")
        print(f"  SHOCK: {description}")
        print(f"{'':=<55}\n")

    def inject_speculative_attack(self, country: str, description: str = None):
        """
        Manually fire a full currency crisis on a country (or its union),
        regardless of accumulated stress. Safety valve when class pacing
        outruns the stress accumulator and you want the lesson NOW.
        """
        state = self._mon(country)
        state["depreciation_factor"] *= CRISIS_DEVALUATION
        state["stress"] = 0
        state["warning_active"] = False
        state["_forced_attack"] = True  # consumed by next run_round
        if description is None:
            description = (
                f"Speculative attack on {country}: currency devalues "
                f"{1 - CRISIS_DEVALUATION:.0%}"
            )
        print(f"\n{'':=<55}")
        print(f"  SHOCK: {description}")
        print(f"{'':=<55}\n")

    def inject_capital_flight(self, country: str, severity: float = 0.5,
                              description: str = None):
        """
        Sudden capital outflow: depreciates the currency by `severity` (a
        fraction, e.g. 0.5 = 30%... no — 0.5 means depreciation_factor *= 0.5).
        Models a balance-of-payments shock for Round 22. Capital controls do
        not stop it here, but in the debrief discuss whether they should have.
        """
        if not (0 < severity < 1):
            raise ValueError("severity must be in (0, 1)")
        state = self._mon(country)
        state["depreciation_factor"] *= severity
        if description is None:
            description = (
                f"Capital flight from {country}: currency loses "
                f"{1 - severity:.0%} of its value"
            )
        print(f"\n{'':=<55}")
        print(f"  SHOCK: {description}")
        print(f"{'':=<55}\n")

    def form_monetary_union(self, *countries, name: str):
        """
        Merge two or more countries into a monetary union with shared currency,
        monetary policy, stress counter, and depreciation factor. The union's
        initial state is seeded from the average depreciation factor of members
        and a managed-float / open-capital / independent-monetary default.
        """
        members = list(countries)
        if len(members) < 2:
            raise ValueError("A monetary union needs at least two members.")
        for c in members:
            if c not in self.countries:
                raise ValueError(f"Unknown country: {c}")
            if self.countries[c].get("union_id"):
                raise ValueError(f"{c} is already in a union.")
        avg_dep = sum(
            self.countries[c].get("depreciation_factor", 1.0) for c in members
        ) / len(members)
        self.monetary_unions[name] = {
            "members": members,
            "state": {
                "currency": name,
                "fx_regime": "managed",
                "capital_controls": False,
                "independent_monetary": True,
                "money_supply_growth": 0.0,
                "stress": 0,
                "depreciation_factor": avg_dep,
                "warning_active": False,
            },
        }
        for c in members:
            self.countries[c]["union_id"] = name
        print(f"\n{'':=<55}")
        print(f"  MONETARY UNION FORMED: {name}")
        print(f"{'':=<55}")
        print(f"  Members: {', '.join(members)}")
        print(f"  Shared currency, monetary policy, and crisis exposure.")
        print(f"  Members trade with each other at 0% FX friction.\n")

    def dissolve_monetary_union(self, name: str):
        """Dissolve a union; members revert to their own currency state,
        inheriting the union's final depreciation factor."""
        if name not in self.monetary_unions:
            raise ValueError(f"No such union: {name}")
        union = self.monetary_unions.pop(name)
        final_dep = union["state"]["depreciation_factor"]
        for c in union["members"]:
            self.countries[c]["union_id"] = None
            self.countries[c]["depreciation_factor"] = final_dep
            self.countries[c]["stress"] = 0
            self.countries[c]["warning_active"] = False
        print(f"\n{'':=<55}")
        print(f"  MONETARY UNION DISSOLVED: {name}")
        print(f"{'':=<55}")
        print(f"  {', '.join(union['members'])} revert to own currencies "
              f"(dep factor {final_dep:.2f}).\n")

    # ── Display ───────────────────────────────────────────────────

    def print_results(self, round_num: int = None):
        """Pretty-print results for a given round (default: latest)."""
        if not self.history:
            print("No rounds played yet.")
            return

        if round_num is None:
            rd = self.history[-1]
        else:
            rd = self.history[round_num - 1]

        res = rd["results"]
        phase = rd["phase"]
        names = list(self.countries.keys())
        goods = self.goods

        print(f"\n{'':=<65}")
        print(f"  ROUND {rd['round']} RESULTS  --  Phase {phase}")
        print(f"{'':=<65}")

        # Production table
        print(f"\n  PRODUCTION")
        header = f"  {'':16s}" + "".join(f"{g:>12s}" for g in goods)
        print(header)
        print(f"  {'':16s}" + "".join("       -----" for _ in goods))
        for name in names:
            row = f"  {name:16s}"
            for g in goods:
                row += f"{res[name]['production'].get(g, 0):12.1f}"
            print(row)

        # Trade log
        print(f"\n  TRADES")
        if rd["trade_log"]:
            for line in rd["trade_log"]:
                print(line)
        else:
            print("  (no trades this round)")

        # Consumption + welfare table
        print(f"\n  CONSUMPTION & WELFARE")
        header = (
            f"  {'':16s}"
            + "".join(f"{g:>12s}" for g in goods)
            + f"{'Welfare':>12s}{'vs No Trade':>12s}"
        )
        print(header)
        print(
            f"  {'':16s}"
            + "".join("       -----" for _ in range(len(goods) + 2))
        )
        for name in names:
            row = f"  {name:16s}"
            for g in goods:
                row += f"{res[name]['consumption'].get(g, 0):12.1f}"
            row += f"{res[name]['welfare']:12.1f}"
            gains = res[name]["gains_from_trade_pct"]
            if gains == float("inf"):
                row += f"{'n/a (0)':>11s}"
            else:
                sign = "+" if gains >= 0 else ""
                row += f"{sign}{gains:10.1f}%"
            print(row)

        # Tariff losses
        any_tariffs = any(
            any(v > 0 for v in res[n]["tariff_losses"].values())
            for n in names
        )
        if any_tariffs:
            print(f"\n  TARIFF LOSSES (goods destroyed)")
            header = f"  {'':16s}" + "".join(f"{g:>12s}" for g in goods)
            print(header)
            for name in names:
                tr = res[name]["tariff_losses"]
                if any(v > 0 for v in tr.values()):
                    row = f"  {name:16s}"
                    for g in goods:
                        row += f"{tr.get(g, 0):12.1f}"
                    print(row)

        # Factor prices (Phase 2+)
        if phase >= 2:
            print(f"\n  FACTOR PRICES (marginal products)")
            print(f"  {'':16s}{'Avg Wage':>12s}{'Avg Return':>12s}")
            print(f"  {'':16s}{'-----':>12s}{'-----':>12s}")
            for name in names:
                fp = res[name]["factor_prices"]
                print(
                    f"  {name:16s}"
                    f"{fp['avg_wage']:12.2f}"
                    f"{fp['avg_capital_return']:12.2f}"
                )

        # Firms (Phase 3+)
        if phase >= 3 and "firms" in rd:
            print(f"\n  FIRMS (MNCs)")
            print(
                f"  {'ID':4s}{'Variety':10s}{'Host':10s}"
                f"{'Scale':>8s}{'Output':>10s}{'Profit':>10s}{'Cum.Prof':>10s}"
            )
            print(f"  {'-'*52}")
            for fid, fr in rd["firms"].items():
                tag = ""
                if fr["relocated"]:
                    tag = " (moved)"
                elif fr["exported"]:
                    tag = " (exp)"
                print(
                    f"  {fid:4s}{fr['variety']:10s}{fr['host']:10s}"
                    f"{fr['scale']:8.1f}{fr['output']:10.1f}"
                    f"{fr['profit']:10.1f}{fr['cumulative_profit']:10.1f}{tag}"
                )

        # Tariff floor in effect (Phase 4+; populist regimes)
        if phase >= 4:
            floors = {
                n: self.countries[n].get("tariff_floor", 0.0)
                for n in names
                if self.countries[n].get("tariff_floor", 0.0) > 0
            }
            if floors:
                print(f"\n  TARIFF FLOORS (populist regimes)")
                for n, f in floors.items():
                    print(f"  {n:16s}{f:.0%} floor on all imports")

        # MNC tax revenue (Phase 4+; separate ledger, NOT in welfare)
        if phase >= 4 and "mnc_tax_this_round" in rd:
            tr = rd["mnc_tax_this_round"]
            cum = rd["mnc_tax_cumulative"]
            if any(t > 0 for t in tr.values()) or any(c > 0 for c in cum.values()):
                print(f"\n  MNC TAX LEDGER (separate; does NOT enter welfare)")
                print(
                    f"  {'':16s}{'This round':>14s}{'Cumulative':>14s}"
                )
                print(f"  {'-'*44}")
                for n in names:
                    this_round = tr.get(n, 0.0)
                    cumulative = cum.get(n, 0.0)
                    if this_round > 0 or cumulative > 0:
                        print(
                            f"  {n:16s}{this_round:14.2f}{cumulative:14.2f}"
                        )

        # Monetary regimes + crisis events (Phase 5+)
        if phase >= 5:
            print(f"\n  MONETARY  (regime / capital / money | stress, FX index)")
            print(
                f"  {'':12s}{'Regime':>9s}{'Cap':>6s}{'Money':>7s}"
                f"{'Stress':>8s}{'FXindex':>9s}"
            )
            print(f"  {'-'*51}")
            for n in names:
                m = res[n].get("monetary", {})
                cap = "ctrl" if m.get("capital_controls") else "open"
                mon = f"{m.get('money_supply_growth', 0):.0%}" \
                    if m.get("independent_monetary") else "pass"
                print(
                    f"  {n:12s}{m.get('fx_regime', '?'):>9s}{cap:>6s}{mon:>7s}"
                    f"{m.get('stress', 0):8d}"
                    f"{m.get('depreciation_factor', 1.0):9.2f}"
                )
            # Crisis / warning banners
            for n in names:
                m = res[n].get("monetary", {})
                if m.get("crisis"):
                    print(
                        f"\n  ** CRISIS in {n}: currency devalued "
                        f"{1 - CRISIS_DEVALUATION:.0%}, welfare "
                        f"-{m.get('crisis_welfare_loss', 0):.1f} **"
                    )
                elif m.get("warning"):
                    print(
                        f"\n  ! WARNING in {n}: currency slipped "
                        f"{1 - WARNING_DEVALUATION:.0%} (one more overreach round "
                        f"= full crisis)"
                    )

        print(f"\n{'':=<65}\n")

    # ── Visualization ─────────────────────────────────────────────

    def plot_welfare(self, figsize=(10, 5)):
        """
        Line chart of welfare. Splits into one subplot per phase, because
        Cobb-Douglas utility with different numbers of goods is not
        comparable across phases.
        """
        if not self.history:
            print("No rounds to plot.")
            return

        phases = sorted({h["phase"] for h in self.history})
        fig, axes = plt.subplots(
            1, len(phases),
            figsize=(figsize[0] * len(phases), figsize[1]),
            squeeze=False,
        )

        for i, ph in enumerate(phases):
            ax = axes[0, i]
            ph_hist = [h for h in self.history if h["phase"] == ph]
            rounds = [h["round"] for h in ph_hist]
            ph_names = list(ph_hist[-1]["results"].keys())
            for name in ph_names:
                welfares = [
                    h["results"][name]["welfare"]
                    for h in ph_hist if name in h["results"]
                ]
                ax.plot(rounds, welfares, marker="o", linewidth=2, label=name)
            ax.set_xlabel("Round", fontsize=12)
            ax.set_ylabel("Welfare (utility index)", fontsize=12)
            ax.set_title(
                f"Phase {ph} -- Welfare Over Time",
                fontsize=14, fontweight="bold",
            )
            ax.legend(loc="best", fontsize=10)
            ax.set_xticks(rounds)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_production(self, round_num: int = None, figsize=(10, 5)):
        """Grouped bar chart of production by country."""
        if not self.history:
            print("No rounds to plot.")
            return

        rd = (
            self.history[-1] if round_num is None
            else self.history[round_num - 1]
        )
        res = rd["results"]
        names = list(self.countries.keys())

        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(names))
        width = 0.8 / len(self.goods)

        for i, good in enumerate(self.goods):
            values = [res[n]["production"].get(good, 0) for n in names]
            offset = (i - len(self.goods) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=good.capitalize())

        ax.set_xlabel("Country", fontsize=12)
        ax.set_ylabel("Output", fontsize=12)
        ax.set_title(
            f"Round {rd['round']} -- Production",
            fontsize=14, fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    def plot_gains_from_trade(self, figsize=(10, 5)):
        """Bar chart: who benefits from trade (latest round)."""
        if not self.history:
            print("No rounds to plot.")
            return

        rd = self.history[-1]
        res = rd["results"]
        names = list(self.countries.keys())
        gains = [res[n]["gains_from_trade_pct"] for n in names]
        colors = ["#2ecc71" if g >= 0 else "#e74c3c" for g in gains]

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(names, gains, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_ylabel("Gains from Trade (%)", fontsize=12)
        ax.set_title(
            f"Round {rd['round']} -- Who Benefits from Trade?",
            fontsize=14, fontweight="bold",
        )
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.show()

    # Colorblind-safe palette (Wong 2011, Nat Methods). Reused by all the
    # spatial plots so cloth/wine/machinery have consistent colors throughout.
    _GOOD_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#F0E442"]

    def _good_color_map(self):
        return {g: self._GOOD_PALETTE[i % len(self._GOOD_PALETTE)]
                for i, g in enumerate(self.goods)}

    def _country_layout(self, radius=1.0):
        """Even circular layout. Country 0 at the bottom (-y), going clockwise."""
        names = list(self.countries.keys())
        n = len(names)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) - np.pi / 2
        return {c: (float(radius * np.cos(a)), float(radius * np.sin(a)))
                for c, a in zip(names, angles)}

    def _draw_country_labels(self, ax, pos, fontsize=12):
        """Country names as text with bbox; no scatter dot."""
        for c, (x, y) in pos.items():
            ax.text(
                x, y, c,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#222222",
                bbox=dict(
                    boxstyle="round,pad=0.45",
                    facecolor="white",
                    edgecolor="#444444",
                    linewidth=1.4,
                ),
                zorder=5,
            )

    def plot_trade_network(self, round_num: int = None, figsize=(10, 8)):
        """
        Network diagram of trade flows for one round.
        - Countries laid out evenly on a circle (text labels, no dots).
        - Each successful trade draws TWO curved arrows (one per direction).
        - Arrow color = good (colorblind-safe Wong palette); arrow width
          is proportional to the received quantity (post-tariff).
        Projectable for classroom debrief — students see hubs vs. periphery,
        and the visual shrinks when tariffs hit.
        """
        if not self.history:
            print("No rounds to plot.")
            return
        rd = (
            self.history[-1] if round_num is None
            else self.history[round_num - 1]
        )
        trades = rd.get("trades_executed", [])

        pos = self._country_layout(radius=1.0)
        good_colors = self._good_color_map()

        fig, ax = plt.subplots(figsize=figsize)
        self._draw_country_labels(ax, pos)

        if not trades:
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                f"Round {rd['round']} — No trades this round",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.show()
            return

        # Width normalization (received quantity, post-tariff)
        all_qtys = [t["qty_out_received"] for t in trades] + \
                   [t["qty_in_received"]  for t in trades]
        max_qty = max(all_qtys) if all_qtys else 1.0

        def draw_arrow(src, dst, color, qty, rad=0.18):
            width = max(0.8, 5.5 * qty / max_qty)
            ax.annotate(
                "", xy=pos[dst], xytext=pos[src],
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=width,
                    mutation_scale=22,            # arrowhead size
                    connectionstyle=f"arc3,rad={rad}",
                    shrinkA=32, shrinkB=32,       # buffer around text bbox
                    alpha=0.92,
                ),
                zorder=2,
            )

        for t in trades:
            draw_arrow(t["exporter"], t["importer"],
                       good_colors.get(t["good_out"], "gray"),
                       t["qty_out_received"], rad=0.18)
            draw_arrow(t["importer"], t["exporter"],
                       good_colors.get(t["good_in"], "gray"),
                       t["qty_in_received"], rad=0.18)

        # Legend
        from matplotlib.lines import Line2D
        legend = [
            Line2D([0], [0], color=c, lw=4, label=g.capitalize())
            for g, c in good_colors.items()
        ]
        ax.legend(handles=legend, loc="upper right", fontsize=11,
                  frameon=True, framealpha=0.95, title="Good")

        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"Round {rd['round']} — Trade Flows  "
            f"(arrow width ∝ received quantity, post-tariff)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_firm_locations(self, figsize=(11, 9)):
        """
        MNC location map (Phase 3+). Each firm shown as a colored disk near
        its host country; color by industry, size by productivity.

        Use this after each Phase 3+ round to see where firms cluster
        and how relocations shift the map.
        """
        if not self.firms:
            print("No firms loaded. Run upgrade_to_phase3() first.")
            return

        pos = self._country_layout(radius=1.0)
        good_colors = self._good_color_map()

        # Productivity → marker size (scatter `s` units)
        def prod_size(p):
            # 0.7 -> 220, 1.0 -> 340, 1.3 -> 460
            return max(140, 80 + 280 * p)

        fig, ax = plt.subplots(figsize=figsize)
        self._draw_country_labels(ax, pos)

        # Group firms by host
        by_host = {c: [] for c in pos}
        for fid in self.firms:
            host = self.firms[fid]["host"]
            if host in by_host:
                by_host[host].append(fid)

        # For each country, place firms in an outward arc
        for c, (x, y) in pos.items():
            firms_here = by_host[c]
            if not firms_here:
                continue
            r = float(np.hypot(x, y)) or 1e-9
            # Outward unit vector + perpendicular for arc spread
            ux, uy = x / r, y / r
            px, py = -uy, ux
            offset_r = 0.30
            k = len(firms_here)
            for i, fid in enumerate(firms_here):
                cfg = self.firm_config[fid]
                spread = (i - (k - 1) / 2) * 0.18
                fx = x + ux * offset_r + px * spread
                fy = y + uy * offset_r + py * spread
                sz = prod_size(cfg["productivity"])
                ax.scatter(
                    fx, fy, s=sz,
                    c=good_colors.get(cfg["industry"], "gray"),
                    alpha=0.9, edgecolors="black", linewidths=1.2,
                    zorder=3,
                )
                ax.text(
                    fx, fy, fid, fontsize=9, ha="center", va="center",
                    color="white", fontweight="bold", zorder=4,
                )

        # Two legends: industry color and productivity size
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        ind_legend = [
            Patch(facecolor=good_colors[g], edgecolor="black", label=g.capitalize())
            for g in self.goods
        ]
        size_legend = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#777777", markeredgecolor="black",
                   markersize=14, label="HIGH (1.3)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#777777", markeredgecolor="black",
                   markersize=11, label="MED  (1.0)"),
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#777777", markeredgecolor="black",
                   markersize=8,  label="LOW  (0.7)"),
        ]
        leg1 = ax.legend(handles=ind_legend, loc="upper right",
                         fontsize=11, title="Industry", frameon=True)
        ax.add_artist(leg1)
        ax.legend(handles=size_legend, loc="lower right",
                  fontsize=11, title="Productivity", frameon=True)

        ax.set_xlim(-1.7, 1.7)
        ax.set_ylim(-1.55, 1.55)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "MNC Locations  (color = industry, size = productivity)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.show()

    def plot_currency_health(self, figsize=(11, 6)):
        """
        Two-panel time series across Phase 5 rounds:
          left  = FX index (depreciation factor) per country
          right = stress level per country
        Crisis rounds show as sharp drops in the FX index.
        """
        mon_rounds = [h for h in self.history if h.get("phase", 0) >= 5]
        if not mon_rounds:
            print("No Phase 5 rounds to plot yet.")
            return
        names = list(self.countries.keys())
        rounds = [h["round"] for h in mon_rounds]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for name in names:
            fx = [h["results"][name]["monetary"]["depreciation_factor"]
                  for h in mon_rounds]
            st = [h["results"][name]["monetary"]["stress"]
                  for h in mon_rounds]
            ax1.plot(rounds, fx, marker="o", linewidth=2, label=name)
            ax2.plot(rounds, st, marker="s", linewidth=2, label=name)

        ax1.axhline(y=1.0, color="black", linewidth=0.8, alpha=0.5)
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("FX index (1.00 = par)", fontsize=12)
        ax1.set_title("Currency strength", fontsize=13, fontweight="bold")
        ax1.set_xticks(rounds)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9)

        ax2.axhline(y=2, color="#c0392b", linewidth=0.8, alpha=0.6,
                    linestyle="--", label="crisis threshold")
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Trilemma stress", fontsize=12)
        ax2.set_title("Stress accumulation", fontsize=13, fontweight="bold")
        ax2.set_xticks(rounds)
        ax2.set_yticks([0, 1, 2])
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plt.show()

    # ── Country briefs ────────────────────────────────────────────

    def print_country_briefs(self):
        """Print country endowments and comparative advantage hints."""
        print(f"\n{'':=<65}")
        print(f"  COUNTRY BRIEFS  --  Phase {self.phase}")
        print(f"{'':=<65}\n")

        for name, config in self.countries.items():
            print(f"  {name}")
            print(f"  {config.get('description', '')}")

            if self.phase == 1:
                print(f"  Labor: {config['labor']} workers")
                prods = ", ".join(
                    f"{g} = {config['productivity'][g]}"
                    for g in self.goods
                )
                print(f"  Productivity: {prods}")

                # Opportunity costs (2-good case)
                if len(self.goods) == 2:
                    g0, g1 = self.goods
                    opp = config["productivity"][g0] / config["productivity"][g1]
                    print(
                        f"  Opportunity cost: 1 {g1} = {opp:.2f} {g0} foregone"
                    )
            else:
                print(f"  Labor: {config['labor']}  |  Capital: {config['capital']}")
                ratio = config["capital"] / config["labor"]
                print(f"  K/L ratio: {ratio:.2f}")

            print()

    # ── Firm roster + firm forms (Phase 3+) ───────────────────────

    def print_firm_roster(self):
        """Display the MNC roster: firm IDs, varieties, hosts, productivity."""
        if not self.firms:
            print("No firms loaded. Run upgrade_to_phase3() first.")
            return
        print(f"\n{'':=<70}")
        print(f"  MNC ROSTER  --  {len(self.firms)} firms")
        print(f"{'':=<70}")
        print(
            f"  {'ID':4s}{'Variety':10s}{'Industry':11s}{'Host':10s}"
            f"{'Prod.':>7s}{'Max scale':>11s}{'Unit cost':>11s}"
        )
        print(f"  {'-'*64}")
        for fid, cfg in self.firm_config.items():
            host = self.firms[fid]["host"]
            print(
                f"  {fid:4s}{cfg['variety']:10s}{cfg['industry']:11s}"
                f"{host:10s}{cfg['productivity']:7.1f}"
                f"{cfg['max_scale']:11.0f}{cfg['unit_cost']:11.2f}"
            )
        print(f"\n  Productivity tiers: HIGH=1.3, MED=1.0, LOW=0.7")
        print(f"  Owner is always a student from a DIFFERENT country.\n")

    def print_firm_decision_forms(self, round_num: int = None):
        """Printable decision form per firm for MNC owners."""
        if not self.firms:
            print("No firms loaded.")
            return
        rnd = (self.round_num + 1) if round_num is None else round_num
        for fid, cfg in self.firm_config.items():
            host = self.firms[fid]["host"]
            cumprof = self.firms[fid]["cumulative_profit"]
            print(f"+{'-'*55}+")
            header = f"|  ROUND {rnd} -- FIRM {fid} ({cfg['variety']})"
            print(f"{header:<56s}|")
            print(f"+{'-'*55}+")
            line = (
                f"|  Industry: {cfg['industry']}   "
                f"Productivity: {cfg['productivity']}"
            )
            print(f"{line:<56s}|")
            line = (
                f"|  Current host: {host}   "
                f"Cumulative profit: {cumprof:.1f}"
            )
            print(f"{line:<56s}|")
            line = (
                f"|  Max scale: {cfg['max_scale']}   "
                f"Unit cost: {cfg['unit_cost']:.2f}"
            )
            print(f"{line:<56s}|")
            print(f"|{' '*55}|")
            line = f"|  SCALE (0-{cfg['max_scale']}): ______"
            print(f"{line:<56s}|")
            line = f"|  RELOCATE TO: ___________  (blank = stay)"
            print(f"{line:<56s}|")
            if self.phase >= 4:
                line = (
                    f"|  EXPORT? (Y/N, fixed cost "
                    f"{cfg['fixed_export_cost']:.0f}): _____"
                )
                print(f"{line:<56s}|")
            print(f"+{'-'*55}+")
            print()

    # ── Paper form generator ──────────────────────────────────────

    def print_decision_forms(self, round_num: int = None):
        """Print paper decision forms for each country."""
        rnd = (self.round_num + 1) if round_num is None else round_num
        names = list(self.countries.keys())

        for name in names:
            config = self.countries[name]
            print(f"+{'-'*55}+")
            print(f"|  ROUND {rnd} -- {name:40s}|")
            print(f"+{'-'*55}+")

            if self.phase == 1:
                line = f"|  Labor endowment: {config['labor']} workers"
                print(f"{line:<56s}|")
                prods = ", ".join(
                    f"{g}={config['productivity'][g]}" for g in self.goods
                )
                line = f"|  Productivity: {prods}"
                print(f"{line:<56s}|")
                print(f"|{' '*55}|")
                line = f"|  PRODUCTION (must sum to {config['labor']})"
                print(f"{line:<56s}|")
                for g in self.goods:
                    line = f"|    {g.capitalize():15s} workers: ________"
                    print(f"{line:<56s}|")
            else:
                line = f"|  Labor: {config['labor']}  Capital: {config['capital']}"
                print(f"{line:<56s}|")
                print(f"|{' '*55}|")
                line = f"|  PRODUCTION -- Labor (must sum to {config['labor']})"
                print(f"{line:<56s}|")
                for g in self.goods:
                    line = f"|    {g.capitalize():15s} workers: ________"
                    print(f"{line:<56s}|")
                line = f"|  PRODUCTION -- Capital (must sum to {config['capital']})"
                print(f"{line:<56s}|")
                for g in self.goods:
                    line = f"|    {g.capitalize():15s} capital: ________"
                    print(f"{line:<56s}|")

            print(f"|{' '*55}|")
            line = f"|  TARIFFS (0-100%)"
            print(f"{line:<56s}|")
            for partner in names:
                if partner != name:
                    for g in self.goods:
                        line = f"|    {g.capitalize()} from {partner}: _______%"
                        print(f"{line:<56s}|")

            print(f"|{' '*55}|")
            line = f"|  TRADE OFFERS"
            print(f"{line:<56s}|")
            line = f"|  We offer ____ units of ________"
            print(f"{line:<56s}|")
            line = f"|  to ____________"
            print(f"{line:<56s}|")
            line = f"|  in exchange for ____ units of ________"
            print(f"{line:<56s}|")
            line = f"|  (use back of sheet for additional offers)"
            print(f"{line:<56s}|")
            print(f"+{'-'*55}+")
            print()

    # ── Narrative prompt for local LLM ────────────────────────────

    def generate_narrative_prompt(self, round_num: int = None) -> str:
        """
        Generate a prompt for a local LLM (Llama, Gemma, etc.)
        to produce country briefings.

        Usage:
            prompt = sim.generate_narrative_prompt()
            # Feed to ollama, transformers pipeline, etc.
        """
        rd = (
            self.history[-1] if round_num is None
            else self.history[round_num - 1]
        )
        res = rd["results"]
        names = list(self.countries.keys())

        data_block = []
        for name in names:
            r = res[name]
            entry = f"  {name}:\n"
            entry += f"    Production: {r['production']}\n"
            entry += f"    Consumption: {r['consumption']}\n"
            entry += f"    Welfare: {r['welfare']:.1f}\n"
            entry += f"    Gains from trade: {r['gains_from_trade_pct']:+.1f}%\n"
            if "factor_prices" in r:
                fp = r["factor_prices"]
                entry += f"    Avg wage: {fp['avg_wage']:.2f}, "
                entry += f"Avg return to capital: {fp['avg_capital_return']:.2f}\n"
            data_block.append(entry)

        trade_block = (
            "\n".join(rd["trade_log"]) if rd["trade_log"]
            else "  No trades."
        )

        prompt = dedent(f"""\
        You are a narrator for a classroom simulation of the international
        political economy. Students play as countries making trade decisions.

        Write a short (100-150 word) briefing for EACH country after Round {rd['round']}.
        The tone should be like a news dispatch -- vivid, concrete, with
        political texture (mention lobbies, public opinion, strategic concerns).
        Reference the actual numbers but embed them in narrative.

        ROUND {rd['round']} DATA:

        {chr(10).join(data_block)}
        TRADES:
        {trade_block}

        Write one briefing per country, labeled clearly. Keep each under 150 words.
        """)
        return prompt

    # ── Save / load state ─────────────────────────────────────────

    def get_state(self) -> dict:
        """Export full simulation state as a JSON-serializable dict."""
        return {
            "countries": self.countries,
            "goods": self.goods,
            "phase": self.phase,
            "round_num": self.round_num,
            "history": self.history,
            # Phase 3+ state
            "firm_config": self.firm_config,
            "firms": self.firms,
            "world_prices": self.world_prices,
            "variety_rho": self.variety_rho,
            # Phase 4+ state
            "mnc_tax_revenue": self.mnc_tax_revenue,
            "reserve_currency_holder": self.reserve_currency_holder,
            # Phase 5+ state (monetary fields live inside `countries`)
            "monetary_unions": self.monetary_unions,
        }

    @classmethod
    def from_state(cls, state: dict) -> "IPESimulation":
        """Restore simulation from saved state. Backward compatible with
        saves that pre-date Phase 3/4/5."""
        sim = cls(state["countries"], state["goods"], state["phase"])
        sim.round_num = state["round_num"]
        sim.history = state["history"]
        # Phase 3+ fields (optional for backward compat)
        sim.firm_config = state.get("firm_config", {})
        sim.firms = state.get("firms", {})
        sim.world_prices = state.get("world_prices", dict(WORLD_PRICES))
        sim.variety_rho = state.get("variety_rho", VARIETY_RHO)
        # Phase 4+ fields
        sim.mnc_tax_revenue = state.get("mnc_tax_revenue", {})
        sim.reserve_currency_holder = state.get("reserve_currency_holder", None)
        # Phase 5+ fields
        sim.monetary_unions = state.get("monetary_unions", {})
        return sim
