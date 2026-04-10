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
    "Altera": {
        "labor": 150,
        "productivity": {"cloth": 2.0, "wine": 2.0},
        "description": "Large, balanced economy. No clear comparative advantage.",
    },
    "Brevia": {
        "labor": 60,
        "productivity": {"cloth": 4.0, "wine": 1.0},
        "description": "Small but highly productive in cloth. Textiles powerhouse.",
    },
    "Calida": {
        "labor": 100,
        "productivity": {"cloth": 1.0, "wine": 3.0},
        "description": "Medium economy with strong wine sector.",
    },
    "Deltia": {
        "labor": 120,
        "productivity": {"cloth": 3.0, "wine": 1.5},
        "description": "Large economy, leans toward cloth but diversified.",
    },
    "Eridia": {
        "labor": 50,
        "productivity": {"cloth": 3.5, "wine": 3.5},
        "description": "Tiny but efficient. Absolute advantage in both goods, small market.",
    },
    "Fortuna": {
        "labor": 100,
        "productivity": {"cloth": 1.0, "wine": 1.5},
        "description": "Medium economy, low productivity. Comparative advantage in wine.",
    },
}

PHASE1_GOODS = ["cloth", "wine"]

# Phase 2: Heckscher-Ohlin — add capital, factor intensities, third good
# Cloth: labor-intensive | Wine: moderate | Machinery: capital-intensive

PHASE2_COUNTRIES = {
    "Altera": {
        "labor": 150, "capital": 150,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.0, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.0, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Balanced factor endowments. Can compete in any sector.",
    },
    "Brevia": {
        "labor": 60, "capital": 25,
        "tech": {
            "cloth":     {"tfp": 1.2, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 0.8, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 0.7, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Labor-abundant, capital-scarce. Strong in textiles.",
    },
    "Calida": {
        "labor": 100, "capital": 80,
        "tech": {
            "cloth":     {"tfp": 0.8, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.3, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 0.9, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Moderate endowments. TFP advantage in wine.",
    },
    "Deltia": {
        "labor": 120, "capital": 200,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 0.9, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.2, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Capital-abundant industrial power. Machinery is natural strength.",
    },
    "Eridia": {
        "labor": 50, "capital": 120,
        "tech": {
            "cloth":     {"tfp": 1.0, "labor_share": 0.70, "capital_share": 0.30},
            "wine":      {"tfp": 1.0, "labor_share": 0.55, "capital_share": 0.45},
            "machinery": {"tfp": 1.1, "labor_share": 0.25, "capital_share": 0.75},
        },
        "description": "Small, very capital-rich. High-tech potential but limited labor.",
    },
    "Fortuna": {
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

    # ── Core round logic ──────────────────────────────────────────

    def run_round(self, decisions: dict, trades: list) -> dict:
        """
        Execute one round of the simulation.

        Parameters
        ----------
        decisions : dict
            Phase 1 format:
                {country: {"production": {"cloth": 60, "wine": 40},
                           "tariffs": {partner: {good: rate, ...}, ...}}}
            Phase 2 format:
                {country: {"production": {"labor": {"cloth": 40, ...},
                                          "capital": {"cloth": 20, ...}},
                           "tariffs": {partner: {good: rate, ...}, ...}}}

            Tariffs are optional (default 0). Rates are fractions (0.1 = 10%).

        trades : list of tuples
            Each tuple: (exporter, importer, good_out, qty_out, good_in, qty_in)
            These are the AGREED trades (instructor mediates matching).

        Returns
        -------
        dict with round results.
        """
        self.round_num += 1
        errors = self._validate_decisions(decisions)
        if errors:
            self.round_num -= 1
            raise ValueError("Decision validation failed:\n" + "\n".join(errors))

        # Step 1: Production
        production = self._compute_production(decisions)

        # Step 2: Initialize consumption = production
        consumption = {c: dict(p) for c, p in production.items()}

        # Step 3: Execute trades, apply tariffs
        trade_log, tariff_revenue = self._execute_trades(
            consumption, decisions, trades
        )

        # Step 4: Compute welfare and other metrics
        results = {}
        for name in self.countries:
            welfare = self._utility(consumption[name])
            autarky_welfare = self._utility(production[name])
            gains_pct = (
                ((welfare - autarky_welfare) / autarky_welfare * 100)
                if autarky_welfare > 0 else 0.0
            )

            results[name] = {
                "production": production[name],
                "consumption": consumption[name],
                "welfare": welfare,
                "autarky_welfare": autarky_welfare,
                "gains_from_trade_pct": gains_pct,
                "tariff_revenue": tariff_revenue[name],
            }

            # Phase 2: factor prices
            if self.phase == 2:
                results[name]["factor_prices"] = self._compute_factor_prices(
                    name, decisions[name], production[name]
                )

        round_result = {
            "round": self.round_num,
            "phase": self.phase,
            "results": results,
            "trade_log": trade_log,
        }
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

    # ── Trade execution ───────────────────────────────────────────

    def _execute_trades(self, consumption, decisions, trades):
        """
        Execute trades. Tariffs destroy a fraction of imports (deadweight loss).
        Returns trade log and tariff revenue tracker.
        """
        trade_log = []
        tariff_revenue = {c: {g: 0.0 for g in self.goods} for c in self.countries}

        for trade in trades:
            exporter, importer, good_out, qty_out, good_in, qty_in = trade

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

            # Look up tariffs (default 0)
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

            # Tariff friction: destroys a fraction of imports
            received_by_importer = qty_out * (1 - t_importer)
            received_by_exporter = qty_in * (1 - t_exporter)
            destroyed_at_importer = qty_out * t_importer
            destroyed_at_exporter = qty_in * t_exporter

            # Update consumption
            consumption[exporter][good_out] -= qty_out
            consumption[exporter][good_in] += received_by_exporter
            consumption[importer][good_in] -= qty_in
            consumption[importer][good_out] += received_by_importer

            # Track tariff losses
            tariff_revenue[importer][good_out] += destroyed_at_importer
            tariff_revenue[exporter][good_in] += destroyed_at_exporter

            # Log
            tariff_str = ""
            if t_importer > 0 or t_exporter > 0:
                tariff_str = (
                    f" [tariffs: {t_importer:.0%} on {good_out}, "
                    f"{t_exporter:.0%} on {good_in}]"
                )
            trade_log.append(
                f"  {exporter} -> {importer}: "
                f"{qty_out:.0f} {good_out} for {qty_in:.0f} {good_in}"
                f"{tariff_str}"
            )

        return trade_log, tariff_revenue

    # ── Welfare ───────────────────────────────────────────────────

    def _utility(self, consumption_bundle: dict) -> float:
        """Cobb-Douglas utility with equal weights: U = prod(c_j)^(1/J)."""
        J = len(self.goods)
        u = 1.0
        for good in self.goods:
            c = consumption_bundle.get(good, 0.0)
            if c > 0:
                u *= c ** (1.0 / J)
            else:
                return 0.0
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

    # ── Instructor shock injection ────────────────────────────────

    def inject_shock(self, description: str, changes: dict):
        """
        Modify country parameters between rounds.

        Examples:
            sim.inject_shock("Oil crisis hits Calida", {
                "Calida": {"productivity": {"wine": 1.5}}  # Phase 1
            })
            sim.inject_shock("Capital flight from Deltia", {
                "Deltia": {"capital": 150}  # Phase 2
            })
        """
        print(f"\n{'':=<55}")
        print(f"  SHOCK: {description}")
        print(f"{'':=<55}")
        for country, params in changes.items():
            for key, value in params.items():
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        old = self.countries[country][key][subkey]
                        self.countries[country][key][subkey] = subval
                        print(f"  {country}: {key}.{subkey}: {old} -> {subval}")
                else:
                    old = self.countries[country][key]
                    self.countries[country][key] = value
                    print(f"  {country}: {key}: {old} -> {value}")
        print()

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
            + f"{'Welfare':>12s}{'vs Autarky':>12s}"
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
            sign = "+" if gains >= 0 else ""
            row += f"{sign}{gains:10.1f}%"
            print(row)

        # Tariff losses
        any_tariffs = any(
            any(v > 0 for v in res[n]["tariff_revenue"].values())
            for n in names
        )
        if any_tariffs:
            print(f"\n  TARIFF LOSSES (goods destroyed)")
            header = f"  {'':16s}" + "".join(f"{g:>12s}" for g in goods)
            print(header)
            for name in names:
                tr = res[name]["tariff_revenue"]
                if any(v > 0 for v in tr.values()):
                    row = f"  {name:16s}"
                    for g in goods:
                        row += f"{tr.get(g, 0):12.1f}"
                    print(row)

        # Factor prices (Phase 2)
        if phase == 2:
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

        print(f"\n{'':=<65}\n")

    # ── Visualization ─────────────────────────────────────────────

    def plot_welfare(self, figsize=(10, 5)):
        """Line chart of welfare across all rounds."""
        if not self.history:
            print("No rounds to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        names = list(self.countries.keys())
        rounds = [h["round"] for h in self.history]

        for name in names:
            welfares = [h["results"][name]["welfare"] for h in self.history]
            ax.plot(rounds, welfares, marker="o", linewidth=2, label=name)

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Welfare (utility index)", fontsize=12)
        ax.set_title("Welfare Over Time", fontsize=14, fontweight="bold")
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
        """Export full simulation state as a serializable dict."""
        return {
            "countries": self.countries,
            "goods": self.goods,
            "phase": self.phase,
            "round_num": self.round_num,
            "history": self.history,
        }

    @classmethod
    def from_state(cls, state: dict) -> "IPESimulation":
        """Restore simulation from saved state."""
        sim = cls(state["countries"], state["goods"], state["phase"])
        sim.round_num = state["round_num"]
        sim.history = state["history"]
        return sim
