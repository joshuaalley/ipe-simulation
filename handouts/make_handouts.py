"""
Generate printable classroom handouts (country briefs + decision forms) as
LaTeX, driven by the live constants in engine.py. Re-run after retuning any
endowment, firm, or parameter and the PDFs stay in sync.

    cd ipe-simulation/handouts
    python make_handouts.py          # all six countries, full firm roster

    # ...or match a smaller class:
    python make_handouts.py --countries Sabine Bosque Llano Trinity
    # then compile (twice is unnecessary; no cross-refs):
    #   pdflatex -interaction=nonstopmode <file>.tex

Outputs (this directory):
    country-briefs.tex          one page per country (identity + endowments)
    forms-phase1-ricardo.tex    Phase 1 decision form, per country
    forms-phase2plus-trade.tex  Phase 2+ trade form (production/tariffs/trades)
    forms-firms.tex             MNC owner forms (Phase 3+; export line = Phase 4+)
    forms-finance.tex           Monetary/debt/institutions add-on (Phase 5-7)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # engine imports pyplot; keep it headless

# import engine from the parent simulation directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Generate printable country briefs and decision forms.")
    ap.add_argument(
        "--countries", nargs="+", metavar="NAME", default=None,
        help="subset of countries to print for (default: all six). "
             "Must match the country set you run the simulation with.")
    ap.add_argument(
        "--firms", type=int, default=None, metavar="N",
        help="print fewer MNC forms than the default of two per country "
             "(more than that is trimmed back to two per country).")
    ap.add_argument(
        "--challenge-rule", choices=engine.CHALLENGE_RULES, default=None,
        help="how a challenge for the reserve currency (and the hegemony) is "
             "decided, as the finance and capstone forms state it. Match "
             f"sim.challenge_rule; default {engine.CHALLENGE_RULE!r}.")
    return ap.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else argparse.Namespace(
    countries=None, firms=None, challenge_rule=None)

P1_GOODS = engine.PHASE1_GOODS
P2_GOODS = engine.PHASE2_GOODS
WORLD_PRICES = engine.WORLD_PRICES
MONEY_CHOICES = engine.PHASE5_MONEY_GROWTH_CHOICES

if ARGS.countries:
    unknown = [c for c in ARGS.countries if c not in engine.PHASE1_COUNTRIES]
    if unknown:
        raise SystemExit(
            f"Unknown country/countries: {unknown}\n"
            f"Choose from: {sorted(engine.PHASE1_COUNTRIES)}")
    NAMES = list(ARGS.countries)
    # Rehome/trim the firm roster to match, exactly as the engine requires.
    FIRMS = engine.build_firm_roster(NAMES, n_firms=ARGS.firms, verbose=False)
else:
    NAMES = list(engine.PHASE1_COUNTRIES.keys())
    FIRMS = engine.build_firm_roster(NAMES, n_firms=ARGS.firms, verbose=False)

P1 = {n: engine.PHASE1_COUNTRIES[n] for n in NAMES}
P2 = {n: engine.PHASE2_COUNTRIES[n] for n in NAMES}


# ── LaTeX helpers ─────────────────────────────────────────────────────

def esc(s):
    """Escape LaTeX specials that appear in our data (mostly & and %)."""
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#"))


PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[letterpaper,margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage{tikz}
\usepackage{xcolor}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\newcommand{\blank}[1]{\underline{\hspace{#1}}}
\newcommand{\boxx}{$\square$}
\newcommand{\hr}{\par\vspace{2pt}\noindent\rule{\linewidth}{0.4pt}\par\vspace{2pt}}
\pagestyle{empty}
\begin{document}
"""

FOOTER = r"""\end{document}
"""


def write_tex(filename, body):
    path = os.path.join(HERE, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(PREAMBLE + body + FOOTER)
    return path


# ── Country briefs ────────────────────────────────────────────────────

def home_firms(name):
    """Firms whose default host is this country (Phase 3 starting roster)."""
    out = []
    for fid, cfg in FIRMS.items():
        if cfg["default_host"] == name:
            out.append(f"{cfg['variety']} ({fid}, {cfg['industry']})")
    return out


def country_brief_page(name):
    p1 = P1[name]
    p2 = P2[name]
    desc = p1.get("description", "")

    # Phase 1 numbers
    prod = p1["productivity"]
    cloth, wine = P1_GOODS[0], P1_GOODS[1]
    opp = prod[cloth] / prod[wine]
    prod_line = ", ".join(f"{esc(g)} = {prod[g]:.1f}/worker" for g in P1_GOODS)

    # Phase 2 numbers
    L, K = p2["labor"], p2["capital"]
    kl = K / L
    tech = p2["tech"]
    tfp_rows = "\n".join(
        rf"    {esc(g)} & {tech[g]['tfp']:.1f} & "
        rf"{tech[g]['labor_share']:.2f} & {tech[g]['capital_share']:.2f} \\"
        for g in P2_GOODS
    )

    hf = home_firms(name)
    hf_line = ", ".join(esc(x) for x in hf) if hf else "(none)"

    return rf"""
{{\Large\bfseries Country Brief: {esc(name)}}}\par
\textit{{{esc(desc)}}}
\hr

{{\bfseries Phase 1 --- Ricardo}} \hfill (one factor: labor)\par
Labor force: \textbf{{{p1['labor']}}} workers\par
Productivity: {prod_line}\par
Your own opportunity cost: 1 {esc(wine)} costs \textbf{{{opp:.2f}}} {esc(cloth)} foregone.\par
\textit{{What can you make more cheaply than your trading partners?}}

\vspace{{6pt}}
{{\bfseries Phase 2 --- Heckscher--Ohlin}} \hfill (two factors: labor + capital)\par
Labor: \textbf{{{L}}} \quad Capital: \textbf{{{K}}} \quad K/L ratio: \textbf{{{kl:.2f}}}\par
\vspace{{2pt}}
\begin{{tabular}}{{lccc}}
\toprule
Good & Your TFP & Labor share & Capital share \\
\midrule
{tfp_rows}
\bottomrule
\end{{tabular}}\par
\vspace{{2pt}}
\textit{{Factor intensities are the same for everyone: cloth is labor-intensive,
machinery is capital-intensive, wine is in between.}}

\vspace{{6pt}}
{{\bfseries Later in the game}}\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
  \item \textbf{{Phase 3+ (MNCs):}} firms that start on your soil --- {hf_line}.
    You will also \emph{{own}} a firm on someone else's soil: its profits are
    yours, not its host's.
  \item \textbf{{Phase 5 (money):}} you run your own currency --- choose a regime and live with the trilemma.
  \item \textbf{{Phase 6 (debt):}} you may borrow to spend now or to invest in projects --- and you may default.
  \item \textbf{{Phase 7 (institutions):}} join the WTO, bind your tariffs, and decide whether to back the system.
\end{{itemize}}
"""


def build_country_briefs():
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(country_brief_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("country-briefs.tex", "\n".join(pages))


# ── Shared form components ────────────────────────────────────────────

def tariff_block(name):
    lines = [r"{\bfseries Tariffs} (0--100\%, on imports, per partner per good):\par",
             r"\begin{itemize}[nosep,leftmargin=1.4em]"]
    for partner in NAMES:
        if partner == name:
            continue
        goods_bits = " \\quad ".join(
            rf"{esc(g)}: \blank{{1.4cm}}\%" for g in current_goods
        )
        lines.append(rf"  \item from \textbf{{{esc(partner)}}}: \quad {goods_bits}")
    lines.append(r"\end{itemize}")
    return "\n".join(lines)


def trade_block(more=r"\textit{(Use the back of the sheet for more offers.)}"):
    offer = (r"  \item We give \blank{1.8cm} units of \blank{2.2cm} "
             r"to \blank{2.2cm};\\[2pt] in return: \blank{1.8cm} units of \blank{2.2cm}.")
    return (r"{\bfseries Trade offers} (negotiate first, then commit):\par" + "\n"
            r"\begin{itemize}[itemsep=4pt,leftmargin=1.4em]" + "\n"
            + offer + "\n" + offer + "\n" + offer + "\n"
            r"\end{itemize}" + "\n" + more)


current_goods = P1_GOODS  # rebound per builder


def politics_choices():
    """The compensation and side-payment lines, without the note."""
    cap = _pct(engine.COMPENSATION_MAX_SHARE)
    return (
        r"\begin{itemize}[nosep,leftmargin=1.4em]" "\n"
        rf"  \item This round we compensate: \blank{{1.4cm}}\% of our consumption "
        rf"\quad \textit{{(0--{cap}\%)}}" "\n"
        r"  \item Side payment to another country: \blank{1.6cm} units of "
        r"\blank{2cm} to \blank{2.2cm}." "\n"
        r"\end{itemize}"
    )


def compensation_note():
    dead = _pct(engine.COMPENSATION_DEADWEIGHT * 0.10)      # cost of a 10% payment
    buys = f"{engine.APPROVAL_COMPENSATION * 0.10:.1f}"     # what it buys
    return (rf"Compensating 10\% buys about +{buys} approval a round and costs "
            rf"{dead}\% of your welfare --- roughly what a 20\% tariff buys, for less, "
            rf"and your trade stays open. A side payment does the same for whoever "
            rf"receives it (only the net counts) --- but if you pay each other, it is "
            rf"a trade, tariffs and all.")


def politics_block():
    """Compensation + side payments: the alternatives to closing the border."""
    return (
        r"{\bfseries Compensation} --- buy off the groups trade displaced, "
        r"instead of taxing imports:\par" "\n"
        + politics_choices() + "\n"
        + r"\textit{" + compensation_note() + "}"
    )


def mnc_tax_choice(name):
    top = _pct(engine.MNC_TAX_MAX)
    low = _pct(engine.MNC_TAX_MIN)
    return (rf"{{\bfseries MNC tax}} (Phase 3+) --- on foreign-owned firms in "
            rf"{esc(name)}: \blank{{1.4cm}}\% of their revenue "
            rf"\quad \textit{{({low}--{top}\%)}}\par")


def mnc_tax_note():
    return (r"Blank keeps last round's rate. What you collect adds to your "
            r"welfare. Set it too high and the owners can move their firm out, "
            r"taking its output with it.")


def mnc_tax_block(name):
    """The MNC tax: one rate on foreign-owned firms' revenue (Phase 3+)."""
    return mnc_tax_choice(name) + "\n" + r"\textit{" + mnc_tax_note() + "}"


# ── Phase 1 decision form ─────────────────────────────────────────────

def phase1_form_page(name):
    p1 = P1[name]
    prod = p1["productivity"]
    prod_line = ", ".join(f"{esc(g)} = {prod[g]:.1f}/worker" for g in P1_GOODS)
    prod_lines = "\n".join(
        rf"  \item {esc(g).capitalize()}: \blank{{2.5cm}} workers" for g in P1_GOODS
    )
    return rf"""
{{\large\bfseries ROUND \blank{{1cm}} --- {esc(name)}}} \hfill (Phase 1: Ricardo)\par
Labor endowment: \textbf{{{p1['labor']}}} workers \quad|\quad Productivity: {prod_line}
\hr

{{\bfseries Production}} --- allocate your labor (must sum to \textbf{{{p1['labor']}}}):\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
{prod_lines}
\end{{itemize}}

\vspace{{4pt}}
{tariff_block(name)}

\vspace{{4pt}}
{politics_block()}

\vspace{{4pt}}
{trade_block()}
"""


def build_phase1_forms():
    global current_goods
    current_goods = P1_GOODS
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(phase1_form_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("forms-phase1-ricardo.tex", "\n".join(pages))


# ── Phase 2+ trade form ───────────────────────────────────────────────

def phase2_form_page(name):
    p2 = P2[name]
    L, K = p2["labor"], p2["capital"]
    labor_lines = "\n".join(
        rf"  \item {esc(g).capitalize()}: \blank{{2.5cm}} workers" for g in P2_GOODS
    )
    cap_lines = "\n".join(
        rf"  \item {esc(g).capitalize()}: \blank{{2.5cm}} capital" for g in P2_GOODS
    )
    return rf"""
{{\large\bfseries ROUND \blank{{1cm}} --- {esc(name)}}} \hfill (Phase 2+: trade form)\par
Labor: \textbf{{{L}}} \quad|\quad Capital: \textbf{{{K}}}
\hr

{{\bfseries Production --- Labor}} (must sum to \textbf{{{L}}}):\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
{labor_lines}
\end{{itemize}}

{{\bfseries Production --- Capital}} (must sum to \textbf{{{K}}}):\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
{cap_lines}
\end{{itemize}}
\textit{{A good needs \emph{{both}} labor and capital to be produced.}}

\vspace{{4pt}}
{tariff_block(name)}

\vspace{{4pt}}
{politics_block()}

\vspace{{4pt}}
{mnc_tax_block(name)}

\vspace{{4pt}}
{trade_block()}
"""


def build_phase2_forms():
    global current_goods
    current_goods = P2_GOODS
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(phase2_form_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("forms-phase2plus-trade.tex", "\n".join(pages))


# ── Firm (MNC) forms ──────────────────────────────────────────────────
#
# One page per firm: the decision slip on top, then a worked example in that
# firm's own numbers at a scale nobody picks (EXAMPLE_SCALE), so the method is
# on the page but the decision isn't.

EXAMPLE_SCALE = 23


def _money(x):
    """Round half-up to cents, the way students round by hand."""
    from decimal import Decimal, ROUND_HALF_UP
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _units(x):
    from decimal import Decimal, ROUND_HALF_UP
    return Decimal(str(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def firm_slip(fid):
    cfg = FIRMS[fid]
    price = engine.WORLD_PRICES[cfg["industry"]]
    return rf"""
\noindent\fbox{{\begin{{minipage}}{{0.97\linewidth}}
\vspace{{3pt}}
{{\large\bfseries ROUND \blank{{1.2cm}} --- FIRM {esc(fid)}: {esc(cfg['variety'])}}}
\hfill Owner(s): \blank{{4.8cm}}\par
\vspace{{3pt}}
Industry: \textbf{{{esc(cfg['industry'])}}} \quad
Productivity: \textbf{{{cfg['productivity']:.1f}}} \quad
Starting host: \textbf{{{esc(cfg['default_host'])}}}\par
Price per unit sold: \textbf{{{price:.2f}}} \quad
Unit cost: \textbf{{{cfg['unit_cost']:.2f}}} \quad
Max scale: \textbf{{{cfg['max_scale']:.0f}}} \quad
Export fixed cost: \textbf{{{cfg['fixed_export_cost']:.0f}}}\par
\vspace{{8pt}}
SCALE (0--{cfg['max_scale']:.0f}): \blank{{2.2cm}} \qquad
RELOCATE TO: \blank{{3.4cm}} \textit{{(blank = stay)}}\par
{{\small\itshape Relocate anywhere except your own country --- no re-shoring.}}\par
\vspace{{6pt}}
EXPORT this round? \quad \boxx\ Yes \quad \boxx\ No
\hfill\textit{{(Phase 4+; pays the fixed cost)}}\par
\vspace{{3pt}}
\end{{minipage}}}}
"""


def firm_example(fid):
    """Stay home vs export at EXAMPLE_SCALE, in this firm's own numbers."""
    cfg = FIRMS[fid]
    k, phi = EXAMPLE_SCALE, cfg["productivity"]
    price, uc = engine.WORLD_PRICES[cfg["industry"]], cfg["unit_cost"]
    fx, gain = cfg["fixed_export_cost"], engine.EXPORT_MARKET_GAIN
    units = _units(k * phi)
    rev_home = _money(units * _money(price))
    rev_exp = _money(rev_home * (1 + _money(gain)))
    cost = _money(k * uc)
    fixed = _money(fx)
    prof_home = rev_home - cost
    prof_exp = rev_exp - cost - fixed
    extra = rev_exp - rev_home
    diff = prof_exp - prof_home
    pays = diff > 0
    verdict = (rf"at scale {k}, exporting \textbf{{adds {diff:.2f}}}: the extra "
               rf"revenue ({extra:.2f}) covers the fixed cost ({fixed:.2f})."
               if pays else
               rf"at scale {k}, exporting \textbf{{loses {-diff:.2f}}}: the extra "
               rf"revenue ({extra:.2f}) does not cover the fixed cost ({fixed:.2f}). "
               rf"Stay home.")
    pct = f"{gain * 100:.0f}"
    local = f"{engine.FIRM_LOCAL_SHARE * 100:.0f}"
    return rf"""
\vspace{{10pt}}
\noindent\tikz\draw[dashed,gray] (0,0) -- (0.99\linewidth,0);\par
\vspace{{-2pt}}
{{\footnotesize\color{{gray}} Hand in the slip above; keep this half.}}\par
\vspace{{8pt}}
{{\large\bfseries How the numbers work --- a worked example at scale {k}}}\par
\textit{{Scale {k} is an illustration, not a recommendation. Redo the table at the
scale you actually choose.}}

\vspace{{6pt}}
\begin{{center}}
\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabular}}{{lrr}}
\toprule
 & \textbf{{Stay home}} & \textbf{{Export}} (premium {pct}\%) \\
\midrule
Units made: scale $\times$ productivity & {k} $\times$ {phi:.1f} $=$ {units} & {units} \\
Revenue: units $\times$ price & {units} $\times$ {price:.2f} $=$ {rev_home}
  & {rev_home} $\times$ {1 + gain:.2f} $=$ {rev_exp} \\
Production cost: scale $\times$ unit cost & {k} $\times$ {uc:.2f} $=$ {cost} & {cost} \\
Export fixed cost & --- & {fixed} \\
\midrule
\textbf{{Profit}} & \textbf{{{prof_home}}} & \textbf{{{prof_exp}}} \\
\bottomrule
\end{{tabular}}
\end{{center}}

\textbf{{Verdict:}} {verdict}

\vspace{{4pt}}
\begin{{itemize}}[nosep,leftmargin=1.2em]
  \item \textbf{{The rule:}} export only if units $\times$ price $\times$ premium
    $>$ the fixed export cost. The fixed cost is the same whether you ship 5 units
    or 50, so exporting needs volume.
  \item \textbf{{The premium}} is on the board each round. {pct}\% is the most it
    can be; tariffs other countries put on your good from your host shrink it.
  \item \textbf{{Your local sales}} --- {local}\% of your units --- count toward your
    \emph{{host's}} welfare, export or not; the rest sells on world markets.
    \textbf{{Your profit}} counts every unit, and it is yours alone.
  \item \textbf{{Your host's MNC tax}} takes its \% of your revenue (premium
    included), so a lower-tax host keeps more of your profit --- but moving
    means producing nothing that round. From Phase 5 your profit is worth what
    your host's currency is worth, unless you export: exports are paid in world
    prices.
\end{{itemize}}
"""


def build_firm_forms():
    intro = (r"{\large\bfseries MNC Decision Form} \hfill (Phase 3+)\par" + "\n"
             r"You own this firm --- alone or with a partner --- even though it "
             r"sits in another country. Each round: choose how much to produce "
             r"(\emph{scale}), whether to \emph{relocate} to a new host, and "
             r"(Phase 4+) whether to pay the fixed cost to \emph{export}. "
             r"It can move anywhere except your own country. "
             r"Productivity tiers: HIGH 1.3, MED 1.0, LOW 0.7." + "\n"
             r"\vspace{6pt}" + "\n")
    pages = [intro + firm_slip(fid) + firm_example(fid) for fid in FIRMS]
    return write_tex("forms-firms.tex", "\n\\newpage\n".join(pages))


# ── Finance add-on (Phases 5-6) and the capstone form (Phase 7) ───────
#
# The money, debt and institutions parts are written once and used twice. The
# finance add-on staples to the trade form in Phases 5 and 6. The capstone form
# puts every decision of a Phase 7 round on the front of one sheet and the rules,
# with their numbers, on the back. Every number comes from engine.py.

def _pct(x):
    """0.15 -> '15' (a rule's number, straight from the engine)."""
    return f"{x * 100:.0f}"


def _rules(items, size=r"\small"):
    """A small description list of rules: [(label, text), ...]."""
    body = "\n".join(rf"  \item[{label}] {text}" for label, text in items)
    return ("{" + size + "\n"
            r"\begin{description}[nosep,style=sameline,leftmargin=5.6em,"
            r"font=\normalfont\bfseries]" "\n" + body + "\n"
            r"\end{description}}")


def money_choices(phase7=False):
    pct = " \\quad ".join(rf"\boxx\ {'$-$' if g < 0 else ''}{abs(g) * 100:.0f}\%"
                          for g in MONEY_CHOICES)
    # From Phase 7 the hegemon is the reserve-currency holder: one prize.
    prize = "Reserve currency and hegemony" if phase7 else "Reserve currency"
    return (r"\begin{itemize}[nosep,leftmargin=1.4em]" "\n"
            r"  \item Exchange rate: \quad \boxx\ Peg \quad \boxx\ Float" "\n"
            r"  \item Capital account: \quad \boxx\ Open \quad \boxx\ Controls" "\n"
            rf"  \item Money growth: \quad {pct}\newline \textit{{(0\% = follow the anchor)}}" "\n"
            rf"  \item {prize}: \quad \boxx\ No challenge \quad \boxx\ We challenge"
            r" \quad \boxx\ We back \blank{2.2cm}" "\n"
            r"\end{itemize}")


def _challenge_rule():
    n = len(NAMES)
    if (getattr(ARGS, "challenge_rule", None) or engine.CHALLENGE_RULE) == "welfare":
        return ("the challenger and the countries backing it have more cumulative "
                "welfare than everyone else")
    return (f"the challenger and the countries backing it are a majority: "
            f"{n // 2 + 1} of the {n} countries")


def money_rules(size=r"\small", phase7=False):
    fric, cut = _pct(engine.BASE_FX_FRICTION), _pct(engine.CONTROLS_FIRM_CUT)
    stim = f"{engine.STIMULUS_PER_POINT:g}"
    steps = [f"$-${_pct(-g)}" for g in sorted(MONEY_CHOICES, reverse=True) if g < 0]
    tight = ", ".join(steps[:-1]) + " or " + steps[-1] if len(steps) > 1 else steps[0]
    half = {0.5: "half", 1.0: "all"}.get(engine.WEAK_FX_IMPORT_COST,
                                          f"{engine.WEAK_FX_IMPORT_COST:g} times")
    eg = _pct(engine.WEAK_FX_IMPORT_COST * 0.10)
    drag = f"{engine.WEAK_FX_WELFARE_DRAG * 10:g}"          # % per 10 points
    xg = f"{engine.WEAK_FX_EXPORT_GAIN * 10:g}"            # % per 10 points, all output exported
    xeg = f"{engine.WEAK_FX_EXPORT_GAIN * 0.10 * 0.5 * 100:g}"   # FX 0.90, half exported
    return _rules([
        ("Peg", rf"No currency friction on trades with other pegged currencies; other "
                rf"cross-currency trades lose {fric}\%. Trades with the reserve country carry "
                rf"no currency friction. An open peg is what speculators attack. A peg is tied "
                rf"to the reserve currency and moves with it: when the reserve country prints, "
                rf"tightens, or its currency crashes, yours moves the same way."),
        ("Controls", rf"Foreign firms in your country produce {cut}\% less, and (Phase 6) "
                     rf"you cannot borrow abroad. Speculators cannot run on a closed account."),
        ("Printing", rf"+{stim}\% welfare this round for each point printed, times your FX "
                     rf"index. Your FX index falls by the same \% for good. A weak currency "
                     rf"shrinks every import you receive by {half} the drop, and costs {drag}\% "
                     rf"of welfare every round for each 10 points below 1.00 (FX 0.90 $\to$ "
                     rf"imports $-${eg}\%, welfare $-${drag}\%). But it makes your exports cheap "
                     rf"abroad: welfare rises {xg}\% a round for each 10 points below 1.00, times "
                     rf"the share of your own output you export outside your union (FX 0.90, "
                     rf"half exported $\to$ +{xeg}\%)."),
        ("Tightening", rf"{tight}\%: your FX index rises by that \% for good (it can pass "
                       rf"1.00), and welfare falls {stim}\% this round for each point."),
        ("Challenge", (rf"The hegemon is whoever holds the reserve currency, so one "
                       rf"challenge decides both. It wins if {_challenge_rule()}, is settled "
                       rf"after the round, and the winner holds the reserve currency and the "
                       rf"hegemony from the next one." if phase7 else
                       rf"A challenge for the reserve currency wins if {_challenge_rule()}. "
                       rf"It is settled after the round, and the winner holds the reserve "
                       rf"currency from the next one (from Phase 7, the hegemony too).")),
    ], size)


def debt_choices():
    return (r"\begin{itemize}[nosep,leftmargin=1.4em]" "\n"
            r"  \item Borrow to spend: \blank{2.2cm} \quad Borrow to invest: \blank{2.2cm}"
            r" \quad Repay: \blank{2.2cm}" "\n"
            r"  \item Default on the debt? \quad \boxx\ Yes \quad \boxx\ No" "\n"
            r"\end{itemize}")


def debt_rules(size=r"\small"):
    dcost = f"{engine.DEBT_DEFAULT_COST:g}"
    cap, base = _pct(engine.BORROW_CAP_SHARE), _pct(engine.DEBT_BASE_RATE)
    prem, ban = _pct(engine.DEBT_RISK_PREMIUM), engine.DEBT_DEFAULT_BAN_ROUNDS
    dfric = _pct(engine.DEBT_DEFAULT_FRICTION)
    ret = _pct(engine.PROJECT_RETURN)
    ret10 = f"{engine.PROJECT_RETURN * 10:g}"               # what a project costing 10 pays
    relief = {0.5: "half your debt"}.get(engine.IMF_DEBT_RELIEF,
                                         rf"{_pct(engine.IMF_DEBT_RELIEF)}\% of your debt")
    when = ("this round and the next" if engine.IMF_AUSTERITY_ROUNDS == 2
            else f"for {engine.IMF_AUSTERITY_ROUNDS} rounds")
    return _rules([
        ("Borrow", rf"Up to {cap}\% of your consumption a round, spending and investing "
                   rf"together. Not under capital controls."),
        ("Spend", r"Welfare rises by the same share of consumption this round."),
        ("Invest", rf"Builds a project instead: nothing now, then {ret}\% of its cost every "
                   rf"round from the next one to the end of the game, in your own money (a "
                   rf"project costing 10 pays {ret10} a round)."),
        ("Interest", rf"{base}\% + {prem}\% $\times$ (debt $\div$ consumption), every round. "
                     rf"Debt is owed in the reserve currency, so what counts is your currency "
                     rf"against it: payments are divided by (your FX index $\div$ the "
                     rf"reserve's). Yours at 0.70 while the reserve's holds at 1.00 $\to$ 43\% "
                     rf"heavier. A peg falls with the reserve, so its debt gets no heavier; if "
                     rf"the reserve falls and yours holds, your debt gets lighter."),
        ("Default", rf"The debt disappears, but welfare falls by {dcost} $\times$ (debt "
                    rf"$\div$ consumption) that round --- more than repaying would cost at par; "
                    rf"no borrowing for {ban} rounds and +{dfric}\% friction on your trades "
                    rf"meanwhile. It only pays after a big fall in your currency against the "
                    rf"reserve's."),
        ("IMF", rf"Ask the instructor before the forms are in: {relief} is refinanced away "
                rf"now, and your welfare is {_pct(engine.IMF_AUSTERITY)}\% lower {when}."),
        ("Last round", r"Everything still owed is repaid."),
    ], size)


def institutions_choices():
    binds = r" \quad ".join(rf"{esc(g)} \blank{{1.3cm}}\%" for g in P2_GOODS)
    return (r"\begin{itemize}[nosep,leftmargin=1.4em]" "\n"
            r"  \item WTO: \quad \boxx\ Join / stay in \quad \boxx\ Stay out / leave" "\n"
            r"  \item Bindings, the most we promise to charge anyone (blank = no promise):"
            r"\newline " + binds + "\n"
            r"  \item \textit{Hegemon only:} \quad \boxx\ Provide the public good"
            r" \quad \boxx\ Withhold it" "\n"
            r"\end{itemize}")


def institutions_rules(size=r"\small"):
    base, div = engine.BASE_FX_FRICTION, engine.WTO_DIVIDEND
    w, cost = engine.HEGEMON_WITHHOLD_PENALTY, engine.HEGEMON_PROVISION_COST
    # _global_crisis_factor: provision halves the cut; WTO breadth cuts up to half again
    worst, best = 20, 20 * 0.5 * (1 - 0.5 * 1.0)
    return _rules([
        ("WTO", rf"Trades between two members lose {_pct(div)}\% less of their goods to "
                rf"friction, as long as neither breaks a binding that round (half that during "
                rf"a default ban): an ordinary cross-currency trade loses "
                rf"{_pct(max(0.0, base - div))}\% instead of {_pct(base)}\%, or "
                rf"{_pct(base + w - div)}\% instead of {_pct(base + w)}\% while the hegemon "
                rf"withholds. Leaving costs nothing but the dividend."),
        ("Bindings", r"A binding is a promise not to charge more than that tariff on a good, "
                     r"to anyone. Charge more --- a populist tariff floor counts --- and you "
                     r"defect: no WTO dividend on any of your trades that round, and the board "
                     r"names you. No other penalty. Blank means no promise."),
        ("Provide", rf"The hegemon pays {_pct(cost)}\% of the average country's welfare that "
                    rf"round (average 50 $\to$ {cost * 50:g}), and friction stays as it is."),
        ("Withhold", rf"No bill, but every trade in the world, the hegemon's own included, "
                     rf"loses {_pct(w)}\% more of its goods ({_pct(base)}\% $\to$ "
                     rf"{_pct(base + w)}\%; with the reserve country, 0\% $\to$ {_pct(w)}\%)."),
        ("Crisis", rf"A global crisis cuts every country's welfare. The cut is halved if the "
                   rf"hegemon provides, and shrinks by up to half again with the share of "
                   rf"countries in the WTO: a {worst}\% crisis costs {worst}\% with neither, "
                   rf"{best:g}\% with the hegemon providing and every country in."),
    ], size)


def finance_form_page(name):
    return rf"""
{{\large\bfseries ROUND \blank{{1cm}} --- {esc(name)}}} \hfill (Finance add-on: money \& debt)\par
\textit{{Phases 5 and 6: attach this to your trade form. In Phase 7, use the capstone form.}}
\hr

{{\bfseries Money}} (Phase 5+) \hfill \textit{{one box in each row}}\par
{money_choices()}
{money_rules()}

\vspace{{4pt}}
{{\bfseries Sovereign debt}} (Phase 6+)\par
{debt_choices()}
{debt_rules()}
"""


def build_finance_forms():
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(finance_form_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("forms-finance.tex", "\n".join(pages))


def capstone_form_page(name):
    """Front: every decision of a Phase 7 round. Back: the rules, with numbers."""
    p2 = P2[name]
    L, K = p2["labor"], p2["capital"]
    rows = "\n".join(rf"{esc(g).capitalize()} & \blank{{2.4cm}} & \blank{{2.4cm}} \\"
                     for g in P2_GOODS)
    trades = trade_block(more=r"\textit{(More offers? Attach a sheet.)}")
    back = r"\footnotesize"            # the rules page has to fit on one side
    politics = _rules([("Compensate", compensation_note()), ("MNC tax", mnc_tax_note())],
                      back)
    money = money_rules(back, phase7=True)
    debt, inst = debt_rules(back), institutions_rules(back)
    money_front = money_choices(phase7=True)
    return rf"""
{{\large\bfseries ROUND \blank{{1cm}} --- {esc(name)}}} \hfill (Capstone form: every decision)\par
Labor: \textbf{{{L}}} \quad|\quad Capital: \textbf{{{K}}} \hfill \textit{{The rules, with their numbers, are on the back.}}
\hr

{{\bfseries Production}} --- workers must sum to \textbf{{{L}}}, capital to \textbf{{{K}}}; a good needs both.\par
\begin{{tabular}}{{lll}}
 & Workers & Capital \\
{rows}
\end{{tabular}}

{tariff_block(name)}

{{\bfseries Compensation and side payments}}\par
{politics_choices()}

{mnc_tax_choice(name)}

{trades}

{{\bfseries Money}} \hfill \textit{{one box in each row}}\par
{money_front}

{{\bfseries Debt}}\par
{debt_choices()}

{{\bfseries Institutions \& power}}\par
{institutions_choices()}

\newpage
{{\large\bfseries How the choices work --- {esc(name)}}} \hfill (Capstone form, back)\par
\hr

{{\bfseries Trade politics}}\par
{politics}

{{\bfseries Money}}\par
{money}

{{\bfseries Debt}}\par
{debt}

{{\bfseries Institutions \& power}}\par
{inst}
"""


def build_capstone_forms():
    global current_goods
    current_goods = P2_GOODS
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(capstone_form_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("forms-capstone.tex", "\n".join(pages))


# ── Submitting from a laptop (one page) ───────────────────────────────

PAGE_URL = "https://joshuaalley.github.io/ipe-simulation/"


def build_submit_howto():
    body = rf"""
{{\Large\bfseries Submitting your round from a laptop}}\par
\vspace{{2pt}}
The class page: \texttt{{{esc(PAGE_URL)}}} \hfill One person per team submits.
\hr

\begin{{enumerate}}[leftmargin=1.6em,itemsep=6pt]
  \item \textbf{{Tap the phase}} (it's on the board), then \textbf{{your country}}.
  \item \textbf{{Production.}} Type where your workers and capital go. Green ticks
    mean everything is placed; the page won't let you submit until it is.
  \item \textbf{{What your team decided.}} Tariffs on your imports, the trades you
    agreed, compensation, and from Phase 3 your MNC tax (blank keeps last
    round's). Money, debt and WTO choices appear once those phases start.
  \item \textbf{{Submit.}} The page saves a small file (\texttt{{Bosque.json}}) and
    opens the class Dropbox page. \textbf{{Drop the file in.}} That's it.
\end{{enumerate}}

\vspace{{6pt}}
{{\bfseries Good to know}}
\begin{{itemize}}[leftmargin=1.4em,itemsep=4pt]
  \item \textbf{{Trades need both sides.}} List every swap you agreed. Your partner
    must list the same swap on the same terms, or it doesn't happen.
  \item \textbf{{Changed your mind?}} Submit again. The newest file counts.
  \item \textbf{{Firm owners:}} tap \emph{{A firm I own}}, pick your firm, and set scale,
    move and export. Submit only when something changes; otherwise your firm
    keeps doing what it did.
  \item \textbf{{No file?}} Your country repeats last round's production, tariffs
    and compensation, with no trades.
  \item \textbf{{No laptop?}} Hand in the paper form as before.
\end{{itemize}}
"""
    return write_tex("how-to-submit.tex", body)


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = [
        build_country_briefs(),
        build_phase1_forms(),
        build_phase2_forms(),
        build_firm_forms(),
        build_finance_forms(),
        build_capstone_forms(),
        build_submit_howto(),
    ]
    for p in out:
        print("wrote", os.path.basename(p))
