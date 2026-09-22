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
        help="number of MNC forms to print, normally one per student "
             "(default: the full roster).")
    return ap.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else argparse.Namespace(
    countries=None, firms=None)

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
    FIRMS = engine.PHASE3_FIRMS
    if ARGS.firms is not None:
        FIRMS = engine.build_firm_roster(NAMES, n_firms=ARGS.firms,
                                         verbose=False)

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
  \item \textbf{{Phase 6 (debt):}} you may borrow against future consumption --- and you may default.
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


def trade_block():
    offer = (r"  \item We give \blank{1.8cm} units of \blank{2.2cm} "
             r"to \blank{2.2cm};\\[2pt] in return: \blank{1.8cm} units of \blank{2.2cm}.")
    return (r"{\bfseries Trade offers} (negotiate first, then commit):\par" + "\n"
            r"\begin{itemize}[itemsep=4pt,leftmargin=1.4em]" + "\n"
            + offer + "\n" + offer + "\n" + offer + "\n"
            r"\end{itemize}" + "\n"
            r"\textit{(Use the back of the sheet for more offers.)}")


current_goods = P1_GOODS  # rebound per builder


def politics_block():
    """Compensation + side payments: the alternatives to closing the border."""
    cap = _pct(engine.COMPENSATION_MAX_SHARE)
    dead = _pct(engine.COMPENSATION_DEADWEIGHT * 0.10)      # cost of a 10% payment
    buys = f"{engine.APPROVAL_COMPENSATION * 0.10:.1f}"     # what it buys
    return (
        r"{\bfseries Compensation} --- buy off the groups trade displaced, "
        r"instead of taxing imports:\par" "\n"
        r"\begin{itemize}[nosep,leftmargin=1.4em]" "\n"
        rf"  \item This round we compensate: \blank{{1.4cm}}\% of our consumption "
        rf"\quad \textit{{(0--{cap}\%)}}" "\n"
        r"  \item Side payment to another country: \blank{1.6cm} units of "
        r"\blank{2cm} to \blank{2.2cm}." "\n"
        r"\end{itemize}" "\n"
        rf"\textit{{Compensating 10\% buys about +{buys} approval a round and costs "
        rf"{dead}\% of your welfare --- roughly what a 20\% tariff buys, for less, "
        rf"and your trade stays open. A side payment does the same for whoever "
        rf"receives it (only the net counts).}}"
    )




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

def firm_form_block(fid):
    cfg = FIRMS[fid]
    price = engine.WORLD_PRICES[cfg["industry"]]
    return rf"""
\noindent\fbox{{\begin{{minipage}}{{0.95\linewidth}}
{{\bfseries ROUND \blank{{1cm}} --- FIRM {esc(fid)}: {esc(cfg['variety'])}}}
\hfill Owner(s): \blank{{4.5cm}}\par
Industry: \textbf{{{esc(cfg['industry'])}}} \quad
Productivity: \textbf{{{cfg['productivity']:.1f}}} \quad
Starting host: \textbf{{{esc(cfg['default_host'])}}}\par
Price per unit sold: \textbf{{{price:.2f}}} \quad
Unit cost: \textbf{{{cfg['unit_cost']:.2f}}} \quad
Max scale: \textbf{{{cfg['max_scale']:.0f}}} \quad
Export fixed cost: \textbf{{{cfg['fixed_export_cost']:.0f}}}\par
\vspace{{3pt}}
SCALE (0--{cfg['max_scale']:.0f}): \blank{{2cm}} \qquad
RELOCATE TO: \blank{{3cm}} \textit{{(blank = stay)}}\par
EXPORT this round? \boxx\ Yes \quad \boxx\ No \hfill\textit{{(Phase 4+; pays the fixed cost)}}\par
\vspace{{2pt}}
{{\small You make \textbf{{scale $\times$ productivity}} units and sell them at the price
above. \textbf{{Profit}} $=$ units sold $\times$ price $-$ scale $\times$ unit cost
$-$ export fixed cost (only if you export) $-$ your host's MNC tax (Phase 4+).}}
\end{{minipage}}}}
\vspace{{8pt}}
"""


def build_firm_forms():
    intro = (r"{\large\bfseries MNC Decision Forms} \hfill (Phase 3+)\par " + "\n"
             r"You own this firm -- alone or with a partner -- even though it "
             r"sits in another country. Each round: choose how much to produce "
             r"(\emph{scale}), whether to \emph{relocate} to a new host, and "
             r"(Phase 4+) whether to pay the fixed cost to \emph{export}. The "
             r"profit is yours; the output counts for whichever country hosts you. "
             r"Productivity tiers: HIGH 1.3, MED 1.0, LOW 0.7." + "\n\\hr\n")
    blocks = [firm_form_block(fid) for fid in FIRMS]
    return write_tex("forms-firms.tex", intro + "\n".join(blocks))


# ── Finance add-on form (monetary / debt / institutions) ──────────────

def _pct(x):
    """0.15 -> '15' (a rule's number, straight from the engine)."""
    return f"{x * 100:.0f}"


def finance_form_page(name):
    pct = " \\quad ".join(rf"\boxx\ {g*100:.0f}\%" for g in MONEY_CHOICES)
    fric, cut = _pct(engine.BASE_FX_FRICTION), _pct(engine.CONTROLS_FIRM_CUT)
    stim = f"{engine.STIMULUS_PER_POINT:g}"
    half = {0.5: "half", 1.0: "all"}.get(engine.WEAK_FX_IMPORT_COST,
                                          f"{engine.WEAK_FX_IMPORT_COST:g} times")
    eg = _pct(engine.WEAK_FX_IMPORT_COST * 0.10)
    drag = f"{engine.WEAK_FX_WELFARE_DRAG * 10:g}"          # % per 10 points
    dcost = f"{engine.DEBT_DEFAULT_COST:g}"
    warn, crisis = _pct(1 - engine.WARNING_DEVALUATION), _pct(1 - engine.CRISIS_DEVALUATION)
    hit = _pct(engine.CRISIS_WELFARE_HIT)
    cap, base = _pct(engine.BORROW_CAP_SHARE), _pct(engine.DEBT_BASE_RATE)
    prem, ban = _pct(engine.DEBT_RISK_PREMIUM), engine.DEBT_DEFAULT_BAN_ROUNDS
    dfric = _pct(engine.DEBT_DEFAULT_FRICTION)
    return rf"""
{{\large\bfseries ROUND \blank{{1cm}} --- {esc(name)}}} \hfill (Finance \& institutions add-on)\par
\textit{{Attach this to your trade form once the relevant phase opens.}}
\hr

{{\bfseries Money}} (Phase 5+) \hfill \textit{{one box in each row}}\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
  \item Exchange rate: \quad \boxx\ Peg \quad \boxx\ Float
  \item Capital account: \quad \boxx\ Open \quad \boxx\ Controls
  \item Money growth: \quad {pct} \quad \textit{{(0\% = follow the anchor)}}
\end{{itemize}}
{{\small
\begin{{description}}[nosep,style=sameline,leftmargin=5.6em,font=\normalfont\bfseries]
  \item[Peg] No currency friction on trades with other pegged currencies; other
    cross-currency trades lose {fric}\%. Trades with the reserve country are always
    friction-free. An open peg is what speculators attack.
  \item[Controls] Foreign firms in your country produce {cut}\% less, and (Phase 6)
    you cannot borrow abroad. Speculators cannot run on a closed account.
  \item[Printing] +{stim}\% welfare this round for each point printed, times your FX
    index. Your FX index falls by the same \% for good. A weak currency shrinks every
    import you receive by {half} the drop, and costs {drag}\% of welfare every round
    for each 10 points below 1.00 (FX 0.90 $\to$ imports $-${eg}\%, welfare $-${drag}\%).
  \item[Trilemma] Peg + open + printing: a warning (FX $-${warn}\%). Again next round:
    a crisis (FX $-${crisis}\%, welfare $-${hit}\%).
\end{{description}}}}

\vspace{{4pt}}
{{\bfseries Sovereign debt}} (Phase 6+)\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
  \item Borrow this round: \blank{{2.5cm}} \quad Repay this round: \blank{{2.5cm}}
  \item Default on the debt? \quad \boxx\ Yes \quad \boxx\ No
\end{{itemize}}
{{\small
\begin{{description}}[nosep,style=sameline,leftmargin=5.6em,font=\normalfont\bfseries]
  \item[Borrow] Up to {cap}\% of your consumption a round; welfare rises by the same
    share this round. Not under capital controls.
  \item[Interest] {base}\% + {prem}\% $\times$ (debt $\div$ consumption), every round.
    Debt is owed in the reserve currency, so payments are divided by your FX index
    (FX 0.70 $\to$ 43\% heavier).
  \item[Default] The debt disappears, but welfare falls by {dcost} $\times$
    (debt $\div$ consumption) that round --- more than repaying would cost at par;
    no borrowing for {ban} rounds and +{dfric}\% friction on your trades meanwhile.
    It only pays after a big fall in your currency.
  \item[Last round] Everything still owed is repaid.
\end{{description}}}}

\vspace{{4pt}}
{{\bfseries Institutions \& power}} (Phase 7+)\par
\begin{{itemize}}[nosep,leftmargin=1.4em]
  \item Join / remain in the WTO? \quad \boxx\ Yes \quad \boxx\ No
  \item Bound tariff commitments (good: ceiling): \blank{{6cm}}
  \item \textit{{Hegemon only:}} provide the public good this round? \quad \boxx\ Yes \quad \boxx\ No
\end{{itemize}}
"""


def build_finance_forms():
    pages = []
    for i, name in enumerate(NAMES):
        pages.append(finance_form_page(name))
        if i != len(NAMES) - 1:
            pages.append(r"\newpage")
    return write_tex("forms-finance.tex", "\n".join(pages))


# ── main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = [
        build_country_briefs(),
        build_phase1_forms(),
        build_phase2_forms(),
        build_firm_forms(),
        build_finance_forms(),
    ]
    for p in out:
        print("wrote", os.path.basename(p))
