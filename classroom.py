"""
Classroom helpers for the IPE simulation.

Two jobs, both about running the thing in front of a room:

  1. Projection display -- sim.show(scale=1.4)
     A big, tight scoreboard for the projector. print_results() stays as the
     detailed record; this is the thing students actually read from 30 feet.

  2. Spreadsheet round I/O -- sim.write_round_template(...) / sim.load_round(...)
     Type a round into Excel instead of into nested Python dicts. The filled
     workbooks are the semester's data, so re-running next year is a load,
     not a retype.

Kept out of engine.py so the engine stays focused on mechanics, and so the
engine keeps working if pandas/openpyxl are ever missing.
"""

import os

import pandas as pd


# ═══════════════════════════════════════════════════════════════════
#  1. PROJECTION SCOREBOARD
# ═══════════════════════════════════════════════════════════════════
#
# Design constraints (from projecting the old print_results):
#   - print() renders at the notebook's font size, with no knob. -> explicit px
#   - 12-char numeric columns are mostly padding.                -> tight cells
#   - Every section every round is too tall to read at once.     -> essentials
#   - Notebooks may be dark-themed; projectors want light.       -> forced light

BASE_FONT_PX = 20          # scale=1.0; bump scale for a bigger room
POSITIVE = "#1a7f37"
NEGATIVE = "#b3261e"
MUTED = "#5b5b5b"
RULE = "#c9c9c9"


def _gain_cell(pct):
    """Return (text, color) for a gains-from-trade percentage."""
    if pct == float("inf"):
        return "n/a", MUTED
    return f"{pct:+.1f}%", (POSITIVE if pct >= 0 else NEGATIVE)


def _badges(res_c, is_hegemon=False):
    """Short flags shown next to a country name (crisis, default, etc.)."""
    out = []
    mon = res_c.get("monetary") or {}
    if mon.get("crisis"):
        out.append(("CRISIS", NEGATIVE))
    elif mon.get("warning"):
        out.append(("warning", "#b26a00"))
    debt = res_c.get("debt") or {}
    if debt.get("defaulted"):
        out.append(("DEFAULT", NEGATIVE))
    if debt.get("austerity_active"):
        out.append(("austerity", "#b26a00"))
    inst = res_c.get("institutions") or {}
    if inst.get("defected"):
        out.append(("defected", NEGATIVE))
    if inst.get("wto_member"):
        out.append(("WTO", MUTED))
    if is_hegemon:
        out.append(("hegemon", "#5a3fc0"))
    return out


ALL_COLUMNS = {
    "welfare": "Welfare",
    "gains": "vs. autarky",
    "wage": "Wage",
    "ret": "Return to K",
    "fx": "FX",
    "stress": "Stress",
    "debt": "Debt",
}
CORE_COLUMNS = ["welfare", "gains"]


def _columns_for_phase(phase, columns=None):
    """
    Which numeric columns the scoreboard shows.

    columns : None       phase-appropriate default (grows with the phase)
              "core"     just Welfare + vs. autarky -- use this when a high
                         `scale` in a later phase would push columns off the
                         edge of the screen
              list       explicit keys from ALL_COLUMNS
    """
    if columns == "core":
        keys = list(CORE_COLUMNS)
    elif isinstance(columns, (list, tuple)):
        keys = [k for k in columns if k in ALL_COLUMNS]
        if not keys:
            raise ValueError(
                f"No valid column keys in {columns!r}; "
                f"choose from {sorted(ALL_COLUMNS)}"
            )
    else:
        keys = list(CORE_COLUMNS)
        if phase >= 2:
            keys += ["wage", "ret"]
        if phase >= 5:
            keys += ["fx", "stress"]
        if phase >= 6:
            keys += ["debt"]
    return [(ALL_COLUMNS[k], k) for k in keys]


def _row_values(res_c):
    """Pull the scoreboard values out of one country's result dict."""
    fp = res_c.get("factor_prices") or {}
    mon = res_c.get("monetary") or {}
    debt = res_c.get("debt") or {}
    return {
        "welfare": res_c["welfare"],
        "gains": res_c["gains_from_trade_pct"],
        "wage": fp.get("avg_wage"),
        "ret": fp.get("avg_capital_return"),
        "fx": mon.get("depreciation_factor"),
        "stress": mon.get("stress"),
        "debt": debt.get("debt_stock"),
    }


def _fmt(key, val):
    """Format one scoreboard value; returns (text, color)."""
    if val is None:
        return "--", MUTED
    if key == "gains":
        return _gain_cell(val)
    if key == "stress":
        return str(int(val)), (NEGATIVE if val >= 2 else
                               "#b26a00" if val == 1 else MUTED)
    if key == "fx":
        return f"{val:.2f}", (NEGATIVE if val < 0.95 else "inherit")
    if key in ("wage", "ret"):
        return f"{val:.2f}", "inherit"
    if key == "debt":
        return f"{val:.1f}", (NEGATIVE if val > 0 else MUTED)
    return f"{val:.1f}", "inherit"


def scoreboard_html(sim, round_num=None, scale=1.0, sort=None, trades=True,
                    columns=None):
    """
    Build the projection scoreboard as an HTML string.

    scale  : float  1.0 is already much larger than notebook default; raise it
                    for a bigger room (1.4-1.8 is typical for a deep hall).
    sort   : None   keep the engine's country order (stable across rounds --
                    best for round-to-round comparison)
             "gains" | "welfare"  sort descending by that column
    trades : bool   include the round's trade log underneath
    columns: None   phase-appropriate columns; "core" for just welfare+gains
                    (use when a high scale would push columns off screen), or
                    an explicit list of keys from ALL_COLUMNS.
    """
    if not sim.history:
        return "<p>No rounds played yet.</p>"

    rd = sim.history[-1] if round_num is None else sim.history[round_num - 1]
    res = rd["results"]
    phase = rd["phase"]
    hegemon = rd.get("hegemon")
    names = list(sim.countries.keys())

    rows = [(n, _row_values(res[n])) for n in names]
    if sort in ("gains", "welfare"):
        rows.sort(key=lambda r: (r[1][sort] if r[1][sort] not in (None, float("inf"))
                                 else -1e18), reverse=True)

    f = BASE_FONT_PX * scale
    cols = _columns_for_phase(phase, columns)

    pad = f"{0.18 * f:.0f}px {0.42 * f:.0f}px"
    th = (f"padding:{pad};text-align:right;font-weight:600;"
          f"border-bottom:2px solid {RULE};white-space:nowrap;")
    td = (f"padding:{pad};text-align:right;"
          f"border-bottom:1px solid {RULE};white-space:nowrap;"
          "font-variant-numeric:tabular-nums;")

    parts = [
        f'<div style="background:#ffffff;color:#111111;padding:{0.9*f:.0f}px;'
        f'font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;'
        f'font-size:{f:.0f}px;line-height:1.25;">',
        f'<div style="font-size:{1.45*f:.0f}px;font-weight:700;'
        f'margin-bottom:{0.5*f:.0f}px;">Round {rd["round"]}'
        f'<span style="color:{MUTED};font-weight:400;font-size:{0.75*f:.0f}px;">'
        f'&nbsp;&nbsp;Phase {phase}</span></div>',
        '<table style="border-collapse:collapse;">',
        f'<tr><th style="{th}text-align:left;">Country</th>'
        + "".join(f'<th style="{th}">{label}</th>' for label, _ in cols)
        + "</tr>",
    ]

    for name, vals in rows:
        badges = _badges(res[name], is_hegemon=(name == hegemon))
        badge_html = "".join(
            f'<span style="font-size:{0.62*f:.0f}px;color:{c};'
            f'border:1px solid {c};border-radius:{0.25*f:.0f}px;'
            f'padding:0 {0.25*f:.0f}px;margin-left:{0.3*f:.0f}px;'
            f'vertical-align:middle;">{txt}</span>'
            for txt, c in badges
        )
        cells = ""
        for _, key in cols:
            text, color = _fmt(key, vals[key])
            weight = "600" if key == "gains" else "400"
            cells += (f'<td style="{td}color:{color};font-weight:{weight};">'
                      f"{text}</td>")
        parts.append(
            f'<tr><td style="{td}text-align:left;font-weight:600;">'
            f"{name}{badge_html}</td>{cells}</tr>"
        )

    parts.append("</table>")

    if trades:
        log = rd.get("trade_log") or []
        parts.append(
            f'<div style="margin-top:{0.7*f:.0f}px;font-size:{0.8*f:.0f}px;">'
            f'<span style="font-weight:600;">Trades</span>'
        )
        if log:
            parts.append(
                '<ul style="margin:0.2em 0 0 1.1em;padding:0;">'
                + "".join(f"<li>{line.strip()}</li>" for line in log)
                + "</ul>"
            )
        else:
            parts.append(
                f'<span style="color:{MUTED};">&nbsp;-- none this round</span>'
            )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def scoreboard_text(sim, round_num=None, sort=None, columns=None):
    """Plain-text fallback -- narrower columns than print_results()."""
    if not sim.history:
        return "No rounds played yet."
    rd = sim.history[-1] if round_num is None else sim.history[round_num - 1]
    res, phase = rd["results"], rd["phase"]
    cols = _columns_for_phase(phase, columns)
    rows = [(n, _row_values(res[n])) for n in sim.countries]
    if sort in ("gains", "welfare"):
        rows.sort(key=lambda r: (r[1][sort] if r[1][sort] not in (None, float("inf"))
                                 else -1e18), reverse=True)

    out = [f"ROUND {rd['round']}  (Phase {phase})",
           f"{'Country':10s}" + "".join(f"{lab:>12s}" for lab, _ in cols)]
    out.append("-" * (10 + 12 * len(cols)))
    for name, vals in rows:
        line = f"{name:10s}"
        for _, key in cols:
            text, _c = _fmt(key, vals[key])
            line += f"{text:>12s}"
        out.append(line)
    return "\n".join(out)


def show(sim, round_num=None, scale=1.0, sort=None, trades=True, columns=None):
    """
    Display the projection scoreboard. Renders as HTML in a notebook;
    falls back to plain text anywhere else.
    """
    html = scoreboard_html(sim, round_num=round_num, scale=scale,
                           sort=sort, trades=trades, columns=columns)
    try:
        from IPython.display import HTML, display
        from IPython import get_ipython
        if get_ipython() is None:
            raise ImportError
        display(HTML(html))
        return None
    except Exception:
        print(scoreboard_text(sim, round_num=round_num, sort=sort,
                              columns=columns))
        return None


# ═══════════════════════════════════════════════════════════════════
#  2. SPREADSHEET ROUND I/O
# ═══════════════════════════════════════════════════════════════════
#
# One workbook per round. Sheets map onto run_round()'s arguments:
#
#   production -> decisions[country]["production"]
#   tariffs    -> decisions[country]["tariffs"]      (long; non-zero rows only)
#   trades     -> trades  (exporter, importer, good_out, qty_out,
#                          good_in, qty_in)  -- mirrors the paper form
#   firms      -> firm_decisions                     (Phase 3+)
#   finance    -> monetary / debt / institutional    (Phase 5+)

TRUE_WORDS = {"true", "t", "yes", "y", "1"}
FALSE_WORDS = {"false", "f", "no", "n", "0", ""}


def _blank(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or \
        (isinstance(v, str) and not v.strip())


def _as_bool(v, default=False, where=""):
    if _blank(v):
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in TRUE_WORDS:
        return True
    if s in FALSE_WORDS:
        return False
    raise ValueError(f"{where}: expected yes/no, got {v!r}")


def _as_num(v, default=0.0, where=""):
    if _blank(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: expected a number, got {v!r}")


def _as_text(v, default=None):
    if _blank(v):
        return default
    return str(v).strip()


# ── Template writer ───────────────────────────────────────────────

def write_round_template(sim, path, round_num=None):
    """
    Write a blank workbook for the next round, pre-filled with this
    simulation's countries, goods, firm roster and current policy settings.
    You only type numbers.

    Returns the path written.
    """
    rnd = (sim.round_num + 1) if round_num is None else round_num
    names = list(sim.countries.keys())
    goods = list(sim.goods)
    phase = sim.phase
    sheets = {}

    # -- production ------------------------------------------------
    if phase == 1:
        prod = pd.DataFrame({"country": names})
        for g in goods:
            prod[g] = 0
        prod["labor_available"] = [sim.countries[n]["labor"] for n in names]
    else:
        prod = pd.DataFrame({"country": names})
        for g in goods:
            prod[f"labor_{g}"] = 0
        for g in goods:
            prod[f"capital_{g}"] = 0
        prod["labor_available"] = [sim.countries[n]["labor"] for n in names]
        prod["capital_available"] = [sim.countries[n]["capital"] for n in names]
    sheets["production"] = prod

    # -- trades (blank; one row per agreed swap) -------------------
    sheets["trades"] = pd.DataFrame(
        columns=["exporter", "importer", "good_out", "qty_out",
                 "good_in", "qty_in"]
    )

    # -- tariffs (blank; only non-zero rows needed) ----------------
    sheets["tariffs"] = pd.DataFrame(
        columns=["importer", "partner", "good", "tariff"]
    )

    # -- firms (Phase 3+) ------------------------------------------
    if phase >= 3 and sim.firms:
        firms = pd.DataFrame({
            "firm": list(sim.firm_config.keys()),
            "variety": [c["variety"] for c in sim.firm_config.values()],
            "current_host": [sim.firms[f]["host"] for f in sim.firm_config],
            "max_scale": [c["max_scale"] for c in sim.firm_config.values()],
        })
        firms["scale"] = 0
        firms["relocate_to"] = ""
        firms["export"] = "no"
        sheets["firms"] = firms

    # -- finance (Phase 5+) ----------------------------------------
    if phase >= 5:
        fin = pd.DataFrame({"country": names})
        mon = [sim._mon(n) for n in names]
        fin["fx_regime"] = [m.get("fx_regime", "float") for m in mon]
        fin["capital_controls"] = [
            "yes" if m.get("capital_controls") else "no" for m in mon]
        fin["independent_monetary"] = [
            "yes" if m.get("independent_monetary", True) else "no" for m in mon]
        fin["money_supply_growth"] = [
            m.get("money_supply_growth", 0.0) for m in mon]
        if phase >= 6:
            fin["borrow"] = 0
            fin["repay"] = 0
            fin["default"] = "no"
            fin["debt_now"] = [
                sim.countries[n].get("debt_stock", 0.0) for n in names]
        if phase >= 7:
            fin["join_wto"] = [
                "yes" if sim.countries[n].get("wto_member") else "no"
                for n in names]
            fin["hegemon_provides"] = [
                ("yes" if sim.hegemon_provides else "no")
                if n == sim.hegemon else "" for n in names]
        sheets["finance"] = fin

    # -- reference sheet (not read back) ---------------------------
    sheets["_reference"] = pd.DataFrame({
        "field": ["round", "phase", "goods", "countries",
                  "reserve_currency", "hegemon"],
        "value": [rnd, phase, ", ".join(goods), ", ".join(names),
                  str(sim.reserve_currency_holder), str(sim.hegemon)],
    })

    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for sheet, df in sheets.items():
            df.to_excel(xl, sheet_name=sheet, index=False)
            ws = xl.sheets[sheet]
            for i, col in enumerate(df.columns, start=1):
                width = max(12, min(22, len(str(col)) + 4))
                ws.column_dimensions[
                    ws.cell(row=1, column=i).column_letter].width = width
    return path


# ── Loader ────────────────────────────────────────────────────────

def _read(path, sheet):
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        return None


def load_round(sim, path):
    """
    Read a filled round workbook and return keyword arguments for run_round():

        sim.run_round(**sim.load_round("rounds/round07.xlsx"))

    Raises ValueError listing every problem found, rather than failing on the
    first one -- so you can fix a whole sheet in one pass.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such round workbook: {path}")

    names = list(sim.countries.keys())
    goods = list(sim.goods)
    phase = sim.phase
    problems = []

    # -- production ------------------------------------------------
    prod_df = _read(path, "production")
    if prod_df is None:
        raise ValueError(f"{path}: missing required sheet 'production'")

    decisions = {}
    seen = set()
    for _, row in prod_df.iterrows():
        country = _as_text(row.get("country"))
        if country is None:
            continue
        if country not in sim.countries:
            problems.append(f"production: unknown country {country!r}")
            continue
        seen.add(country)
        where = f"production[{country}]"
        if phase == 1:
            alloc = {g: _as_num(row.get(g), 0.0, where) for g in goods}
            decisions[country] = {"production": alloc}
        else:
            decisions[country] = {"production": {
                "labor": {g: _as_num(row.get(f"labor_{g}"), 0.0, where)
                          for g in goods},
                "capital": {g: _as_num(row.get(f"capital_{g}"), 0.0, where)
                            for g in goods},
            }}
    for missing in set(names) - seen:
        problems.append(f"production: no row for {missing}")

    # -- tariffs (long format, non-zero rows only) -----------------
    tar_df = _read(path, "tariffs")
    if tar_df is not None:
        for i, row in tar_df.iterrows():
            imp = _as_text(row.get("importer"))
            partner = _as_text(row.get("partner"))
            good = _as_text(row.get("good"))
            if imp is None and partner is None and good is None:
                continue
            where = f"tariffs row {i + 2}"
            if imp not in sim.countries:
                problems.append(f"{where}: unknown importer {imp!r}")
                continue
            if partner not in sim.countries:
                problems.append(f"{where}: unknown partner {partner!r}")
                continue
            if good not in goods:
                problems.append(f"{where}: unknown good {good!r}")
                continue
            rate = _as_num(row.get("tariff"), 0.0, where)
            if rate > 1.0:          # tolerate "25" meaning 25%
                rate = rate / 100.0
            decisions.setdefault(imp, {}).setdefault("tariffs", {})
            decisions[imp]["tariffs"].setdefault(partner, {})[good] = rate

    # -- trades ----------------------------------------------------
    trades = []
    tr_df = _read(path, "trades")
    if tr_df is not None:
        for i, row in tr_df.iterrows():
            ex = _as_text(row.get("exporter"))
            im = _as_text(row.get("importer"))
            if ex is None and im is None:
                continue
            where = f"trades row {i + 2}"
            g_out = _as_text(row.get("good_out"))
            g_in = _as_text(row.get("good_in"))
            if ex not in sim.countries:
                problems.append(f"{where}: unknown exporter {ex!r}")
                continue
            if im not in sim.countries:
                problems.append(f"{where}: unknown importer {im!r}")
                continue
            if g_out not in goods or g_in not in goods:
                problems.append(
                    f"{where}: goods must be one of {goods}; "
                    f"got {g_out!r} and {g_in!r}")
                continue
            trades.append((ex, im, g_out,
                           _as_num(row.get("qty_out"), 0.0, where),
                           g_in,
                           _as_num(row.get("qty_in"), 0.0, where)))

    kwargs = {"decisions": decisions, "trades": trades}

    # -- firms (Phase 3+) ------------------------------------------
    if phase >= 3 and sim.firms:
        firm_decisions = {
            fid: {"scale": 0, "relocate_to": None, "export": False}
            for fid in sim.firms
        }
        f_df = _read(path, "firms")
        if f_df is None:
            problems.append("missing sheet 'firms' (required from Phase 3)")
        else:
            for i, row in f_df.iterrows():
                fid = _as_text(row.get("firm"))
                if fid is None:
                    continue
                where = f"firms row {i + 2}"
                if fid not in sim.firms:
                    problems.append(f"{where}: unknown firm {fid!r}")
                    continue
                dest = _as_text(row.get("relocate_to"))
                if dest is not None and dest not in sim.countries:
                    problems.append(f"{where}: unknown relocate_to {dest!r}")
                    dest = None
                firm_decisions[fid] = {
                    "scale": _as_num(row.get("scale"), 0.0, where),
                    "relocate_to": dest,
                    "export": _as_bool(row.get("export"), False, where),
                }
        kwargs["firm_decisions"] = firm_decisions

    # -- finance (Phase 5+) ----------------------------------------
    if phase >= 5:
        fin_df = _read(path, "finance")
        if fin_df is None:
            problems.append("missing sheet 'finance' (required from Phase 5)")
        else:
            monetary, debt, inst = {}, {}, {}
            for i, row in fin_df.iterrows():
                country = _as_text(row.get("country"))
                if country is None:
                    continue
                where = f"finance row {i + 2} ({country})"
                if country not in sim.countries:
                    problems.append(f"{where}: unknown country")
                    continue
                cur = sim._mon(country)
                monetary[country] = {
                    "fx_regime": _as_text(row.get("fx_regime"),
                                          cur.get("fx_regime", "float")),
                    "capital_controls": _as_bool(
                        row.get("capital_controls"),
                        bool(cur.get("capital_controls")), where),
                    "independent_monetary": _as_bool(
                        row.get("independent_monetary"),
                        bool(cur.get("independent_monetary", True)), where),
                    "money_supply_growth": _as_num(
                        row.get("money_supply_growth"),
                        cur.get("money_supply_growth", 0.0), where),
                }
                if phase >= 6:
                    debt[country] = {
                        "borrow": _as_num(row.get("borrow"), 0.0, where),
                        "repay": _as_num(row.get("repay"), 0.0, where),
                        "default": _as_bool(row.get("default"), False, where),
                    }
                if phase >= 7:
                    entry = {}
                    if not _blank(row.get("join_wto")):
                        entry["join_wto"] = _as_bool(
                            row.get("join_wto"), False, where)
                    if entry:
                        inst[country] = entry
                    if country == sim.hegemon and \
                            not _blank(row.get("hegemon_provides")):
                        inst["hegemon_provides"] = _as_bool(
                            row.get("hegemon_provides"), True, where)
            kwargs["monetary_decisions"] = monetary
            if phase >= 6:
                kwargs["debt_decisions"] = debt
            if phase >= 7 and inst:
                kwargs["institutional_decisions"] = inst

    if problems:
        raise ValueError(
            f"Problems in {os.path.basename(path)}:\n  - "
            + "\n  - ".join(problems)
        )
    return kwargs
