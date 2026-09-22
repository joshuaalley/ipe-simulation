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

import json
import os
import re

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
    appr = res_c.get("approval") or {}
    if appr.get("government_fell"):
        out.append(("GOVT FELL", NEGATIVE))
    elif appr.get("low_rounds"):
        out.append(("unpopular", "#b26a00"))
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
    "approval": "Approval",
    "wage": "Wage",
    "ret": "Return to K",
    "fx": "FX",
    "stress": "Stress",
    "debt": "Debt",
}
CORE_COLUMNS = ["welfare", "gains", "approval"]


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
    appr = res_c.get("approval") or {}
    mon = res_c.get("monetary") or {}
    debt = res_c.get("debt") or {}
    return {
        "welfare": res_c["welfare"],
        "gains": res_c["gains_from_trade_pct"],
        "approval": appr.get("approval"),
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
    if key == "approval":
        colour = (NEGATIVE if val < 30 else
                  "#b26a00" if val < 40 else POSITIVE if val >= 60 else "inherit")
        return f"{val:.0f}", colour
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
    # Buying off the groups trade displaced: a percent of your consumption.
    # Blank or 0 = none. Buys approval, costs a little welfare, border stays open.
    prod["compensation_pct"] = 0
    sheets["production"] = prod

    # -- trades (blank; one row per agreed swap) -------------------
    sheets["trades"] = pd.DataFrame(
        columns=["exporter", "importer", "good_out", "qty_out",
                 "good_in", "qty_in"]
    )

    # -- side payments (blank; goods one country sends another) ----
    sheets["side_payments"] = pd.DataFrame(
        columns=["donor", "recipient", "good", "qty"]
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
        # peg | float ; yes | no ; 0, 2, 5 or 10 (percent -- 0 follows the
        # anchor, anything above is printing money)
        fin["fx_regime"] = [sim._regime(m.get("fx_regime", "float")) for m in mon]
        fin["capital_controls"] = [
            "yes" if m.get("capital_controls") else "no" for m in mon]
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
    """Read a sheet, dropping entirely-blank rows. None if the sheet is absent."""
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        return None
    return df.dropna(how="all")


def _require_columns(df, sheet, required, problems):
    """
    Verify a sheet has the headers we are about to read.

    Without this, a renamed or wrong-phase header reads as a missing value and
    silently becomes 0.0 (or drops a whole row) -- so a typed-in trade vanishes,
    or every country reports zero production, and the failure surfaces later as
    a confusing complaint about the *data* rather than the *headers*.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(
            f"{sheet}: missing column(s) {missing}.\n"
            f"      Found instead: {list(df.columns)}.\n"
            "      Rename the headers to match, or regenerate a blank sheet "
            "with sim.write_round_template(...)."
        )
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
#  3. STATE SNAPSHOTS (so nobody has to remember to save)
# ═══════════════════════════════════════════════════════════════════
#
# play_round writes one after every round it plays, beside the workbook:
#   rounds/round07.xlsx -> rounds/state/round07.json
# Next class, IPESimulation.resume(default=sim) picks up the newest. The
# workbooks stay the source of truth; these are the convenience copy, and
# they carry what a replay cannot: the shocks, unions and bailouts you
# triggered by hand, already applied.

STATE_DIRNAME = "state"
DEFAULT_STATE_DIR = os.path.join("rounds", STATE_DIRNAME)


def state_path(workbook):
    """rounds/round07.xlsx -> rounds/state/round07.json"""
    folder, name = os.path.split(os.path.abspath(workbook))
    stem = os.path.splitext(name)[0]
    return os.path.join(folder, STATE_DIRNAME, stem + ".json")


def save_state(sim, path):
    """Write the whole simulation state as JSON. Returns the path written."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sim.get_state(), f, indent=2)
    return path


def _rel(path):
    """A short path for printing: relative when that is actually shorter."""
    try:
        rel = os.path.relpath(path)
    except ValueError:                      # a different drive on Windows
        return path
    return path if rel.startswith("..") else rel


def latest_state(folder=None):
    """
    The newest snapshot in `folder`: the highest round number in a filename,
    or -- if none are numbered -- the most recently modified. None when there
    are no snapshots at all.
    """
    folder = DEFAULT_STATE_DIR if folder is None else folder
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, n) for n in os.listdir(folder)
             if n.lower().endswith(".json")]
    if not files:
        return None
    numbered = []
    for f in files:
        m = re.search(r"(\d+)", os.path.splitext(os.path.basename(f))[0])
        if m:
            numbered.append((int(m.group(1)), f))
    if numbered:
        return max(numbered)[1]
    return max(files, key=os.path.getmtime)


def resume(folder=None, default=None, verbose=True):
    """
    Pick up where the last class left off. Pass a folder of snapshots (the
    default is rounds/state) or a single .json file. Returns `default` when
    nothing is saved yet, so the first class of a term keeps the fresh
    simulation from the setup cell:

        sim = IPESimulation.resume(default=sim)
    """
    from engine import IPESimulation          # local: engine imports us too
    folder = DEFAULT_STATE_DIR if folder is None else folder
    if str(folder).lower().endswith(".json") and os.path.isfile(folder):
        path = folder
    else:
        path = latest_state(folder)
    if path is None:
        if verbose:
            print(f"  No snapshot in {_rel(folder)} yet -- keeping the "
                  f"simulation from the setup cell.")
        return default
    with open(path, encoding="utf-8") as f:
        sim = IPESimulation.from_state(json.load(f))
    if verbose:
        print(f"\n  Resumed round {sim.round_num}, Phase {sim.phase} -- "
              f"{', '.join(sim.countries)}")
        print(f"  (from {_rel(path)})\n")
    return sim


def _round_number(path):
    """round07.xlsx -> 7; None for a workbook not named that way."""
    m = re.fullmatch(r"round(\d+)\.xlsx", os.path.basename(path), flags=re.I)
    return int(m.group(1)) if m else None


def play_round(sim, path, scale=1.0, autosave=True, **show_kwargs):
    """
    The whole round in one call. Run the same cell twice:

      1. **Before class** -- the workbook doesn't exist yet, so this writes a
         blank template pre-filled with your countries, goods and firm roster,
         and stops.
      2. **After collecting forms** -- you've typed the numbers in, so this
         loads them, runs the round, and projects the scoreboard.

    Safe to run at any time: it never overwrites a workbook you have filled in,
    and it never fails just because a future round's file isn't there yet.
    Re-running a cell whose round has already been played is refused rather
    than silently advancing the clock a second time. A workbook named
    roundNN.xlsx only ever plays as round NN: an earlier one re-projects its
    own board, even with ``replay=True``, and one further ahead waits its turn.
    (To redo a round, resume the snapshot from before it and play it again.)

        sim.play_round("rounds/round05.xlsx", scale=1.4)

    After the round plays, the whole state is saved beside the workbook
    (rounds/state/round05.json), so the end of class needs no ceremony. Pass
    autosave=False to skip it.
    """
    replay = show_kwargs.pop("replay", False)
    if not os.path.exists(path):
        write_round_template(sim, path)
        print(f"\n  Blank workbook written to: {path}")
        print("  Fill in the sheets from the paper forms, then re-run this "
              "cell to play the round.\n")
        return None

    # A workbook named roundNN.xlsx can only ever be played as round NN.
    # This is what makes running the notebook from the top safe after a
    # resume: the earlier round cells re-project their boards instead of
    # replaying on top of the restored state.
    n = _round_number(path)
    if n is not None and n <= sim.round_num:
        print(f"\n  {os.path.basename(path)} is already in history "
              f"(you're on round {sim.round_num}). Showing round {n}.\n")
        show(sim, round_num=n, scale=scale, **show_kwargs)
        return None
    if n is not None and n > sim.round_num + 1:
        print(f"\n  {os.path.basename(path)} is ahead: you're on round "
              f"{sim.round_num}, so round {sim.round_num + 1} comes first.\n")
        return None

    key = os.path.abspath(path)
    played = getattr(sim, "_played_workbooks", None)
    if played is None:
        played = sim._played_workbooks = set()
    if key in played and not replay:
        print(f"\n  Already played: {os.path.basename(path)} "
              f"(you are on round {sim.round_num}).")
        print("  Re-running would play it a second time and advance the round "
              "again.")
        print("  Projecting the existing result instead. To really replay, "
              "call with replay=True.\n")
        show(sim, scale=scale, **show_kwargs)
        return None

    result = sim.run_round(**load_round(sim, path))
    played.add(key)
    if autosave:
        # A failed save must never cost the round: the workbook is still the
        # record and the result is already in sim.history.
        try:
            print(f"  saved: {_rel(save_state(sim, state_path(path)))}")
        except Exception as e:
            print(f"  could not save the snapshot ({e}) -- your workbook is "
                  f"still the record.")
    show(sim, scale=scale, **show_kwargs)
    return result


def _check_reference(sim, path):
    """
    Compare the workbook's recorded phase and country set against the live
    simulation, and fail with the actual diagnosis if they disagree.

    The common cause is not a bad spreadsheet: it is a restarted kernel where
    an upgrade cell never re-ran, so the simulation is behind the workbook.
    """
    ref = _read(path, "_reference")
    if ref is None or "field" not in ref.columns or "value" not in ref.columns:
        return                                   # older workbook; nothing to check
    rec = {str(f).strip(): str(v).strip()
           for f, v in zip(ref["field"], ref["value"])}

    written = rec.get("phase")
    if written and written.isdigit() and int(written) != sim.phase:
        want, have = int(written), sim.phase
        step = ("run the Phase %d upgrade cell before this one"
                % want if want > have else
                "this workbook is from an earlier phase of the game")
        raise ValueError(
            f"{os.path.basename(path)} was written for Phase {want}, but the "
            f"simulation is currently at Phase {have}.\n"
            f"  Your headers are fine -- the simulation is out of step.\n"
            f"  Fix: {step}. After a kernel restart you must re-run the setup "
            f"and upgrade cells in order before replaying a round."
        )
    # Deliberately NO country-set check here. The recorded country list says
    # how the blank template was generated, not what was typed into it: a
    # workbook made by a six-country sim but filled with four valid rows is
    # perfectly good. The production rows are validated one by one below
    # ("unknown country", "no row for"), which catches real mismatches without
    # rejecting data that is fine.


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

    # A workbook records the phase and country set it was built for. Check
    # that FIRST: if the simulation has since moved on -- or, far more often,
    # if the kernel was restarted and an upgrade cell never re-ran -- every
    # column below will look "missing" and the error will blame your headers
    # when the headers are fine.
    _check_reference(sim, path)

    # -- production ------------------------------------------------
    prod_df = _read(path, "production")
    if prod_df is None:
        raise ValueError(f"{path}: missing required sheet 'production'")

    if phase == 1:
        need = ["country"] + goods
    else:
        need = (["country"]
                + [f"labor_{g}" for g in goods]
                + [f"capital_{g}" for g in goods])
    prod_ok = _require_columns(prod_df, "production", need, problems)

    decisions = {}
    seen = set()
    for _, row in (prod_df.iterrows() if prod_ok else []):
        country = _as_text(row.get("country"))
        if country is None:
            continue
        if country not in sim.countries:
            problems.append(
                f"production: unknown country {country!r} -- this game has "
                f"{sorted(sim.countries)}. Delete that row (a country you "
                "dropped), or regenerate the sheet.")
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
        comp = _as_num(row.get("compensation_pct"), 0.0, where)
        if comp > 1:                       # typed as a percent: 10 -> 0.10
            comp = comp / 100
        if comp:
            decisions[country]["compensation"] = comp
    for missing in set(names) - seen:
        problems.append(f"production: no row for {missing}")

    # -- tariffs (long format, non-zero rows only) -----------------
    tar_df = _read(path, "tariffs")
    if tar_df is not None and len(tar_df):
        _require_columns(tar_df, "tariffs",
                         ["importer", "partner", "good", "tariff"], problems)
    if tar_df is not None and len(tar_df) and all(
            c in tar_df.columns for c in ("importer", "partner", "good")):
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
    trade_cols = ["exporter", "importer", "good_out", "qty_out",
                  "good_in", "qty_in"]
    trades_ok = True
    if tr_df is not None and len(tr_df):
        trades_ok = _require_columns(tr_df, "trades", trade_cols, problems)
    if tr_df is not None and trades_ok:
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

    # -- side payments (goods one country sends another) -----------
    side_payments = []
    sp_df = _read(path, "side_payments")
    sp_cols = ["donor", "recipient", "good", "qty"]
    sp_ok = True
    if sp_df is not None and len(sp_df):
        sp_ok = _require_columns(sp_df, "side_payments", sp_cols, problems)
    if sp_df is not None and sp_ok:
        for i, row in sp_df.iterrows():
            donor = _as_text(row.get("donor"))
            recipient = _as_text(row.get("recipient"))
            if donor is None and recipient is None:
                continue
            where = f"side_payments row {i + 2}"
            good = _as_text(row.get("good"))
            if donor not in sim.countries:
                problems.append(f"{where}: unknown donor {donor!r}")
                continue
            if recipient not in sim.countries:
                problems.append(f"{where}: unknown recipient {recipient!r}")
                continue
            if good not in goods:
                problems.append(
                    f"{where}: good must be one of {goods}; got {good!r}")
                continue
            side_payments.append(
                (donor, recipient, good, _as_num(row.get("qty"), 0.0, where)))

    kwargs = {"decisions": decisions, "trades": trades}
    if side_payments:
        kwargs["side_payments"] = side_payments

    # -- firms (Phase 3+) ------------------------------------------
    if phase >= 3 and sim.firms:
        firm_decisions = {
            fid: {"scale": 0, "relocate_to": None, "export": False}
            for fid in sim.firms
        }
        f_df = _read(path, "firms")
        if f_df is None:
            problems.append("missing sheet 'firms' (required from Phase 3)")
        elif not _require_columns(f_df, "firms",
                                  ["firm", "scale", "relocate_to", "export"],
                                  problems):
            pass
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
        fin_need = ["country", "fx_regime", "capital_controls",
                    "money_supply_growth"]
        if phase >= 6:
            fin_need += ["borrow", "repay", "default"]
        if fin_df is None:
            problems.append("missing sheet 'finance' (required from Phase 5)")
        elif not _require_columns(fin_df, "finance", fin_need, problems):
            pass
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
                growth = _as_num(row.get("money_supply_growth"),
                                 cur.get("money_supply_growth", 0.0), where)
                if growth > 1:                       # typed as a percent: 5 -> 0.05
                    growth = growth / 100
                regime = _as_text(row.get("fx_regime"), cur.get("fx_regime", "float"))
                monetary[country] = {
                    "fx_regime": regime.lower() if isinstance(regime, str) else regime,
                    "capital_controls": _as_bool(
                        row.get("capital_controls"),
                        bool(cur.get("capital_controls")), where),
                    "money_supply_growth": growth,
                }
                # Older workbooks carry an independent_monetary column. Pass
                # it through so a contradiction ("no" while printing) is caught.
                if not _blank(row.get("independent_monetary")):
                    monetary[country]["independent_monetary"] = _as_bool(
                        row.get("independent_monetary"), True, where)
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
