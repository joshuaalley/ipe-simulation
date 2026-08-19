# Running the IPE Simulation in Class

A facilitation guide for operating the simulation live, with the Jupyter
notebook (`simulation.ipynb`) projected for the whole room.

## It's a simultaneous-move game, not turn-based

Every country commits its decisions for a round **at the same time, in
ignorance of what the others chose**. The instructor enters all the forms and
the engine resolves the whole round at once in a single `run_round` call.
Nobody reacts to anyone else *within* a round.

That simultaneity is the point: it makes the trust problems real. A country
can promise a low tariff during negotiation, then defect on the form you
collect — and nobody finds out until the reveal. (If it were turn-based, players
could react inside a round and that tension would disappear.)

## What projects well

Three kinds of output:

- **`sim.show()`** — the big-screen scoreboard. Use this one live.
- **matplotlib charts** (`plot_*`) — your "aha" visuals for debrief.
- **Text tables** (`print_*`) — the detailed record; better for your own
  reference than for the back row.

Drive everything from the notebook, running **one cell at a time** so the class
watches each result appear.

### Sizing the scoreboard to the room

`print_results()` renders at the notebook's own font size, with fixed 12-character
columns — small and padding-heavy on a projector. `sim.show()` exists to fix that:
large type, tight cells, colored gains, and event badges (CRISIS, DEFAULT,
defected, WTO, hegemon).

```python
sim.show()                              # good in a normal classroom
sim.show(scale=1.5)                     # deeper room
sim.show(scale=1.8, columns="core")     # very deep room: welfare + gains only
```

`scale` multiplies every dimension, so **check it once in each new room**. Columns
grow with the phase (factor prices from Phase 2, FX/stress from Phase 5, debt from
Phase 6); if a high `scale` pushes them off the edge, trim with `columns="core"`
or an explicit list such as `columns=["welfare", "gains", "debt"]`.

Other options: `sort="gains"` turns it into a leaderboard (the default keeps a
fixed country order, which is easier to compare round to round), and
`trades=False` drops the trade log.

## Entering a round from a spreadsheet

Typing nested dicts in front of the class is slow and a mistyped key throws a
traceback mid-session. Instead, keep the paper forms and transcribe into Excel.

**Before class** — generate a blank workbook, pre-filled with your countries,
goods, firm roster, and current policy settings:

```python
sim.write_round_template("rounds/round07.xlsx")
```

**In or after class** — type the numbers from the paper forms into the sheets,
then load and run:

```python
sim.run_round(**sim.load_round("rounds/round07.xlsx"))
sim.show(scale=1.4)
```

Sheets map onto the engine's decisions:

| Sheet | Holds | Notes |
|---|---|---|
| `production` | labor (and capital, Phase 2+) per good | endowments shown for checking |
| `trades` | `exporter, importer, good_out, qty_out, good_in, qty_in` | one row per agreed swap — mirrors the paper form |
| `tariffs` | `importer, partner, good, tariff` | only non-zero rows; `0.25` and `25` both mean 25% |
| `firms` | scale, relocate_to, export | Phase 3+; blank `relocate_to` = stay |
| `finance` | FX regime, capital controls, money growth, borrow/repay/default, WTO | Phase 5+; pre-filled with current settings |

`load_round` validates before anything runs and reports **every** problem at once
("unknown country 'Atlantis'", "no row for Pecos"), so you fix a sheet in one pass
rather than one error at a time.

The filled workbooks are the semester's data. Keep the `rounds/` folder and a
future class is a load, not a retype.

## The rhythm of one round (~75-minute session)

### 1. Recap & set the scene (~5 min, projected)
Run the dashboard for the current phase so everyone sees the standing state:

| Phase | Dashboard |
|---|---|
| 1–4 | `sim.print_results()` from the last round |
| 5+ | `sim.print_monetary_dashboard()` |
| 6+ | `sim.print_debt_dashboard()` |
| 7 | `sim.print_institutions_dashboard()` |

This is the "where we stand" board — who's winning, who's in crisis, who's in debt.

### 2. Hand out decision forms (~2 min)
At a phase start you've already run `sim.print_country_briefs()` and
`sim.print_decision_forms()` (plus `sim.print_firm_decision_forms()` in
Phase 3+). Students have paper. **Project the briefs** so the room shares one
picture of each country's situation.

### 3. Negotiation (~20–25 min, off-screen)
This is the heart, and it happens **away from the projector** — teams huddle,
send envoys, cut deals: trades, FX-regime coordination, monetary-union talks,
WTO bargaining, side-payment offers, coalition-building for a hegemonic
challenge. They write commitments on their forms. **Nobody reveals to you
yet** — this is the simultaneous-move part. They can lie to each other.

### 4. Collect forms & enter decisions (~10 min)
Type the paper forms into the round's decision dicts in the notebook cell:
`roundN_decisions`, `roundN_firm_decisions`, `roundN_monetary`,
`roundN_debt`, `institutional_decisions`, `side_payments`. The templates are
pre-built — you're just filling in numbers. This is the reveal moment:
defections, surprise tariffs, and broken promises all surface here.

### 5. Run the round & project results (~5 min)
Run the `sim.run_round(...)` cell, then `sim.print_results()`. The consolidated
table appears — production, trade log (annotated with terms-of-trade, tariffs,
FX friction), consumption & welfare, factor prices, and the firm / monetary /
debt / institution sections as relevant. The **trade log** is where students
see whose deals went through and whose say `FAILED`.

### 6. Project the plots & debrief (~15–20 min)
Run the plot cell. The most discussion-generating ones:

- `sim.plot_trade_network()` — who traded with whom; tariffs/friction visibly shrink the arrows
- `sim.plot_welfare()` — the running scoreboard, split per phase
- `sim.plot_gains_from_trade()` — who won/lost this round (trade-only; excludes borrowing)
- `sim.plot_firm_locations()` (Phase 3+) — where MNCs clustered
- `sim.plot_currency_health()` (Phase 5+) — the FX-index and stress curves; crises show as cliffs

Debrief off the visuals: *Why did that currency collapse? Who defaulted, and
why was it tempting? Did the hegemon provide the public good — and who paid?*

### 7. Inject the next shock & preview (~3 min)
Run the relevant `inject_*` cell so the next round opens with a new wrinkle:

- `sim.inject_productivity_surge(country, industry, multiplier)` — automation / new tech / foreign-cost collapse
- `sim.inject_populist_backlash(country)` — tariff floor + MNC tax
- `sim.inject_monetary_shock(country, growth)` — forced money printing
- `sim.inject_speculative_attack(country)` — manual currency crisis (pacing safety valve)
- `sim.inject_capital_flight(country, severity)` — balance-of-payments shock
- `sim.inject_global_crisis(severity)` — system-wide shock (Phase 7)

Each prints a banner — project it as the cliffhanger.

## Phase transitions

At the start of a new phase, run `sim.upgrade_to_phaseN(...)`, then
`print_country_briefs()`, the new dashboard, and new decision forms. The
upgrade prints a banner explaining the new mechanics — project it as your
mini-lecture hook.

Order of the upgrades across the semester:

```
upgrade_to_phase2(PHASE2_COUNTRIES, PHASE2_GOODS)
upgrade_to_phase3(firms_for_class)
sim.phase = 4                      # toggle; firms gain the export decision
award_reserve_currency()           # end of the trade era — names the hegemon
upgrade_to_phase5()                # money & FX
upgrade_to_phase6()                # sovereign debt
upgrade_to_phase7()                # institutions & capstone
```

## Between sessions (do not skip)

At the **end of every class**, run the Save cell:

```python
import json
with open("simulation_state.json", "w") as f:
    json.dump(sim.get_state(), f, indent=2)
```

Next class, run the Restore cell **first thing**:

```python
with open("simulation_state.json") as f:
    state = json.load(f)
sim = IPESimulation.from_state(state)
```

The full state returns intact — round number, phase, every country's
debt / currency / WTO status, firm profits, monetary unions, the hegemon.
You never lose a semester's accumulated history.

## Practical projection tips

- Run cells **one at a time** so results appear live — the suspense of
  `run_round` resolving is part of the theater.
- Bump the Jupyter font size before class (Ctrl/Cmd-`+`); the monospaced
  tables need to read from the back row.
- Full-screen the browser. The `print_*_dashboard` views are narrower and more
  projection-friendly than the full `print_results` for at-a-glance moments.
- Keep **one cell per round** (copy the template each time) so you have a
  scrollable record of the whole game to revisit in the capstone debrief.

## Phase / round map (26 sessions)

| Phase | Rounds | Theme |
|---|---|---|
| 1 | 1–3 | Ricardo — comparative advantage, gains from trade, tariffs |
| 2 | 4–7 | Heckscher-Ohlin — factor endowments, Stolper-Samuelson |
| 3 | 8–10 | MNCs & varieties — new trade theory, divided loyalty |
| 4 | 11–14 | Heterogeneous firms — Melitz selection, populist backlash |
| 5 | 15–20 | Money & FX — the trilemma, speculative attacks, unions |
| 6 | 21–22 | Sovereign debt — original sin, default, IMF |
| 7 | 23–26 | Institutions & capstone — WTO, hegemonic stability |
