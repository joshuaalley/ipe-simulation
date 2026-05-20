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

The engine produces two kinds of output, both projector-friendly:

- **Text tables** (`print_*`) — your live "scoreboard."
- **matplotlib charts** (`plot_*`) — your "aha" visuals for debrief.

Drive everything from the notebook, running **one cell at a time** so the class
watches each result appear.

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
