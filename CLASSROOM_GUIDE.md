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

## Setting up for your class size

The engine ships with six countries and thirteen firms. Neither is fixed — size
both to your enrollment before the first session.

### How many students per country

Aim for **three per country; never fewer than two.** Phase 2 turns on an argument
*inside* each country ("do your workers or your capital owners win from this
trade?"), and a group of two makes that a 1-on-1 with no coalition to form. Three
also survives an absence — two becomes a solo player.

| Students | Countries | Split |
|---|---|---|
| 8–9 | 3 | 3 / 3 / 3 |
| 10–12 | 4 | 3 / 3 / 3 / 2 up to 3 / 3 / 3 / 3 |
| 13–15 | 5 | 3 each |
| 16–18 | 6 (all) | 3 each |

Fewer countries also means shorter negotiations and less to transcribe each
round. The cost is a thinner trade network: four countries gives six trading
pairs, six countries gives fifteen.

### Which countries to drop

Keep the *spread* — two things make the models legible:

- **Phase 1** needs a wide range of opportunity costs. Bosque (4.00) and Llano
  (0.33) are the extremes; keep both or the gains from trade get subtle.
- **Phase 2** needs a clearly labor-abundant *and* a clearly capital-abundant
  country, or Stolper–Samuelson has nothing to contrast.

| Country | K/L | Phase 1 opp. cost | Best TFP | Firms hosted |
|---|---|---|---|---|
| Sabine | 0.35 | 0.67 | cloth 1.1 | 3 |
| Bosque | 0.42 | **4.00** | cloth 1.2 | 2 |
| Llano | 0.80 | **0.33** | wine 1.3 | 2 |
| Brazos | 1.00 | 1.00 | — | 2 |
| Trinity | 1.67 | 2.00 | machinery 1.2 | 3 |
| Pecos | 2.40 | 1.00 | machinery 1.1 | 1 |

**Brazos and Pecos are the natural first cuts** — both sit at opportunity cost
1.00 (no comparative advantage), so they are redundant with each other. Dropping
both leaves Sabine / Bosque / Llano / Trinity: every survivor has a clear role,
both Phase 1 extremes are intact, and two labor-abundant countries face one
capital-abundant one.

If you drop them, note that the Block 2 slide's `\note{}` uses Pecos and Brazos
in its worked opportunity-cost example — worth a touch-up.

### The firm roster must match the country set

**The engine refuses a roster hosted in countries that aren't in play.** This is
deliberate: an off-map host used to pass the upgrade silently and then die with a
bare `KeyError` mid-round, in front of the class.

```python
sim.upgrade_to_phase3(PHASE3_FIRMS)   # 4-country game -> ValueError naming F6, F8, F9
```

Build a roster that matches instead — **one firm per student**:

```python
from engine import build_firm_roster
firms = build_firm_roster(["Sabine", "Bosque", "Llano", "Trinity"], n_firms=11)
sim.upgrade_to_phase3(firms)
```

`build_firm_roster` keeps firms already hosted in surviving countries, rehomes
orphans to the least-loaded host, and trims to `n_firms` by dropping MED-tier
firms first — so the HIGH/LOW productivity spread that drives Melitz selection in
Phase 4 stays intact. It prints what it produced:

```
  FIRM ROSTER BUILT  --  11 firms, 4 countries
  Rehomed off-map firms: F6, F8, F9
  Host          Firms   Mean prod.
  Sabine            3         0.80
  Bosque            2         1.00
  Llano             3         1.10
  Trinity           3         1.00

  By industry: cloth 4, machinery 4, wine 3
  By tier:     HIGH 3, MED 4, LOW 4
```

**Read that table before committing.** The shipped roster is tuned so no country
is over-rewarded or over-punished, and rehoming can drift that. It is not
rebalanced silently precisely so you can catch it — swap a firm by hand if one
host looks favoured.

### Regenerate the handouts to match

The paper handouts are built from the engine's constants, so they must be
regenerated whenever the country set changes:

```
cd handouts
python make_handouts.py --countries Sabine Bosque Llano Trinity --firms 11
```

Omit both flags for the full six-country, thirteen-firm set. The generator uses
the same `build_firm_roster`, so the printed MNC forms always match what the
engine will accept.

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

**Each round cell in the notebook is a single line.** Run it twice:

```python
sim.play_round("rounds/round07.xlsx", scale=1.4)
```

- **1st run** (before class) — the workbook doesn't exist yet, so this writes a
  blank one, pre-filled with your countries, goods, firm roster and current
  policy settings, and stops.
- **2nd run** (after you've typed in the paper forms) — loads it, runs the
  round, and projects the scoreboard.

It never overwrites a workbook you have filled in, never fails just because a
future round's file isn't there yet, and refuses to play the same workbook twice
by accident — a stray re-run reprojects the existing result instead of advancing
the round. Pass `replay=True` if you genuinely mean to replay it.

Give every round its own file. Two cells pointing at the same workbook will
collide, because a template written at one phase is missing the sheets a later
phase needs.

The two halves are still available separately when you want them —
`sim.write_round_template(path)` and `sim.run_round(**sim.load_round(path))`.

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
- `sim.inject_speculative_attack(country)` — currency crisis on a named country
- `sim.inject_capital_flight(country, severity)` — balance-of-payments shock
- `sim.inject_global_crisis(severity)` — system-wide shock (Phase 7)

Each prints a banner — project it as the cliffhanger.

### Let the crisis pick its own victim

For the two crises that fall on a *specific* country, don't choose the target
yourself — a hand-picked victim invites "you singled us out." These pick the
most exposed country from the countries' **own policy choices**:

```python
sim.trigger_speculative_attack()          # Phase 5+ currency crisis
sim.trigger_capital_flight(severity=0.6)  # Phase 6+ debt / BOP shock
```

Both print a scored exposure table before firing, so you can project exactly
why a country was hit:

| Score component | Currency attack | Debt shock |
|---|---|---|
| trilemma overreach (peg + open capital + own money) | 3.0 | — |
| accumulated stress | 2.0 each | — |
| post-warning jitters | 1.0 | — |
| loose money | 10 × growth rate | — |
| leverage (debt ÷ consumption) | — | 4.0 × ratio |
| original sin (weak currency) | — | 2.0 × depreciation |
| open capital account | 1.0 | 1.0 |
| already-weak currency | 2.0 × depreciation | — |
| post-default ban | — | 1.0 |

Ranking is fully deterministic — the same decisions always produce the same
target (ties break on weakest currency, then the larger secondary exposure,
then name), so nothing depends on your judgement or on dict ordering.

**The selection is the lesson.** Project `sim.print_vulnerability("fx")` (or
`"debt"`) *before* firing and ask the room to predict who gets hit. The country
that pegged with an open capital account while printing money is the one the
market comes for — which is precisely the trilemma, made personal.

You can also project the table on its own, without firing anything, as a
mid-phase warning shot.

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
