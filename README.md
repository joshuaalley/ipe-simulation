# ipe-simulation

A progressive, seven-phase classroom simulation of the international political
economy, driven from a Jupyter notebook and projected for the room.

| File | What it is |
|---|---|
| `engine.py` | The simulation engine — all seven phases. |
| `classroom.py` | Projection scoreboard (`sim.show()`) and spreadsheet round I/O. |
| `simulation.ipynb` | The notebook you drive in class. |
| `CLASSROOM_GUIDE.md` | **Start here** — class-size setup, round rhythm, facilitation. |
| `handouts/` | Printable country briefs and decision forms. |
| `calculator.py`, `calculator_template.html` | Students' calculator and round form: `sim.export_calculator()` writes it. Its decision files land in `rounds/inbox`, and `play_round` builds rounds from them. |
| `docs/` | The published calculator page (GitHub Pages). Generated; don't edit by hand. |
| `rounds/`, `rounds/state/` | Your class data: one workbook per round, plus the per-round state snapshots `play_round` writes. Git-ignored, and the only copy — back it up. |
| `engine-math-reference.pdf` | Instructor-only: the math behind each phase. |
| `stress_test*.py`, `notebook_e2e_test.py` | Regression suite (775 checks), including `stress_test_money_balance.py`, which plays six different trading worlds and fails if any fixed money strategy, or serial default, starts winning everywhere. |

## Quick start

```python
from engine import IPESimulation, PHASE1_COUNTRIES, PHASE1_GOODS
sim = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)

# Run twice: 1st writes a blank workbook, 2nd plays the round you typed into it.
sim.play_round("rounds/round01.xlsx", scale=1.4)
```

## Sizing it to your class

Six countries ship by default, hosting two MNCs each; both are adjustable. Aim
for **three students per country**, and hand each country's group its two firms
— with more students than firms, pair them as co-owners:

```python
from engine import build_firm_roster
firms = build_firm_roster(["Sabine", "Bosque", "Llano", "Trinity"])   # 8 firms
sim.upgrade_to_phase3(firms)
```

Then regenerate the paper handouts to match:

```
cd handouts && python make_handouts.py --countries Sabine Bosque Llano Trinity
```

The engine rejects a firm roster hosted in countries that aren't in play, so
these two country lists must agree. See *Setting up for your class size* in
`CLASSROOM_GUIDE.md`.

## Student calculator and round form

One static web page for the whole term. Students pick the phase and their
country, see what their allocation produces, fill in the rest of the round
(tariffs, trades, policies), and submit. The page saves a small decision file
and opens your Dropbox file request, which drops it into `rounds/inbox`.

```python
sim.export_calculator(inbox_url="https://www.dropbox.com/request/...")   # once
sim.inbox_report()             # who's in, while teams decide
```

The round cell then builds `roundNN.xlsx` from the files on its first run and
plays it on the second. An empty inbox means a paper round, as before. Publish
`docs/` with GitHub Pages (Settings → Pages → Deploy from a branch → `master`,
`/docs`). Re-export only after a shock that changes technology or endowments;
an export that changes nothing leaves the file alone. See *Student calculator
and round form* and *Collecting decisions from laptops* in
`CLASSROOM_GUIDE.md`.

## Tests

```
python stress_test.py && python stress_test_phase3.py && python stress_test_phase4.py \
  && python stress_test_phase5.py && python stress_test_phase6.py \
  && python stress_test_phase7.py && python stress_test_approval.py && python stress_test_classroom.py \
  && python stress_test_calculator.py && python stress_test_inbox.py \
  && python stress_test_money_balance.py \
  && python notebook_e2e_test.py
```
