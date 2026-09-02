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
| `engine-math-reference.pdf` | Instructor-only: the math behind each phase. |
| `stress_test*.py`, `notebook_e2e_test.py` | Regression suite (439 checks). |

## Quick start

```python
from engine import IPESimulation, PHASE1_COUNTRIES, PHASE1_GOODS
sim = IPESimulation(PHASE1_COUNTRIES, PHASE1_GOODS, phase=1)

# Run twice: 1st writes a blank workbook, 2nd plays the round you typed into it.
sim.play_round("rounds/round01.xlsx", scale=1.4)
```

## Sizing it to your class

Six countries and thirteen firms ship by default; both are adjustable. Aim for
**three students per country**, and give each student one MNC:

```python
from engine import build_firm_roster
firms = build_firm_roster(["Sabine", "Bosque", "Llano", "Trinity"], n_firms=11)
sim.upgrade_to_phase3(firms)
```

Then regenerate the paper handouts to match:

```
cd handouts && python make_handouts.py --countries Sabine Bosque Llano Trinity --firms 11
```

The engine rejects a firm roster hosted in countries that aren't in play, so
these two country lists must agree. See *Setting up for your class size* in
`CLASSROOM_GUIDE.md`.

## Tests

```
python stress_test.py && python stress_test_phase3.py && python stress_test_phase4.py \
  && python stress_test_phase5.py && python stress_test_phase6.py \
  && python stress_test_phase7.py && python stress_test_classroom.py && python notebook_e2e_test.py
```
