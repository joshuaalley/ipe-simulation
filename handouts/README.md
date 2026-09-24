# Printable handouts

Print-ready PDFs for the IPE simulation — country instructions and decision
forms — so you stop relying on notebook cell output in front of the class.

## What to print

| File | What it is | When |
|---|---|---|
| `country-briefs.pdf` | One page per country: identity, endowments, factor intensities, and how the country's role grows across phases. | Hand to each group **once**, at the start. They keep it. |
| `forms-phase1-ricardo.pdf` | Per-country round form: labor allocation, tariffs, trade offers. | Phase 1 (Rounds ~1–3). One sheet per group per round. |
| `forms-phase2plus-trade.pdf` | Per-country round form: labor **and** capital allocation across three goods, tariffs, compensation, trade offers, and the MNC tax (Phase 3+). | Phase 2 onward — the core trade form for every later round. |
| `forms-firms.pdf` | One page per MNC: the decision slip (scale, relocate, export) plus a worked example in that firm's own numbers. | Phase 3+. Hand each owner (or pair) their firm's page; they hand in the slip and keep the example. The export line activates in Phase 4. |
| `forms-finance.pdf` | Per-country add-on: monetary regime (Phase 5), debt (Phase 6), institutions/WTO/hegemon (Phase 7). | Staple to the trade form once the relevant phase opens. |
| `how-to-submit.pdf` | One page: submitting a round from the class page (phase, country, decisions, Submit, drop into Dropbox). | Hand out once, the first time teams submit from laptops. |

Each form has a blank `ROUND ____` so one printout works for any round.

## Regenerating

The PDFs are generated from the live constants in `../engine.py` (endowments,
firm roster, money-growth choices, etc.). If you retune any of those, rebuild:

```
cd ipe-simulation/handouts
python make_handouts.py            # rewrites the .tex files
pdflatex -interaction=nonstopmode country-briefs.tex
pdflatex -interaction=nonstopmode forms-phase1-ricardo.tex
pdflatex -interaction=nonstopmode forms-phase2plus-trade.tex
pdflatex -interaction=nonstopmode forms-firms.tex
pdflatex -interaction=nonstopmode forms-finance.tex
```

(No second pass needed — there are no cross-references.)

## Matching a smaller class

Two flags size the handouts to your enrollment:

```
python make_handouts.py --countries Sabine Bosque Llano Trinity
```

- `--countries` prints briefs, trade forms and tariff rows for that subset only.
  Firm forms follow: **two MNCs per country**, so four countries print eight.
- `--firms` prints fewer MNC forms than that, if you want a thinner roster.
  Asking for more than two per country trims back to the cap.

The generator routes the firm list through the engine's `build_firm_roster`, so
the forms you hand out always match a roster `upgrade_to_phase3()` will accept.
Use the **same country list** here that you pass to the simulation; the engine
rejects any firm hosted in a country that isn't in play.

See *Setting up for your class size* in `../CLASSROOM_GUIDE.md` for how to
choose which countries to keep.
