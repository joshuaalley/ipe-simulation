# Printable handouts

Print-ready PDFs for the IPE simulation — country instructions and decision
forms — so you stop relying on notebook cell output in front of the class.

## What to print

| File | What it is | When |
|---|---|---|
| `country-briefs.pdf` | One page per country: identity, endowments, factor intensities, and how the country's role grows across phases. | Hand to each group **once**, at the start. They keep it. |
| `forms-phase1-ricardo.pdf` | Per-country round form: labor allocation, tariffs, trade offers. | Phase 1 (Rounds ~1–3). One sheet per group per round. |
| `forms-phase2plus-trade.pdf` | Per-country round form: labor **and** capital allocation across three goods, tariffs, trade offers. | Phase 2 onward — the core trade form for every later round. |
| `forms-firms.pdf` | One boxed form per MNC (scale, relocate, export). | Phase 3+. Hand each firm-owner their firm's box. The export line activates in Phase 4. |
| `forms-finance.pdf` | Per-country add-on: monetary regime (Phase 5), debt (Phase 6), institutions/WTO/hegemon (Phase 7). | Staple to the trade form once the relevant phase opens. |

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
