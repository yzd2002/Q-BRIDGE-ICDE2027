# Q-BRIDGE main figure

`qbridge_overview.pdf` is the publication vector figure. `qbridge_overview.svg` retains editable text, and `qbridge_overview.png` is a 240-dpi preview. The paper uses the PDF at its native 516-point width. Shapes, lines and text are vector objects; no generated raster illustration is used.

The figure explains one fixed confidence view and two query modes. It is a constructed teaching example, not an experimental result. The left panel combines observational intervals and randomized group means. The upper branch shows initial solving, successful evidence reuse, and a fresh solve after reuse is insufficient. The lower branch matches a policy lower bound with a family upper bound; this establishes the best robust guarantee, not the actual effect.

## Rebuild

Use the existing project environment with Matplotlib 3.10.5 (Python 3.12.3 was used for this rendering):

```bash
python generate_main.py --out regenerated
```

The output directory must not already exist. The generator checks both input SHA-256 hashes, reads the original saved query and family certificates, and exports PDF/SVG/PNG plus a generation manifest. It does not run an optimizer, experiment, or statistical interval routine. PDF and PNG reproduction has been checked in the recorded environment; SVG object identifiers may vary across invocations.

`inputs/` contains the unchanged teaching records. Displayed values are rounded; the saved certificates retain their original represented coefficients. The small bar at C denotes probability one, with zero probability on A/B/D; the reference assigns 25% to each action. All gain values are success-rate percentage points.

The layout is original. Visual organization was informed by inspecting primary full-text figures from [MICRO](https://arxiv.org/pdf/2603.13835), [LpBound](https://arxiv.org/pdf/2502.05912), and [Garnet](https://www.vldb.org/pvldb/vol19/p224-chandramouli.pdf); their illustrations are not copied or distributed here. Their awards are recorded by [ICDE 2026](https://icde2026.github.io/awards.html), [SIGMOD 2025](https://2025.sigmod.org/sigmod_awards.shtml), and [VLDB 2026](https://www.vldb.org/2026/conference-awards.html), respectively.
