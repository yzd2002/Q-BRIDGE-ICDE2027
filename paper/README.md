# Build the manuscript

The selected PDF has 12 body pages followed by AI disclosure and references (14 pages in total). From this directory, with an existing LaTeX installation:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Q-BRIDGE_ICDE2027.tex
bibtex Q-BRIDGE_ICDE2027
pdflatex -interaction=nonstopmode -halt-on-error Q-BRIDGE_ICDE2027.tex
pdflatex -interaction=nonstopmode -halt-on-error Q-BRIDGE_ICDE2027.tex
```

IEEEtran.cls, IEEEtran.bst, bibliography, figures and all text inputs are included. Build in a writable copy if this directory belongs to a sealed artifact. The original selected PDF identity is recorded outside the paper directory; regeneration may change PDF metadata without changing content.

The 2026-11-11 visual edition includes three vector figures and five consistently styled tables. Editable PDF/SVG/PNG and pinned drawing inputs for all three figures are in `../figure_assets/`. Original experiment results and table values are unchanged.
