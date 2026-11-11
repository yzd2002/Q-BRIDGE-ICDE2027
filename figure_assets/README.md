# Q-BRIDGE publication figures

All three manuscript figures use one visual system: navy typography, restrained blue/coral/purple/teal colors, thin rules, direct labels, and distinct timing-round markers. PDF and SVG contain vector paths and text. PNG files are 360-dpi previews; the manuscript includes the vector PDFs at 516-point width.

| Figure | Files | Meaning |
|---|---|---|
| 1 | `qbridge_overview.*` | One shared confidence view supports checking a proposed policy and optimizing over the full policy family. |
| 2 | `verified_family_v7_main.*` | All 234 matched timing pairs and the complete 107-view outcome decomposition, including zero-count categories. |
| 3 | `optimized_v6_compact_factorial.*` | All 60 group-round means and their 12 observed five-round ranges. |

The first figure is a constructed teaching example. Its set intersection is schematic; its interval bars use the exact displayed endpoints. The policy bar selects C with probability one, relative to a uniform reference over A/B/C/D. Matching policy and family bounds establish the best robust guarantee of 2.5 percentage points, not an observed treatment effect. The two query branches independently read the same fixed view. All numerical inputs and original experiment measurements are unchanged.

The experiment plots retain every original point, logarithmic horizontal axes, explicit ratio-one reference lines and timing-round shapes. Ranges show observed machine variation, not statistical confidence intervals. Figures 2 and 3 share the same color/marker mapping for timing rounds.

## Regenerate

In an environment with Python and Matplotlib (rendered with Python 3.12.3 and Matplotlib 3.10.5):

```bash
python -B generate_all.py --out regenerated
```

Use a new output directory. The generator verifies four SHA-256-pinned JSON inputs, draws all figures, checks exported point coordinates and canvas text bounds, and writes PDF/SVG/PNG plus `GENERATION_MANIFEST.json`. It does not run an experiment, solver, or statistical interval routine. `generate_main.py` is a compatibility entrypoint for this same three-figure renderer. PDF and PNG regeneration is byte checked in the recorded environment; SVG object identifiers may vary.

- `query_example.json` and `family_certificate.json`: original teaching records.
- `family_plot_data.json`: unchanged data from the verified family-LP figure, 234 points and four outcome counts.
- `factorial_plot_data.json`: unchanged compact factorial plot data, 60 group-round means and 12 ranges.

Historical result bundles retain their original presentation files for provenance. The current manuscript's figure PDFs are those in this directory and `paper/figures/`.

## Visual references

The layout is original. Actual figures and tables from the following primary papers were inspected to study reading order, restrained color, panel alignment, grouped table headers and direct annotation. No third-party artwork or data was copied into these figures.

- ICLR 2026 Outstanding Paper, *LLMs Get Lost In Multi-Turn Conversation*: [official award announcement](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/), [paper](https://arxiv.org/pdf/2505.06120), Figure 1 and Table 1.
- ICML 2026 Outstanding Paper, *The Flexibility Trap*: [official award announcement](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/), [paper](https://arxiv.org/pdf/2601.15165), Figures 1–2 and Table 1.
- NeurIPS 2026 Oral, *Generative Modeling via Drifting*: [authors' project page](https://lambertae.github.io/projects/drifting/), [paper](https://arxiv.org/pdf/2602.04770), Figures 1/3/4 and Table 1. Oral status is reported by the authors; no award is asserted.

Venue/status sources were checked on 2026-11-11. These references inform presentation, not the scientific claims of Q-BRIDGE.
