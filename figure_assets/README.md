# Q-BRIDGE publication figures

All three manuscript figures use one visual system: sans-serif typography, restrained navy/teal/orange colors, white backgrounds, thin rules, direct labels, and distinct timing-round markers. PDF and SVG contain vector paths and text. PNG files are 360-dpi previews; the manuscript includes the vector PDFs at 516-point width.

| Figure | Files | Meaning |
|---|---|---|
| 1 | `qbridge_overview.*` | One shared confidence view supports checking a proposed policy and optimizing over the full policy family. |
| 2 | `verified_family_v7_main.*` | All 234 matched timing pairs and the complete 107-view outcome decomposition, including zero-count categories. |
| 3 | `optimized_v6_compact_factorial.*` | All 60 group-round means and their 12 observed five-round ranges. |

The first figure is a constructed teaching example. It lists the original observational endpoints and two feasible success-rate scenarios. Exact decimal arithmetic checks that both scenarios satisfy all individual and group constraints; they illustrate, rather than enumerate, the confidence view. Policy C is compared with a uniform reference over A/B/C/D. Matching policy and family bounds establish the best robust guarantee of 2.5 percentage points, not an observed treatment effect. The two query branches independently read the same fixed view. All numerical inputs and original experiment measurements are unchanged.

The experiment plots retain every original point, logarithmic horizontal axes, explicit ratio-one reference lines and timing-round shapes. Ranges show observed machine variation, not statistical confidence intervals. Timing-round shapes agree across Figures 2 and 3. Figure 2 uses three teal shades for the rounds; Figure 3 uses orange for screening and teal for retention, with round identity encoded by shape. Figure 2 shows the complete 86/21 split in a count-proportional bar and keeps both zero-count categories explicitly labeled.

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

The five tables share navy headers, full-column alignment, grouped headers and booktabs rules. Table V groups its two views with pale bands. Table IV expresses policy bounds in percentage points; this is an exact unit conversion, with no change to measured outcomes. Short captions are followed by nearby definitions of units, denominators and experimental scope. Venue/status sources were checked on 2026-11-11. These references inform presentation, not the scientific claims of Q-BRIDGE.

Additional 2026 papers inspected in this edition (actual PDF pages, not only abstracts):

- ICLR Honorable Mention, *The Polar Express*: [official announcement](https://blog.iclr.cc/2026/04/23/announcing-the-iclr-2026-outstanding-papers/), [paper](https://arxiv.org/pdf/2505.16932v5), pages 2, 8, 9. Shared legends and redundant marker encoding.
- ICML Honorable Mention, *The Obfuscation Atlas*: [official announcement](https://blog.icml.cc/2026/07/05/announcing-the-icml-2026-awards/), [paper](https://arxiv.org/pdf/2602.15515v2), pages 2, 3, 6, 9. Aligned branches, results and verification.
- ICML Honorable Mention, *How much can language models memorize?*: [official publication](https://proceedings.mlr.press/v306/morris26a.html), final PDF pages 2 and 6. Axis/unit hierarchy and grouped table headers.
- NeurIPS E&D oral, *AgentAbstain*: [author project](https://agentabstain.github.io/), [paper](https://arxiv.org/pdf/2607.10059v1), pages 3, 4, 6, 8. Consistent small multiples and row grouping. Oral status is author-reported; no award is asserted.

Figure 2 places policy recovery before execution cost. Figure 3 uses separate feature and other-feature-state columns, so the four comparisons do not repeat long row labels. Colors, shapes, denominators and axis definitions remain explicit.

The overview uses descriptive headings and arrows rather than lettered panels, following the branch-aligned reading pattern inspected in The Obfuscation Atlas. Lettered panels remain only in comparative experimental figures where captions refer to distinct datasets or axes.
