# Data, code and third-party notices

## Q-BRIDGE-owned source code

Copyright (c) 2026 Q-BRIDGE authors. The unmodified standard MIT license text in
`LICENSE` applies only to Q-BRIDGE-owned Python and shell source listed by path
and SHA-256 in `CODE_LICENSE_SCOPE.json`, for the source files currently included in this repository. The complete archive
has its own separate path-and-hash license scope for historical copies and its verifier. The MIT text is sourced from the
[Open Source Initiative](https://opensource.org/license/mit).

This is a code-only license scope. It does not relicense Coat data or derived
records, IEEEtran or other third-party materials, paper text, tables, figure
images, manuscript files, or the archive as a whole. No additional rights for
those excluded materials are granted by the code license. Existing applicable
licenses and copyright notices remain in force. Python dependencies are installed
separately at the locked versions and retain their own licenses; no Python
environment or scientific dependency wheels are vendored here.

## Coat data and data-derived records

Coat is by Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak,
and Thorsten Joachims, *Recommendations as Treatments: Debiasing Learning
and Evaluation*, ICML 2016, PMLR 48:1670–1679.
[Paper](https://proceedings.mlr.press/v48/schnabel16.html) ·
[author project](https://www.cs.cornell.edu/~schnabts/mnar/) ·
[official data ZIP](https://www.cs.cornell.edu/~schnabts/mnar/coat.zip) ·
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
The author project explicitly links that license (rechecked 2026-11-05).

The separate complete artifact retains the four input files and original README as unchanged bytes. Derived
case records use binary rating >= 4. The original case uses independent uniform
one-rating thinning per user and source; the separately registered full-block
diagnostic retains all ratings and aggregates group scores within users.
Original row and column positions are retained. Attribution and the
noncommercial restriction also accompany these data-derived records. No original
Coat author code is included. No endorsement by the data authors is implied.
The complete archive preserves all unsuccessful, unknown and refused outcomes;
repackaging does not create new empirical evidence.

## IEEEtran

The unchanged `icde2027/paper/IEEEtran.cls` (v1.8b) and `IEEEtran.bst` (v1.14)
retain their original contribution, copyright and LPPL 1.3 license headers.
They are not Q-BRIDGE-owned code and are excluded from the MIT code license.
The original class and bibliography README notices, verbatim LPPL 1.3 text,
and the complete unmodified upstream distribution are included in
`third_party/IEEEtran/`. Both included class/style files were byte-compared
against their counterparts inside that upstream ZIP.

[Official CTAN package](https://ctan.org/pkg/ieeetran) ·
[complete upstream ZIP](https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip) ·
[LPPL 1.3](https://www.latex-project.org/lppl/lppl-1-3.txt) ·
[maintainer site](https://www.michaelshell.org/tex/ieeetran/).
The upstream bundle's own materials keep their respective notices; no claim of
IEEE or maintainer endorsement of Q-BRIDGE is made.

## Repository and complete artifact

This repository contains a lightweight runnable source subset. Its own
`SOURCE_MANIFEST.json` and `CODE_LICENSE_SCOPE.json` refer to files actually
present here. The complete release archive retains the full provenance, all
scientific results, historical source and its separate exhaustive manifests.
The root README will provide download and verification instructions when that archive is published. No private
paper-review or submission-contact directory is distributed here.
