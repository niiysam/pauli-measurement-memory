# Pauli measurement memory

Code and data for **Entanglement barriers and purification in frozen and slowly refreshed Pauli measurement circuits**.

Hancheng Zhou — Polessky State University; Guangdong University of Education.
Correspondence: samvniiy@icloud.com.

Version 1.0.0. DOI: **10.5281/zenodo.23700827** (published archive).
Repository: https://github.com/niiysam/pauli-measurement-memory

## Contents

- `reproducibility/project/`: retained exact-check scripts, results, frozen plans and stabilizer source files.
- `reproducibility/hardware_raw/`: all 151,552 real nine-bit records from 16 IBM jobs, normalized without account metadata. Outcomes, order, shot counts and job identifiers are retained.
- `reproducibility/echo27/` and `diagnostic49/`: frozen readers, input and results.
- `data/numerics/`: 1,024 protocol records representing 512 independent parent pairs, trajectory-level summaries, seeds, configuration, convergence failures and censoring. Large NPZ histories are not included; their hashes and regeneration seeds are indexed in the manifest.
- `figures/v4/`: six figure groups. Figures are derived from the archived evidence, not new experiments.
- `reproducibility/figures_v4/build_figures.py`: figure script. Run from a fresh checkout; it writes figure and provenance outputs.
- `SHA256SUMS`: file integrity manifest.

## Minimal local use

Python 3.12 was used in the source environment. To check the recorded shots and reproduce the two frozen readers:

```bash
python reproducibility/verify_review_bundle.py
```

The record reader uses the Python standard library; optional installed packages are only reported. Figure regeneration needs NumPy, SciPy and Matplotlib. Exact stabilizer crosschecks additionally use Stim. No IBM account or hardware is required. No live submission program or credential is included.

Exact-check scripts preserve their original overwrite guards. Run them only in a fresh copy after reading each frozen configuration. This release does not launch production automatically.

## Scope and negative results

The frozen dictionary is the nonergodic c=1 endpoint. Its analytic bounds do not establish a phase-boundary shift for c=0.6. The finite-memory lifetime-growth primary contrast remains UNRESOLVED. The hardware primary half-recovery contrast also remains UNRESOLVED; the hardware experiment is an instrument control, not a realization of the random chain. Historical stability failures and other negative checks remain retained.

## Licensing

Original project source code is provided under the MIT license in `LICENSE_CODE`. Original numerical and normalized experimental data, plots and archive documentation are provided under CC BY 4.0 in `LICENSE_DATA`. Third-party dependencies retain their respective licenses and are not vendored here. No journal PDF, school template or EndNote application database is included.
