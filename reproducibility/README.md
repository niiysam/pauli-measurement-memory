# Reproducibility bundle

All 151552 real nine-bit records from 16 completed jobs are retained as normalized copies. Outcomes, order, counts, source hashes and job identifiers are preserved. Copies omit account/instance metadata, not outcomes. Nothing has been publicly released and no repository DOI is claimed.

SOURCE_INDEX.json identifies the original 57 theory source/result files, 16 raw-record files, frozen readers/inputs/results, compiled QPY circuits and copied offline circuit-source dependencies. Historical cached target exports are not all bundled: QPY and timing manifests specify the circuits actually executed. Regeneration against a new device is not part of this package.

The portable verify_review_bundle.py has passed: all raw bitstrings match counts and both recent readers reproduce their frozen results exactly. The retained result is evidence/PACKAGE_QA/PORTABLE_READER_CHECK.json. It checks records and readers, not another quantum simulation. Run exact theory checks only in a fresh copy; original scripts protect against overwrite. No generic tests or production runs were repeated for manuscript preparation.

Reader usage: python verify_review_bundle.py. For a separate echo output: python echo27/analyze_echo27.py --input echo27/ECHO27_READER_INPUT.json --output /absolute/new_path/results.json. The output path must not exist.

The original source environment used NumPy/SciPy/Matplotlib and Stim for selected crosschecks; offline hardware preparation used Python3.12.14, Qiskit2.6.1, Aer0.18.0 and Runtime0.51.0. No account or QPU access is needed to read records. No submitter, key or saved account is included.

Old stationarity failures, record-gain null results, the 64-state protection-rate disagreement and the echo primary UNRESOLVED result remain negative evidence. Packaging does not change their status.
