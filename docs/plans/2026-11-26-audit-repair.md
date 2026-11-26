# FedVaDE audit repair implementation plan

> Execute sequentially; the user approved P1–P4 on 2026-11-26. Do not run P5 formal training.

**Goal:** Preserve historical evidence, correct unsupported conclusions, repair the consolidated code, and independently evaluate existing checkpoints on the server.

**Architecture:** Mac is the code/review and backup location; the server is the numerical test/evaluation location. Work in this isolated checkout and a new server audit directory. Preserve author sources, historical runs, and all unrelated/NNDL files. Retain historical metrics alongside newly named evaluations.

**Tech stack:** Python, PyTorch, NumPy, scikit-learn, unittest, SSH, rsync, SHA-256.

1. Inventory server data, source, environment, configurations, results, and final checkpoints; hash and copy lightweight evidence and selected complete final-checkpoint groups to Mac. Do not invent missing historical metadata.
2. Add regression tests that reproduce the standardization, data path, evaluation population, source import, scheduler, and diagnostic pairing defects. Run the failing tests on the server before fixing code.
3. Restore the data contract; isolate training runs in subprocesses; explicitly resolve author modules and data; retain legacy evaluation and add a unique-full-data evaluation; validate scheduler and reweighting patches; preserve sample-label pairing. Record execution/data hashes and checkpoint capabilities.
4. Make coverage generation seeds explicit, enforce sample matching, and add identity and missing-class positive controls. Keep outputs outside historical run directories during verification.
5. Correct README and annotate historical reports. Add conclusion status, artifact provenance, and Mac/server reproduction instructions. Resolve numerical conflicts from original outputs when available; otherwise mark unknown.
6. Run regression and small isolated integration smoke tests on the server, then evaluate existing checkpoints without training. Save commands, environment, hashes and results; synchronize evidence to Mac and verify checksums.
7. Review changes and evidence, publish the repaired local checkout into the original local repository without pushing externally, and report exact remaining limitations.

## Acceptance checks

- Author source and pre-existing historical result/checkpoint hashes remain unchanged.
- Regression tests fail on the original consolidated code and pass on the repair.
- Independent eval/coverage commands work from a fresh process with arbitrary data/output directories.
- Legacy metrics remain distinguishable from unique-sample metrics; generation and inference seeds are separate.
- No formal training, no destructive synchronization, no external upload, no NNDL changes.

## Implemented backup boundary

The local backup contains evaluation-only server-state exports for the selected model groups rather than full training-state archives. This supports frozen-model reevaluation, not exact resumption. Original full checkpoints remain on the server; the existing Mac full C13-s1 reference was independently SHA-verified. This limitation is explicit in the delivery and reproduction notes.
