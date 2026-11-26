# Mac / server reproduction contract

## Locations and preservation

Mac is the editing/review and evidence-backup location. Linux is the numerical test/evaluation location. Keep author sources and historical runs unchanged. Use a new output directory for each smoke test, evaluation or new experiment. Do not synchronize with `--delete`.

Evaluation-only server-state exports are verified tensor-for-tensor against original checkpoints before copying; they do not replace the original full checkpoints or preserve optimizer/RNG state.

The 2026-11-26 repair workspace is `fuxian/reproduction_runs/audit_repair_20261126/`; its `backups/` and `evidence/` contain private local/server artifacts outside Git. The corresponding server folder is under `/root/autodl-tmp/fedvade_project/reproduction_runs/`. No password belongs in configuration, scripts or Git.

## Required inputs

- Explicit author source directory: `main.py`, `model.py`, `dataloader.py`, `experiment.py`; compare SHA-256 with the run configuration, not a guessed Git commit.
- UCI: the original author CSV, including the historical feature columns. Do not silently remove Breast Cancer's ID column.
- Fashion-MNIST: cached raw subset, seed/class counts, scaler convention and client partition. Raw IDX files were not found in the inspected experiment data directory; the current cache is retained and hashed, but its raw-train index mapping has not been independently recovered. Historical subset generation uses seed 0; partition uses seed 42 and a documented 0.15 candidate-swap jitter. This is our partition, not the unpublished FedHCA generator.
- ACD: pinned Fed-HIRE FCPL code and its dependencies; the archived frozen script still contains original server paths and is not a portable training CLI.
- Run configuration, executed source copies/diffs, exact invocation, package versions, source/data/partition hashes and checkpoint.

Missing historical fields must be marked unknown. A hash measured during this audit identifies the current artifact; it does not prove that artifact was executed historically unless a contemporaneous record matches it.

## Environment

Use the existing server GPU/CPU environment for verification. Record Python, torch/CUDA, NumPy, sklearn, scipy, pandas, thread limits, GPU and OS. A fresh environment requires torch, numpy, scikit-learn, scipy, pandas, munkres, tqdm, threadpoolctl; raw-data downloading additionally requires a compatible torchvision. The exported server environment is authoritative for this audit; Mac need not install CUDA packages.

## Commands

From the repaired code folder, using the server environment's Python:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
python scripts/common/evaluate.py --help
python scripts/lines/coverage_metric/main.py --metric-controls
```

Use explicit run, data and output paths for evaluation. `unique_full` evaluates the full input once per original record (identical feature vectors remain distinct records); `legacy_clients` preserves the repeated client-record population. Both are in-sample, not independent test performance. Their inference seeds and populations are recorded separately. Existing metrics are never overwritten without an explicit new output path.

Coverage evaluation uses separate generation and reference-split seed lists. Require `|G|=|R|`; report per-training-run results before aggregation. Identity and synthetic missing-class controls validate limited implementation properties, not complete validity on Fashion-MNIST.

## Checkpoint capabilities

A model state can support evaluation. Exact training continuation additionally requires client state, optimizers, schedulers, RNG state and data order/partition state. Inspect each checkpoint's actual keys; do not infer resumability from a filename containing `full`. Newly written repair checkpoints declare their capability explicitly.

## Validation boundary

The user approved inventory, backup, repair, isolated smoke tests and saved-weight evaluation (P1–P4). No new 300-round experiment or init×flow factorial experiment is part of this repair. GPU bitwise determinism is not assumed. Report repeated-run evidence only for the exact environment/configuration tested.

## Historical final-weight availability

The fine/C13 folder retains an explicitly labelled round-300 compressed full checkpoint (seed0). In the inspected C13_s2 directory, retained best checkpoints are rounds275/100; AUTHOR_s0 retains rounds250/150 and AUTHOR_s1 rounds250/125. No round-300 checkpoint was found in those inspected run directories. Historical final scores still have log evidence, but these candidate models cannot substitute for round300 in reevaluation. The inspected paths and payload keys are recorded in results/audit_20261126/reference_checkpoint_availability.json; external backups elsewhere on the host have not been exhaustively searched.

## Automatic code binding for new runs

New training/evaluation outputs record a content fingerprint of all repaired Python helpers, actual author source hashes, data/input hashes and package versions. Deploy a fixed code snapshot for a running job. The final saved_evaluation_final JSONs all match the final helper fingerprint. Earlier audit reevaluation JSONs predate the automatic package stamp; their input/checkpoint/source hashes and commands remain preserved, and no package identity is retroactively invented for them. The final metadata-only addition is separately regression-tested and checked against the historical runner on matched toy data.
