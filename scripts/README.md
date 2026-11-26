# Code capabilities after the audit repair

The consolidated scripts are a repaired interface, not the historical executed version. Original scripts/configs/logs are archived separately with hashes. Historical result numbers are not automatically results of this implementation.

## Shared modules

- `data.py`: cached raw Fashion-MNIST, UCI author CSVs, standardization, our K-Means-fragment partition and loaders. Partition assignment includes seeded 0.15 candidate-swap jitter. `standardize` preserves the historical `(array, reference_or_None)` contract.
- `provenance.py`: explicit source/data lookup, hashes and non-overwriting JSON output.
- `train.py`: one subprocess per run, author/DP/sameK local fitting, effective scheduler patch, optional server reweight patches, diagnostics and evaluation-only checkpoints. It does not implement every historical ACD/Multi-CA training runner.
- `local_gmm.py`: DP fit/pruning and same-K ordinary GMM. The DP component cap and fitting seed remain implementation parameters; the precheck CLI exposes them. Full training currently uses cap15/fitseed42.
- `evaluate.py`: saved-model prediction using either `legacy_clients` or `unique_full`, with explicit data and output paths. Reports ARI/Purity/NMI/ACC, individual draws and inference-seed SD (ddof0).
- `model.py`, `sampling.py`, `distances.py`, `metrics.py`: explicit author-model construction, server generation, nearest-neighbour distances and clustering metrics.

## Lines and actual entry points

| Directory | Capability |
|---|---|
| `fmnist_benchmark/` | `prepare`, `check`, `run`, `driver`, `eval`; full new runs only in new output folders |
| `latent_width/` | Width-sweep driver; historical results still require original checkpoints/configs |
| `client_dpgmm/` | `dump`, `precheck`, `driver`, `eval`; running a driver is new training, not part of this repair |
| `coverage_metric/` | Saved-model distance ratios, explicit generation/split seeds, identity and synthetic missing-class controls |
| `diagnostics/` | Saved-model component/reconstruction diagnostics; global-tau oracle values remain historical diagnostics, not causal class-capability evidence |
| `client_acd/` | Archived frozen experiment implementation with historical server paths and external FCPL dependency; not a portable/common training CLI |
| `server_multica/` | Algorithm library, not a run/driver CLI; includes normalized-Bures complete-linkage, an optional centroid-linkage branch, a separate plain raw-W2 average-linkage comparison, optional weight-floor sanitation, and conditional component sampling/ownership metadata; a function existing is not evidence that an experimental arm was run |
| `server_reweight/` | Source-patch library; invoked by common/train via `--sample-floor`, `--usage-floor`, `--usage-beta` |

The training arguments `--local-gmm author|dpgmm|samek_gmm` change local construction. `--schedule author|epoch` changes actual client scheduler placement. Server sample-floor rejects budgets too small to cover all pooled components; it never silently changes the total sample count.

## Evaluation without training

From the code folder in the server environment:

```bash
python scripts/common/evaluate.py --run RUN --data-dir DATA --population unique_full --rounds 300 --output NEW_OUTPUT.json
python scripts/common/evaluate.py --run RUN --data-dir DATA --population legacy_clients --rounds 300 --output OTHER_NEW_OUTPUT.json
python scripts/lines/coverage_metric/main.py --metric-controls
python scripts/lines/coverage_metric/main.py --run RUN --data-dir DATA --generation-seeds 0 1 2 --split-seeds 0 --output COVERAGE_NEW.json
python scripts/lines/diagnostics/main.py mode-loss --run RUN --data-dir DATA --output DIAGNOSIS_NEW.json
```

An old config without `data_dir` requires an explicit path. New configs record data/source/input/partition hashes. Standalone commands load the run's own source explicitly. Existing JSON outputs are not overwritten. `unique_full` counts every original input observation once; it does not merge identical feature vectors.

Historical Fashion-MNIST coverage used one generation seed and one split seed. The new CLI distinguishes these from prediction inference seeds; multi-generation uncertainty is reported within each training run. Three repeated draws are not three independent trainings.

## Verification

`tests/test_regressions.py` covers the repaired contracts. `tests/server_smoke.py` uses a synthetic dataset, two pretrain rounds and three formal-loop rounds to check plumbing in isolated outputs; these are smoke tests, not research experiments. `tests/server_verify_saved.py` compares historical and repaired input/partition tensors, preserves legacy metrics and adds unique-sample metrics for saved weights.

Historical BC/yeast/ACD/Multi-CA full-loop runners remain in the evidence archive. Do not imply those implementations were merged solely because a core helper was moved.
