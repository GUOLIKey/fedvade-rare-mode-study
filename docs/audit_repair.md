# Audit repair, 2026-11-26

Approved scope: P1–P4 from the review discussion. The original repository snapshot is a4f5901. A separate local worktree and server directory isolate all changes and new outputs. Author sources, historical checkpoints/logs and unrelated projects are preserved.

## Established defects

- Consolidation changed `standardize` from a two-value historical return to an array while retaining tuple unpacking.
- Standalone source imports and explicit data-directory propagation were missing.
- `--schedule epoch` was recorded without changing execution; sampling/loss switches were advertised without being connected.
- Diagnostic loaders were traversed separately for x and y, despite shuffling.
- Client-record evaluation changes sample weighting across partitions; it must remain named as legacy population.
- Import caching is a risk when crossing author/patched sources in one process. Default DP→sameK uses a shared patched block and is not, by itself, evidence of wrong results.

## Evidence boundaries

Existing server outputs will be inventoried and copied with SHA-256 checks. Missing original outputs remain unknown. Revised evaluation is stored beside audit evidence, never written into historical run folders. Recovered numeric provenance, exact test results and integrity checks are recorded in `results/audit_20261126/` when available.

No historical experiment is declared invalid solely because the consolidated code has a regression. Conversely, historical scores do not validate the repaired implementation without an explicit comparison.
