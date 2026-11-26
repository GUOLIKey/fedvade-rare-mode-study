# Structured repair review

Scope: changes relative to a4f5901; approved P1–P4 only. This is a checklist review, not an external review or a claim that every historical experiment has been reproduced.

## Specification check

- Historical author source and run directories are not repair destinations; original artifacts have independent before/after hashes.
- Repaired standardization reproduces original input/partition tensors in all three inspected Fashion-MNIST settings.
- Legacy client-record evaluation remains explicitly available and matches all nine retained benchmark models exactly; unique-full evaluation is separately named.
- Training runs are isolated processes; schedule and reweighting switches alter executed copies rather than author source. Checkpoint capability is explicit.
- Independent source loading, arbitrary data paths, stale CSV rejection, final-round fallback, and shuffled sample/label pairing are repaired.
- Unsupported mechanisms and certainty claims are downgraded; real-data missing-class controls further retract the new-ratio exclusion argument.

## Quality and verification check

- Eleven targeted regression tests pass on the server; defects were observed before their repairs.
- Six real-model synthetic CPU integration variants pass with two pretrain/three loop rounds. Repeated author server tensors are identical, epoch and batch LR trajectories differ, and DP/sameK retain matching K.
- A matching CUDA smoke path completes without nonfinite events. This does not establish GPU bitwise determinism or formal research performance.
- Standalone evaluation, identity coverage and both component/decoder diagnostics execute from independent processes with explicit paths.
- JSON scientific outputs refuse to overwrite existing files; historical numbers and newly evaluated populations are not merged.
- The final metadata stamp binds helper-code content and dependency versions for new runs. A separate matched CPU toy comparison with the unmodified historical runner gives identical server/client tensors and zero metric deltas.
- Source syntax and current entry links are checked separately; these are not substitutes for numerical tests.

## Remaining boundaries

Archived ACD and Multi-CA full-loop workflows are not all consolidated into the new generic trainer. The ACD archive retains historical resource paths; its limitation is explicitly documented. No new end-to-end method experiment, init×flow factorial design, classifier-based generation assessment or 300-round rerun has been performed. Raw IDX mapping and some historical final checkpoints remain unavailable in the inspected locations. Evaluation-only model backups do not preserve optimizer/client/RNG state.

Disposition: repair code is suitable for the documented interfaces and tested scope. Research claims remain limited to the evidence in reports19–21; no universal method or metric-validity claim is made.
