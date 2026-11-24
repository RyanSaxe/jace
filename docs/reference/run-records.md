# Run records

jace stores every run as plain files on disk with no external tracking service. Gameplay loops write Academy-native folders (`jace.storage.gameplay`), and pretraining loops write file-locked, hash-verified folders (`jace.storage.pretrain`).

## Gameplay runs (`jace.storage.gameplay`)

`jace.train.ten_decks`, `jace.train.full_data`, and `jace.train.harness` write runs to `<runs_dir>/<experiment>/<run_id>/`:

```text
<runs_dir>/<experiment>/<run_id>/
  config.json                       run configuration, deck counts, parameter count
  metrics.jsonl                     one JSON object per train, eval, or fork event
  checkpoints/
    iter-0000.safetensors           portable ActorCritic weights + observation config
    iter-0000.policy.json           Academy PolicySpec pointing to the safetensors file
    resume-state.pt                 atomic resume bundle (iteration, Adam state, SnapshotPool)
  forensics/
    iter-0000.json                  paired eval, ForensicsReport, probe diff, Academy commands
  games/
    train.jsonl                     sampled training games (rank 0 only)
    eval-iter-0100.jsonl            recorded mirrored evaluation games
```

- **Atomic resume.** `save_resume_state` writes `resume-state.pt.partial` and renames it atomically to `resume-state.pt`. `load_resume_state` restores weights, optimizer buffers, and `SnapshotPool` entries with their EMA win rates so interrupted runs continue at `iteration + 1`.
- **Academy compatibility.** `save_policy_checkpoint` writes `player_configs=None` so any checkpoint plays any legal deck in `managym academy`. Each `forensics/iter-XXXX.json` includes ready-to-run `play`, `replay_train`, and `replay_eval` commands.

## Pretraining runs (`jace.storage.pretrain`)

`jace.pretrain.draft` and `jace.pretrain.decks` store runs under `--run-dir/<run_id>/`, where `<run_id>` is `<UTC timestamp>-<8-hex SHA-256 of identity>`:

```text
<run_dir>/<run_id>/
  identity.json                     format (1), run_id, created, config, seed, commits, input refs
  state.jsonl                       one line per state transition
  metrics.jsonl                     one line per logged evaluation step
  evaluations.jsonl                 evaluation records
  checkpoints/
    <model>-<update>.pt             CPU state dict (update zero-padded to 8 digits)
    index.jsonl                     committed checkpoints (model, update, sha256)
  lock                              OS flock held while writing or evaluating
```

A companion `<run_id>.curve.json` sits beside the run folder with the learning curve and baselines. Writing `NaN` or infinite values raises `ValueError`.

### Run states

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/run-states-dark.svg">
  <img alt="Run states in jace.storage.pretrain: created to running, then completed, failed, or interrupted. An interrupted run can resume to running; completed and failed are final." src="../assets/run-states-light.svg">
</picture>

`writing(folder, clock=clock)` acquires an exclusive OS `flock` on `lock`, moves `created` or `interrupted` runs to `running`, and yields a `Running` token required by `log_update`, `save_checkpoint`, and `record_evaluation`. On exit it records `completed`, `interrupted`, or `failed`. If a writer process dies without recording its exit, `read_state` and `writing` detect the unheld lock and append `interrupted`.

### Verified checkpoint loading

- `save_checkpoint` writes to `.partial`, hashes the file, renames it atomically, and appends its SHA-256 digest to `checkpoints/index.jsonl`.
- `load_checkpoint(root, ref, into=model, keeping=embedding)` checks the SHA-256 digest before unpickling and skips keys under `keeping` so loading a model never overwrites shared card vectors.
- `jace.pretrain.common.load_card_table` loads either a `CardEncoder` checkpoint (rebuilding and freezing it into a `CardTable`) or an already-frozen `CardTable`.
