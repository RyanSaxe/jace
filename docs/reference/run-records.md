# Run records

Runs live as plain files on disk with no tracking service: Academy-native
folders for gameplay (`jace.storage.gameplay`) and locked, hash-verified folders
for pretraining (`jace.storage.pretrain`).

## Gameplay runs (`jace.storage.gameplay`)

`jace.train.{ten_decks,full_data,harness}` write to
`<runs_dir>/<experiment>/<run_id>/`:

```text
<runs_dir>/<experiment>/<run_id>/
  config.json                run config, deck counts, parameter count
  metrics.jsonl              one JSON object per train, eval, or fork event
  checkpoints/
    iter-0000.safetensors    ActorCritic weights + observation config
    iter-0000.policy.json    Academy PolicySpec pointing to the safetensors file
    resume-state.pt          atomic resume bundle (iteration, Adam, SnapshotPool)
  forensics/
    iter-0000.json           paired eval, ForensicsReport, probe diff, Academy commands
  games/
    train.jsonl              sampled training games (rank 0 only)
    eval-iter-0100.jsonl     recorded mirrored evaluation games
```

- **Atomic resume.** `save_resume_state` writes `resume-state.pt.partial` and
  atomically renames it to `resume-state.pt`. `load_resume_state` restores
  weights, optimizer buffers, and `SnapshotPool` EMA win rates to continue at
  `iteration + 1`.
- **Academy compatibility.** `save_policy_checkpoint` sets `player_configs=None`
  so checkpoints play any legal deck. Each `forensics/iter-XXXX.json` lists
  `play`, `replay_train`, and `replay_eval` commands.

## Pretraining runs (`jace.storage.pretrain`)

`jace.pretrain.{draft,decks}` write to `--run-dir/<run_id>/`, with `<run_id>` =
`<UTC timestamp>-<8-hex SHA-256 of identity>`:

```text
<run_dir>/<run_id>/
  identity.json              format (1), run_id, created, config, seed, commits, input refs
  state.jsonl                one line per state transition
  metrics.jsonl              one line per logged evaluation step
  evaluations.jsonl          evaluation records
  checkpoints/
    <model>-<update>.pt      CPU state dict (8-digit zero-padded update)
    index.jsonl              committed checkpoints (model, update, sha256)
  lock                       OS flock held while writing or evaluating
```

A sibling `<run_id>.curve.json` stores the learning curve and baselines. Writing
`NaN` or infinite values raises `ValueError`.

### Run states

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/run-states-dark.svg">
  <img alt="Pretraining run state transitions: created, running, completed, failed, and interrupted." src="../assets/run-states-light.svg">
</picture>

`writing(folder, clock=clock)` locks `lock` via OS `flock`, moves `created` or
`interrupted` to `running`, and yields a `Running` token for `log_update`,
`save_checkpoint`, and `record_evaluation`. On exit it records `completed`,
`interrupted`, or `failed`; after an unclean crash, `read_state` and `writing`
detect the unheld lock and append `interrupted`.

### Verified checkpoint loading

- `save_checkpoint` writes `.partial`, hashes it, renames atomically, and
  appends the SHA-256 digest to `checkpoints/index.jsonl`.
- `load_checkpoint(root, ref, into=model, keeping=embedding)` checks SHA-256
  before unpickling and skips `keeping` keys so shared card vectors stay intact.
- `jace.pretrain.common.load_card_table` loads a `CardEncoder` (freezing it into
  a `CardTable`) or an already-frozen `CardTable`.
