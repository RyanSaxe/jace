# Run records

A training run is recorded as plain files in one folder. `jace.tracking` is
the only code that writes them, and any program can read them. The layout
is a contract that a program such as a frontend can rely on.

```text
<root>/<run id>/
  identity.json        what the run is, written once
  state.jsonl          one line per state change
  metrics.jsonl        one line per update per model
  evaluations.jsonl    one line per evaluation
  checkpoints/
    <model>-<update>.pt  the model's weights
    index.jsonl          one line per checkpoint
  lock                 held by the process that is writing
```

A run id is the time the run was created, in UTC to the second, then a
dash and the first 8 hex digits of the SHA-256 of the run's identity, such
as `20261111T153045Z-1a2b3c4d`. Sorting ids as text therefore puts runs in
the order they were created.

A `<model>` is one of `embedding`, `drafter`, `deck_builder`, `deck_score`,
`guess`, `player` and `win_chance`. The win chance is the model that
estimates the acting seat's chance to win, for the player's update.

## Fields

Every `.jsonl` file contains one JSON object per line.

| File                | Fields                                                              |
| ------------------- | ------------------------------------------------------------------- |
| `identity.json`     | `format`, `run_id`, `created`, `config`, `seed`, `jace_commit`, `manabot_commit`, `inputs` |
| `state.jsonl`       | `state`, `time`, and `error` when the state is `failed`             |
| `metrics.jsonl`     | `model`, `update`, `metrics`                                        |
| `evaluations.jsonl` | `checkpoint`, `opponent`, `deals`, `rate`, `low`, `high`, and `details` when the evaluation recorded more |
| `index.jsonl`       | `model`, `update`, `sha256`                                         |

`inputs` lists the checkpoints from other runs that this run started from.
Each is an object with `run_id`, `model`, `update` and `sha256`. Following
`inputs` from run to run traces a run's lineage.

Times are ISO 8601 in UTC. No number in a record is NaN or infinite,
because `jace.tracking` raises `ValueError` instead of writing one.

## States

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/run-states-dark.svg">
  <img alt="The states of a run. A run is created, then running. From running it becomes completed when the writing block ends, failed on an error, or interrupted on Ctrl-C or when the process is killed. An interrupted run can resume to running. Completed and failed are final." src="../assets/run-states-light.svg">
</picture>

The last line of `state.jsonl` is the run's state, with one exception. A
process that is killed cannot record its own end. So when the last line is
`running` and no process holds the lock, the run is interrupted. `read_state`
and `writing` append the `interrupted` line when they find such a run. A
reader that calls neither can test the lock itself.

## Writing a run

```python
folder = create_run(root, identity, clock=clock)
with writing(folder, clock=clock) as run:
    log_update(run, model="player", update=1, metrics={"loss": 0.42})
    reference = save_checkpoint(run, model="player", update=1, module=player)
```

Every function that writes to a run takes the `Running` value that
`writing` yields, `run` above, so a write outside the `with` block does not
type-check. Leaving the block records how the run ended: `completed` on a
normal exit, `interrupted` on Ctrl-C, and `failed`, with the error, on any
other exception. `writing` raises `RunLockedError` while anything else
holds the run's lock, such as another process or another block in the same
process.

## Scoring a run that has ended

```python
with evaluating(run_folder) as run:
    record_evaluation(run, checkpoint=reference, opponent="heuristic", deals=50, rate=0.31, low=0.26, high=0.36)
```

`writing` raises `RunNotResumableError` for a run that is completed or
failed, so a finished run is scored with `evaluating` instead. `evaluating`
takes the same lock but leaves the run's state as it was, so it can add
evaluations to a run in any state. It raises `RunLockedError` while a
process is writing the run. `train_limited` and `train_player` also score
the player while they run, inside their own `writing` block.

`rate` is the checkpoint's win rate over `deals` paired deals. A deal is two
decks and an engine seed, played four times so that each player plays each
deck from each seat. `jace.loops.score` writes one line for each checkpoint
in each match it plays. The training loops write one for each scripted
opponent and each way the player chose its actions. These are the fields
of `details`:

| Field              | Contents                                                       |
| ------------------ | -------------------------------------------------------------- |
| `games`            | The games that finished                                        |
| `abandoned`        | The games the engine could not finish                          |
| `hung`             | The games that ran past `game_seconds`, whose process was killed |
| `seed`             | The seed of the scoring, from `jace.loops.score` only          |
| `game_seconds`     | How long a game could run before it counted as hung            |
| `games_per_second` | How fast the match was played, from `jace.loops.score` only    |
| `action`           | `sampled` when the checkpoint sampled its actions, and `top` when it took its top action |
| `guess`            | The guess model's `loss` and `base_rate_loss` on those games, with its `stated` and `observed` chances. A range of stated chance that no guess fell in is null |

When at least one match pits `random` against another player,
`jace.loops.score` also rates every player in its matches, with `random`
at 1000. It then writes one more line for each rated checkpoint, whose
`opponent` is `population`. That line's `rate` is the checkpoint's win
rate over all the other players it played. Its `details` contain `games`,
`abandoned`, `hung`, `seed` and `action`, as above, and also `rating` with
`rating_low` and `rating_high`, the `anchor` player rated 1000, and the
`opponents` it played.

## Reading a checkpoint

```python
reference = latest_checkpoint(run_folder, "drafter")
load_checkpoint(run_folder.parent, reference, into=drafter, keeping=embedding)
```

`latest_checkpoint` returns the reference of a model's newest checkpoint in
a run, the one with the highest update. `load_checkpoint` checks the file's
hash against the reference before loading it, and raises
`CheckpointMismatchError` for a file that does not match.

A model's checkpoint also contains a copy of the card vectors the model
was saved with. `keeping` is a module inside `into` that the load leaves as
it was. The checkpoint's keys under it are skipped, and every other key
must match `into`, or `load_checkpoint` raises `CheckpointKeysError` before
it loads anything. Loading a model with the shared card vectors as `keeping`
therefore leaves them unchanged. It also lets a drafter saved with a card
encoder load into a drafter that reads a table.

The `embedding` checkpoint of a drafter pretraining run contains a card
encoder, and that of every other run contains a card table.
`load_card_table` in `jace.loops.entry_points` loads either as a table. For
an encoder, it rebuilds the encoder with the shape given in the run's
config, loads it and freezes it into a table.
