# Training

This page explains the pieces in `jace.train` that a loop is written from:
`collect`, which plays games, the advantage calculation, the loss functions
and `apply`. It then explains the player's update rule, where exploration
comes from, and how `jace.loops.schedules` changes a loop's dials over time.

Nothing in `jace.train` decides what is trained when, where decks come from,
or how a dial changes. A loop decides those, and
[Write a loop](../guides/write-a-loop.md) walks through one.

## Collect

`collect` plays games in a `GameBatch` and records every decision.

```python
def collect(
    batch: GameBatch,
    pending: Pending | None,
    *,
    learner: Policy,
    opponent: Policy,
    guesser: Guesser,
    cards: int,
    history: int,
    decisions: int,
    generator: torch.Generator,
) -> Collected
```

The `learner` policy plays one seat of each game and the `opponent` policy
plays the other. `collect` draws the learner's seat at random for each new
game. It steps the batch until the games that have finished hold at least
`decisions` learner decisions.

At each step, `collect` does the following for every slot at once:

1. It asks the `guesser` for the acting seat's guess, and shows it that
   seat's last `history` guesses in the same game.
2. It runs each policy once, on the rows where that policy acts.
3. It samples an action from each row's odds.
4. It reads the opponent's true hidden cards from the batch, as labels for
   the guess.

Each model runs once per step over every game that needs it. The only Python
loop over games visits the few that ended on a step.

`collect` returns a `Collected` value with three fields:

| Field     | Holds                                                                             |
| --------- | --------------------------------------------------------------------------------- |
| `rollout` | The decisions of the games that finished.                                         |
| `games`   | One `GameResult` per finished game: its decks, the learner's seat and its result. |
| `pending` | What the next call on the same batch needs to continue.                           |

### What a rollout holds

A rollout is the decisions of finished games, one row per decision, in the
order they were made. Both seats of many games are interleaved in it.

| Field                      | Each row holds                                                                              |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| `boards`, `guesses`        | What the acting seat saw, and its guess.                                                    |
| `logits`                   | The policy's action scores, with illegal actions at `-inf`.                                 |
| `actions`, `log_probs`     | The action taken, and the log-probability the policy gave it.                               |
| `values`                   | The policy's predicted chance of winning.                                                   |
| `games`, `seats`           | The row's game and seat.                                                                    |
| `learner`                  | False where the opponent policy acted.                                                      |
| `results`                  | The game's final result from the row's seat: 1 for a win, 0.5 for a draw, 0 for a loss.     |
| `deck_truth`, `hand_truth` | The labels for the guess: the opponent's true unseen cards as shares, and 1 where it held the card. |

### Why unfinished games are carried

When `collect` has enough decisions, some games are still running. Dropping
them would favour short games, because a long game is more likely to be
running at the cut.

`collect` therefore hands those games back in `pending`. Pass `None` for a
batch that has just been started, and the previous call's `pending` after
that. A game that was running when one call stopped comes back whole from a
later call.

`jace.loops.toy` carries `pending` from one iteration to the next. The
limited loop starts a new batch every iteration with that iteration's decks,
so it passes `None` and discards the games still running. That matters
little while an iteration collects many games' worth of decisions.

## Advantages

An advantage says how much better a decision turned out than the policy
expected when it made it.

```python
def advantages(values: Tensor, results: Tensor, *, layout: Sequences, lam: float) -> Tensor
```

The only reward is the game's result, and there is no discount. Each step's
surprise is the change in the predicted chance of winning from one decision
to the same seat's next decision. The last step's surprise is the result
minus the last prediction. A decision's advantage is its own surprise plus
the later surprises of its seat in that game, each weighted by one more
factor of `lam`.

`lam` runs from 0 to 1. At 0, a decision is credited only with the next
step's surprise. At 1, it is credited with the whole rest of the game. This
is generalized advantage estimation.

`sequences(games, seats)` builds the `layout`: an index table that groups a
rollout's rows into one sequence per game and seat. The guess model's update
uses the same table to find each decision's earlier guesses.

## Losses and apply

A loss here is a plain function of tensors that returns a scalar. None knows
which model produced its input.

| Function      | Trains                           | What it does                                                                                 |
| ------------- | -------------------------------- | -------------------------------------------------------------------------------------------- |
| `player_loss` | The player                       | The clipped policy loss with two pulls and the win-chance error. The next section explains it. |
| `guess_loss`  | The guess model                  | Scores the guess against the opponent's true deck and hand. A confident wrong guess costs most. |
| `score_loss`  | The deck score                   | Scores each deck's logit against the games it won and lost.                                  |
| `choice_loss` | The drafter and the deck builder | Makes the chosen options more likely in proportion to `weights`.                             |

`choice_loss` covers three cases. Weights of one give ordinary
cross-entropy, which copies the choices. That trains the deck builder from
the decks the search preferred, and it would train a drafter from human
picks. Weights equal to the result minus the average result give the policy
gradient, which trains the drafter from game results.

Two more functions return one value per row for a loop to add to a loss.
`entropy(logits)` is higher when the odds are more even. `kl_to(logits,
reference)` measures how far the odds are from a reference, such as the
output of a drafter trained on human picks.

`apply` is the one place a gradient step happens:

```python
def apply(loss: Tensor, optimizers: Sequence[Optimizer], *, max_grad_norm: float) -> None
```

It zeroes every optimizer it is given, backpropagates the loss, clips the
gradient norm over all their parameters to `max_grad_norm`, and steps each
optimizer. Passing a model's optimizer and the embedding's optimizer
together is how the shared card embedding moves with that model.

Two updates need more than one call, so `jace.train` provides them.
`update_player` and `update_guess` each make one pass over a rollout, in
order, in slices of `slice_rows` rows, and call `apply` once per slice.
`update_guess` gives the guess model each decision's earlier guesses as they
were made during play, so the model learns from its own earlier guesses and
never from the truth.

## The player's rule

`update_player` follows `PlayerRule`, which holds the dials of the player's
update. The toy loop uses this one:

```python
RULE = PlayerRule(clip=0.2, stay=0.1, spread=0.01, keep_quantile=0.0, keep_floor=0.0, value_weight=0.5)
```

| Dial            | What it does                                                                                           |
| --------------- | ------------------------------------------------------------------------------------------------------ |
| `clip`          | Bounds how far one update may move an action's probability.                                            |
| `stay`          | Weighs a pull toward the policy that played the games.                                                 |
| `spread`        | Weighs a pull toward even odds over legal actions. A loop lowers it as training goes on.               |
| `keep_quantile` | Only decisions whose advantage size is at least this quantile of the learner's teach the policy.       |
| `keep_floor`    | A decision must also have an advantage size of at least this much to teach the policy.                 |
| `value_weight`  | Weighs the error of the predicted chance of winning.                                                   |

With `stay`, `spread` and `keep_quantile` at zero, the rule is plain PPO.

`update_player` chooses the decisions that teach the policy once, over the
whole rollout: the learner's own decisions whose advantage is large enough.
Every row trains the predicted chance of winning, kept or not.

It takes two values of `lam`. `lam_advantage` is used for the advantages
that teach the policy, and `lam_value` for the targets of the predicted
chance of winning. It returns three metrics: the mean `loss`, `kept_share`,
the share of decisions kept, and `advantage_size`, the learner's mean
advantage size.

## Exploration

Exploration is spread over four places, and each has its own dial.

| Where                  | How it explores                                                             | Dial                    |
| ---------------------- | --------------------------------------------------------------------------- | ----------------------- |
| Actions in a game      | `collect` samples each action from the policy's odds.                       | None                    |
| The player's update    | The `spread` pull keeps the odds over legal actions from collapsing.        | `PlayerRule.spread`     |
| Draft picks            | `model_drafter` samples picks, and the loop adds an entropy bonus.          | `Dials.drafter_entropy` |
| Which deck is played   | For a share of pools, the loop plays a deck other than the top-scored one.  | `Dials.explore_share`   |

The last two belong to the limited loop, and the pieces below it take no
part in them. `build_decks` returns every candidate and leaves the choice to
its caller.

## Dials on a schedule

A dial is a number that a loop reads each iteration. A schedule is a
function from the iteration number, counted from 1, to that iteration's
dials:

```python
type Schedule = Callable[[int], Dials]
```

How training changes over time is therefore one function you can read and
replace. `Dials` holds what a limited iteration reads:

| Dial              | Meaning                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `player_rate`     | The player's learning rate.                                                |
| `spread`          | The player's pull toward even odds. It replaces `PlayerRule.spread`.       |
| `drafter_rate`    | The drafter's learning rate. At zero, the loop skips the drafter's update. |
| `human_pull`      | The weight of the drafter's pull toward a drafter trained on human picks.  |
| `drafter_entropy` | The drafter's bonus for keeping pick odds spread out.                      |
| `explore_share`   | The share of pools for which a deck other than the top-scored one is played. |

Two small functions give a dial its shape. `power_decay` returns
`scale / iteration ** power`, kept within `[low, high]`. `ramp` returns
`first` up to iteration `start`, `last` from `end` on, and a straight line
between them.

`phased_schedule(drafter_starts=..., pull_ends=...)` builds a schedule from
them, as a starting point to edit:

- The player's rate and spread follow their curves from the first
  iteration.
- The drafter's rate stays at zero until `drafter_starts`, so the player's
  results mean something before they steer picks. It then ramps up over a
  tenth of that span.
- The drafter's pull toward the human drafter falls in a straight line from
  1 to 0 between `drafter_starts` and `pull_ends`.

### Where the numbers come from

The player's two curves follow Ataraxos, which lowers its learning rate and
its pull toward even odds as a power of the iteration number. The comments
in `schedules.py` cite Table S7 of its supplement. The other numbers are
first guesses with no source, to be replaced by measured values.

| Number                                         | Value                                           | Source      |
| ---------------------------------------------- | ----------------------------------------------- | ----------- |
| The player's learning rate                     | `0.5 / iteration ** 1.1`, within `[5e-6, 1e-4]` | Ataraxos    |
| The player's spread                            | `0.05 / iteration ** 0.3`                       | Ataraxos    |
| The drafter's learning rate after its ramp     | `1e-5`                                          | Placeholder |
| The length of the drafter's ramp               | A tenth of `drafter_starts`                     | Placeholder |
| The drafter's entropy bonus                    | `0.01`                                          | Placeholder |
| The explore share                              | `0.1`                                           | Placeholder |

No run has used this schedule for more than one iteration.
