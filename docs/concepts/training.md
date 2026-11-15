# Training

Training turns played games into gradient steps. This page follows one
round of that for the player and the guess model, then lists the losses for
the other models.

| Step            | Function                         | Produces                                  |
| --------------- | -------------------------------- | ----------------------------------------- |
| Play games      | `play_games`                     | A rollout: every decision of a list of games |
| Assign credit   | `advantages`                     | How much better each action was than expected |
| Score the model | `player_loss`, `win_chance_loss`, `guess_loss` | A loss                      |
| Learn           | `apply`                          | One gradient step                         |

`update_player` runs the last three steps over a rollout, for the player
and the win chance. `update_guess` runs the last two, for the guess model,
which needs no advantages.

## Playing games

A round of training is a list of matchups, each a pair of decks.
`play_games` plays every game of the list from its first decision to its
last:

```python
collected = play_games(
    start,
    matchups,
    learner=player.policy,
    opponent=opponent,
    guesser=guess_model.guess,
    cards=len(pool),
    history=4,
    generator=generator,
)
```

`start` makes a `GameBatch` for the list. The batch plays a few games at a
time, and when one ends, the next of the list takes its place. `play_games`
cuts no game off for being long, and no game continues into a later call. So
no decision in a rollout was made with an earlier version of the weights,
and long games count as much as short ones. A game that the engine cannot
finish, or cuts short itself, is counted in `collected.abandoned`, and its
decisions are left out.

A **rollout** contains the decisions of those games, one row each. A row
contains the board the acting seat saw, its guess, the action taken with
the log-probability the player gave it, and the game's result for that
seat. A row's game is its position in the list of matchups. Each row also
contains the opponent's true deck and hand at that decision, which the
guess model is scored against and the win chance reads. When the opponent
is the learner's own policy, both seats' rows are marked as the learner's.

A rollout contains only the board arrays the models read, the ones that
`jace.play.BOARD_ARRAYS` names. They are 115,088 of the 235,152 bytes the
engine encodes for one decision.

A loop gets its games from a `PlayGames`, a function that takes a list of
matchups and returns a `Collected`. `play_games` with its other arguments
bound is one such function, and a function that calls `play_on_workers`,
below, is another.

## Playing games in worker processes

A step of the engine can take very long on a rare board, and Python cannot
interrupt it. If the training process played the games itself, that step
would stop the whole run until the step returned. So `jace.train.workers`
plays the games in worker processes:

```python
workers = start_workers(build_actor, count=24, seed=0)
collected, workers = play_on_workers(
    workers, matchups, weights=weights, stall_seconds=60.0, device=torch.device("mps")
)
```

`build_actor` runs inside each worker and returns its models and the
function that starts a batch. `play_on_workers` deals the matchups out to
the workers in turn and sends each worker the weights. Every worker plays
its share to the end with `play_games`, in self-play. The decisions are
returned on `device`, the device the models train on. Each worker's
decisions are moved there as soon as they arrive, while the other workers
are still playing.

Each worker runs on one thread. On a 16-core M3 Max, the play models'
matrix products do not speed up with more cores. One process alone ran
about 10,000 products of a typical size a second, and 4 to 16 processes
together ran about 22,000. The most decisions a second came from more
workers than cores, each playing its whole share at once. With 24 workers
of 9 games each, a block of 200 games took about 7.4 s, where 8 workers of
8 games took about 9.6 s. Two threads a worker, or fewer games at once on
more workers, were slower. So `--workers` and `--games-per-worker` default
to 24 and 9. Each worker uses about 1 GiB, so 24 workers take about 25 GiB.

A worker that completes no step for `stall_seconds` is killed. The games
of its share are lost, and `collected.lost` counts them. `play_on_workers`
returns what the other workers played, and a new process takes the killed
worker's place for the next round. If no worker returns its games,
`play_on_workers` raises `WorkerFailedError`.

A worker is always in one of three states, each its own type:

| State      | Means                              | Becomes                                  |
| ---------- | ---------------------------------- | ---------------------------------------- |
| `Starting` | Its process is building the models | `Ready` when the process reports that it has built them |
| `Ready`    | It waits for a round               | `Playing` when `play_on_workers` sends it a share of a round |
| `Playing`  | It is playing its share of a round | `Ready` when it sends back its games, or `Starting`, as a new process, when it stalls |

## Assigning credit

The only reward is the game's result, and `advantages` divides the credit
for it among the decisions of each seat. An action gets credit when the
predicted win chance rose after it. `lam_advantage` sets how much of the
later change an action is credited with. At 0, an action is credited only
with the change up to the seat's next decision, and at 1 with every change
to the end of the game.

The win chance comes from its own model, which reads the board and the
opponent's true hand and library. The player has no win chance of its own.

At the start of an update, `update_player` computes the win chance of
every row, without gradients. From those predictions it computes the
advantages, and also the win chance's own targets, the lambda-returns with
`lam_value`. A lambda-return mixes the later predictions and the result in
the same way that `advantages` mixes them, with `lam_value` in place of
`lam_advantage`.

## The player's rule

`player_loss` is the clipped policy loss of PPO, with two pulls added. A
pull is a KL divergence toward another policy, added to the loss with a
weight. `PlayerRule` contains the settings:

| Setting         | What it does                                                   |
| --------------- | -------------------------------------------------------------- |
| `clip`          | Limits how far one update moves an action's probability        |
| `stay`          | The weight of the pull toward the policy that played the games |
| `spread`        | The weight of the pull toward even odds over legal actions, which keeps the player exploring |
| `keep_quantile` | Leaves out of the loss the decisions whose advantage size is below this quantile of the learner's advantage sizes |
| `keep_floor`    | Leaves out of the loss the decisions whose advantage size is below this value |
| `normalize`     | Shifts and scales the advantages so that the kept decisions' have mean zero and standard deviation one |

manabot adapted this rule from
[Ataraxos](../reference/sources.md#game-playing-rl-and-search), a Stratego
agent. With `stay`, `spread`, `keep_quantile` and `keep_floor` at zero, it
is plain PPO.

`update_player` and `update_guess` each go over the rollout `passes` times,
and both take the rows in a new random order on every pass. They shuffle
because a rollout lists its games one worker after another. In
`train_player`, which plays 48 games an iteration by default, a slice of
512 rows cut in play order would contain rows of only about eight games,
and its gradient would come from those games alone. A shuffled slice contains rows
of all of them.

The advantages are computed once, before the first pass. On each slice,
`update_player` steps the policy on `player_loss` and then the win chance
on `win_chance_loss`, its squared error to the targets, each with its own
optimizer. The win chance trains only on the learner's own rows, because
it estimates the chance to win when the learner's policy plays the seat.

`train_limited` and `train_player` default to four passes and normalized
advantages, through `--passes` and `--normalize`. One run with these
settings reached 46% against scripted greedy at 30 iterations, and one run
without them reached 36%.

## The card embedding in self-play

Every model reads cards through the shared embedding. If a step on one
model's loss changed the embedding, every other model's input would change
too, even that of a model whose own learning rate is zero. This was
measured on a saved run of an earlier version, in which other models'
losses still stepped the embedding. A drafter that was meant to stay
frozen agreed with human picks 65.5% of the time before the run. Read
through the embedding as the run left it, the same drafter agreed only
50.7% of the time. Through its original embedding it still agreed 64.2%
of the time, so its own weights had barely moved.

So, as a working choice, only the drafter's pretraining on human picks
trains the card encoder. Deck pretraining and the self-play loops read its
vectors as a `CardTable`, which has no parameters, and each model adapts
through its own projection of the card vectors. Whether the card vectors
should learn from self-play, and how, is an open decision.

## The other losses

| Loss          | Trains                    | Given                                         |
| ------------- | ------------------------- | --------------------------------------------- |
| `guess_loss`  | The guess model           | The opponent's true deck and hand             |
| `score_loss`  | The deck score            | Games won and played by each deck             |
| `choice_loss` | The drafter               | The choices made, and a weight for each       |

`choice_loss` makes each chosen option more likely in proportion to its
weight, and it trains the drafter in two ways. On human picks, every weight
is positive, so the loss trains the drafter to make the same picks. A pick's
weight comes from the player's skill, the event's result and the pick's
place in its round. In self-play, a pick's weight is its deck's win rate
minus the mean of those win rates over the iteration's picks, which makes
the loss the policy gradient.

The deck builder's losses are built on `build_log_probs`, each build's
log-probability under the builder's own sampling. Pretraining lowers the
negative log-probability of the human builds, each weighted by its event's
win rate. Self-play weights each build's log-probability by its win rate
minus the mean of its pool's builds, and adds an entropy bonus. The deck
score that learns from self-play adds `kl_to`, a pull toward the frozen
human-trained score, to its `score_loss`. [Deck building](deck-building.md)
describes both, and when each starts.

`apply` takes one gradient step. It clips the norm of the gradient of all
the given optimizers' parameters together, then steps each optimizer.

A model contains the embedding as a submodule, so `model.parameters()`
includes the embedding's parameters. Build the model's optimizer from
`own_parameters`, which leaves them out:

```python
optimizer = torch.optim.Adam(own_parameters(model, embedding), lr=rate)
```

The drafter's pretraining passes `apply` a second optimizer, over the
encoder's parameters. Every later loop reads a table with no parameters,
so it passes no second optimizer.

## Dials on a schedule

A dial is a value that a schedule sets for each iteration, such as a
learning rate or the weight of a loss term. A schedule is a function from
the iteration number to that iteration's dials:

```python
schedule = phased_schedule(drafter_starts=2000, pull_ends=10000)
dials = schedule(iteration)
```

With `phased_schedule`, the drafter stays frozen through iteration
`drafter_starts`, so that the player's results mean something before the
drafter learns from them. Its rate then ramps up over a tenth of that many
iterations. Its pull toward the human-trained drafter falls in a straight
line, from 1 at `drafter_starts` to 0 at `pull_ends`.

| Iteration | Player rate | `spread` | Drafter rate | Pull to human drafter |
| --------- | ----------- | -------- | ------------ | --------------------- |
| 1         | 1.0e-4      | 0.050    | 0, frozen    | 1.00                  |
| 2,200     | 1.0e-4      | 0.005    | 1.0e-5       | 0.98                  |
| 10,000    | 2.0e-5      | 0.003    | 1.0e-5       | 0                     |
| 40,000    | 5.0e-6      | 0.002    | 1.0e-5       | 0                     |

The player's rate and `spread` follow the formulas Ataraxos published. The
learning deck score's rate is 1e-4 and the builder's 1e-5 at every
iteration. The builder starts learning when the deck-score check switches
the search to the learning deck score, and that switch is not a phase of
the schedule. The drafter's and the deck models' numbers are first guesses
with no source.
