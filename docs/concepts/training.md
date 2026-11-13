# Training

Training turns played games into gradient steps. This page follows one
round of that for the player and the guess model, then lists the losses for
the other models.

| Step            | Function                         | Produces                                  |
| --------------- | -------------------------------- | ----------------------------------------- |
| Play games      | `play_games`                     | A rollout: every decision of a list of games |
| Assign credit   | `advantages`                     | How much better each action was than expected |
| Score the model | `player_loss`, `guess_loss`      | A loss                                    |
| Learn           | `apply`                          | One gradient step                         |

`update_player` and `update_guess` run the last three steps over a rollout.

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
time, and when one ends the next of the list takes its place. No game is
cut off for being long, and none carries over to a later call. So every row
of a rollout was played by one version of the weights, and long games count
as much as short ones. A game the engine abandons is counted in
`collected.abandoned` and its decisions are left out.

A **rollout** is the decisions of those games, one row each: the board, the
guess, the action taken, the chance the player gave it, the predicted win
chance, and how the game ended. A row's game is its position in the list of
matchups. It also holds the opponent's true deck and hand at each decision,
which is what the guess model is scored against. When the opponent is the
learner's own policy, both seats' decisions are the learner's.

A rollout keeps only the board arrays the models read, which
`jace.play.BOARD_ARRAYS` lists. They are 110,608 of the 235,152 bytes the
engine encodes for one decision.

A loop depends on one thing for its games: a `PlayGames`, a function from a
list of matchups to a `Collected`. `play_games` with its other arguments
bound is one. The worker processes below make another.

## Playing games in worker processes

A step of the engine can take very long on a rare board, and Python cannot
interrupt it. Played in the training process, that step freezes the run.
`jace.train.workers` plays the games in worker processes instead:

```python
workers = start_workers(build_actor, count=6, seed=0)
collected, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=60.0)
```

`build_actor` runs inside each worker and returns its models and how it
starts a batch. `play_on_workers` deals the matchups out to the workers
and sends each one the weights. Every worker plays its share to the end
with `play_games`, in self-play.

A worker that completes no step for `stall_seconds` is killed. The games of
its share are lost and `collected.lost` counts them. The round returns what
the others played, and a new process takes its place for the next round.

A worker is always in one of three states, each its own type:

| State      | Means                              | Becomes                                  |
| ---------- | ---------------------------------- | ---------------------------------------- |
| `Starting` | Its process is building the models | `Ready` when it says so                  |
| `Ready`    | It waits for a round               | `Playing` when a round sends its share   |
| `Playing`  | It is playing its share of a round | `Ready` when done, or `Starting` as a new process when it stalls |

## Assigning credit

The only reward is the game's result. `advantages` spreads it back over the
decisions of each seat: an action gets credit when the predicted win chance
rose after it.

## The player's rule

`player_loss` is the clipped policy loss of PPO, with two pulls added:

| Dial            | What it does                                                   |
| --------------- | -------------------------------------------------------------- |
| `clip`          | Limits how far one update moves an action's probability        |
| `stay`          | Pulls toward the policy that played the games                  |
| `spread`        | Pulls toward even odds over legal actions, to keep exploring   |
| `keep_quantile` | Learns only from the decisions with the largest advantages     |
| `normalize`     | Shifts and scales the kept decisions' advantages to mean zero and unit spread |

This is the rule manabot adapted from Ataraxos. With `stay`, `spread` and
`keep_quantile` at zero it is plain PPO.

`update_player` goes over the rollout `passes` times, and `update_guess`
once. Both take the rows in a new random order on every pass. A rollout
lists its games one worker after another, so a slice of 512 rows cut in
play order holds about eight games of a round's 48, and its gradient
follows those games. Shuffled, every slice holds all of them. The
advantages are computed once, before the first pass. The win-chance error
trains only on the learner's own rows.

The entry points default to four passes and normalized advantages. One run
of each reached 46% against scripted greedy at 30 iterations with them and
36% without.

## The card embedding in self-play

Every model reads cards through the shared embedding. If one model's loss
steps it, every other model's input changes, including a model whose own
rate is zero. That was measured on a saved run: a drafter that was meant to
be frozen agreed with human picks 65.5% of the time before the run. With
the run's embedding it agreed 50.7%, though its own weights had barely
moved (64.2% on its original embedding).

So, as a working choice, the self-play loops do not train the embedding at
all. Pretraining on human data trains it as before. In self-play each model
adapts through its own projection of the card vectors. Whether the
embedding should learn from self-play, and how, is an open decision.

## The other losses

| Loss          | Trains                    | Given                                         |
| ------------- | ------------------------- | --------------------------------------------- |
| `guess_loss`  | The guess model           | The opponent's true deck and hand             |
| `score_loss`  | The deck score            | Games won and played by each deck             |
| `choice_loss` | The drafter, the builder  | The choices made, and a weight for each       |

`choice_loss` does two jobs. With weights of one it copies the choices,
which is how the builder learns the decks a search preferred. With weights
of "result minus the average" it is the policy gradient, which is how the
drafter learns from its decks' win rates.

`apply` takes the step for every optimizer it is given.

A model holds the embedding as a submodule, so `model.parameters()`
includes the embedding's. Build the model's optimizer from
`own_parameters`, which leaves them out:

```python
optimizer = torch.optim.Adam(own_parameters(model, embedding), lr=rate)
```

Pretraining on human data passes a second optimizer over
`embedding.parameters()` alongside it. The self-play loops pass none.

## Dials on a schedule

A schedule is a function from the iteration number to that iteration's
dials:

```python
schedule = phased_schedule(drafter_starts=2000, pull_ends=10000)
dials = schedule(iteration)
```

| Iteration | Player rate | `spread` | Drafter rate | Pull to human drafter |
| --------- | ----------- | -------- | ------------ | --------------------- |
| 1         | 1.0e-4      | 0.050    | 0, frozen    | 1.00                  |
| 2,200     | 1.0e-4      | 0.005    | 1.0e-5       | 0.98                  |
| 10,000    | 2.0e-5      | 0.003    | 1.0e-5       | 0                     |
| 40,000    | 5.0e-6      | 0.002    | 1.0e-5       | 0                     |

The player's rate and `spread` follow the formulas Ataraxos published. The
drafter's numbers are first guesses with no source.
