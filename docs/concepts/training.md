# Training

Training turns played games into gradient steps. This page follows one
round of that for the player and the guess model, then lists the losses for
the other models.

| Step            | Function                         | Produces                                  |
| --------------- | -------------------------------- | ----------------------------------------- |
| Play games      | `collect`                        | A rollout: every decision of finished games |
| Assign credit   | `advantages`                     | How much better each action was than expected |
| Score the model | `player_loss`, `guess_loss`      | A loss                                    |
| Learn           | `apply`                          | One gradient step                         |

`update_player` and `update_guess` run the last three steps over a rollout.

## Playing games

```python
collected = collect(
    batch,
    pending,
    learner=player.policy,
    opponent=opponent,
    guesser=guess_model.guess,
    cards=len(pool),
    history=4,
    decisions=2048,
    generator=generator,
)
```

`collect` steps a `GameBatch` until finished games hold enough decisions by
the learner. Each model runs once per step, over every game. When the
opponent is the learner's own policy, both seats' decisions are the
learner's, so a game of self-play teaches twice as much.

A **rollout** is the decisions of finished games, one row each: the board,
the guess, the action taken, the chance the player gave it, the predicted
win chance, and how the game ended.

It also holds the opponent's true deck
and hand at each decision, which is what the guess model is scored against.

Games still running when `collect` stops are not thrown away. Dropping them
would favour short games. They come back in `collected.pending`, and the
next call continues them.

## Playing games in worker processes

The engine can stay inside one step for minutes when a board holds many
mana sources, and Python cannot interrupt it. Collected in the training
process, that step freezes the run. `jace.train.workers` plays the games in
worker processes instead:

```python
workers = start_workers(build_actor, count=12, seed=0)
gathered = gather(workers, weights=weights, decisions=8192, stall_seconds=60.0)
rollout, workers = gathered.rollout, gathered.workers
```

`build_actor` runs inside each worker and returns its `GameBatch` and its
models. `gather` sends every worker the weights, and each one runs
`collect` in self-play until its finished games hold its share of the
decisions. The rollout that comes back joins them, with game ids that are
unique across workers.

A worker that completes no step for `stall_seconds` is killed. It loses its
games in progress, the round returns what the others produced, and a new
process takes its place for the next round. `gathered.lost_workers` counts
them. Games in progress stay inside their worker between rounds.

A worker is always in one of three states, each its own type:

| State        | Means                                       | Becomes                                  |
| ------------ | ------------------------------------------- | ---------------------------------------- |
| `Starting`   | Its process is building the games and models | `Ready` when it says so                  |
| `Ready`      | It waits for a round                        | `Collecting` when `gather` sends weights |
| `Collecting` | It is playing a round                       | `Ready` when done, or `Starting` as a new process when it stalls |

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

`update_player` takes two more choices. `passes` is how many times it goes
over the rollout. With a `generator` each pass takes the rows in a new
random order, and without one it takes them in the order they were played.
The advantages are computed once, before the first pass. `update_guess`
always makes one pass in order.

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

`apply` takes the step for every optimizer it is given. The card embedding
has its own optimizer, passed alongside the model's.

A model holds the embedding as a submodule, so `model.parameters()`
includes the embedding's. Build the model's optimizer from
`own_parameters`, which leaves them out. Otherwise one loss steps the
embedding twice:

```python
optimizers = [
    torch.optim.Adam(own_parameters(model, embedding), lr=rate),
    torch.optim.Adam(embedding.parameters(), lr=rate),
]
```

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
