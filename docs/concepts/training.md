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
the learner. Each model runs once per step, over every game.

A **rollout** is the decisions of finished games, one row each: the board,
the guess, the action taken, the chance the player gave it, the predicted
win chance, and how the game ended. It also holds the opponent's true deck
and hand at each decision, which is what the guess model is scored against.

Games still running when `collect` stops are not thrown away. Dropping them
would favour short games. They come back in `collected.pending`, and the
next call continues them.

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

This is the rule manabot adapted from Ataraxos. With `stay`, `spread` and
`keep_quantile` at zero it is plain PPO.

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
