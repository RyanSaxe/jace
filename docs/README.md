# Documentation

jace learns to draft, build decks and play Magic: The Gathering by self-play.
It is at an early stage, and [Status](status.md) says what works today.

## Getting started

- [Install](getting-started/install.md) lists the requirements and explains
  how to install jace, run the tests, run the example and install the
  pre-commit hook.

## Concepts

- [Architecture](concepts/architecture.md) explains what each module owns,
  which module imports which, the three state machines and the `GameBatch`
  interface.
- [Models](concepts/models.md) explains the shared card embedding, the
  drafter, the deck score, the deck builder, the guess model and the player.
- [Training](concepts/training.md) explains game collection, advantages, the
  losses, the player's update rule, exploration and schedules.

## Guides

- [Write a loop](guides/write-a-loop.md) walks through the limited loop step
  by step and explains how to change the reward, the explore rule and the
  other seats.

## Reference

- [Run records](reference/run-records.md) describes each file a run writes
  to disk, and the states a run moves through.
- [Status](status.md) lists what works today, how it was checked, and what
  is not built yet.

## Change jace

- [Contributing](contributing/README.md) explains how to set up a checkout,
  run the checks, follow the design and testing rules, and add a lint rule.
