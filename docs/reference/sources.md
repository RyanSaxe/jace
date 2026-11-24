# Sources

This page lists the papers, repositories, datasets and documents behind
jace's design. Each entry gives what the source found and which part of jace
uses it. An entry marked "Decided against" is for a source that was read
and not followed, and it gives the reason. To add a source, put it under
the heading that fits, in the same form, in the same commit as the code
that uses it.

See [Gameplay RL](../concepts/gameplay-rl.md) and
[Limited models](../concepts/limited-models.md) for the current architecture.
Entries for earlier designs that jace moved away from are kept with notes on
what jace does instead.

## Game-playing RL and search

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815).
  Silver et al., 2017. One network supplies the move priors and the value
  for a tree search, and trains on the search's root visit counts and the
  game result. AlphaZero updates this network continually, where AlphaGo
  Zero let a new network play only once it beat the best one by a 55%
  margin.

  jace has no search yet. A search added to jace would have this shape,
  with one tree for each dealt world: one deal of the hidden cards,
  searched as if those cards were known. The continual updating is one
  reason the learning deck score trains from the first iteration.

- [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439).
  Anthony, Tian and Barber, 2017. In expert iteration a search guided by
  the network labels positions and the network learns the labels. Learning
  the search's visit distribution beat learning only its chosen move by
  about 50 Elo on Hex.

  If jace adds a search, the player would learn the search's visit counts,
  not only the action the search chose.

- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265).
  Schrittwieser et al., 2019. MuZero trains its value on the game result in
  board games and on n-step returns in Atari.

  In jace, the win chance's targets mix its later predictions with the
  game's result. A search's root value would not become a target.

- [Monte-Carlo Tree Search as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509).
  Grill et al., 2020. AlphaZero's visit counts are a coarse target when a
  search runs few simulations, and a regularized policy built from the
  search's values works better there.

  In jace, `jace.rl.losses.fork_targets_from_returns` turns paired
  counterfactual rollout returns at small simulation budgets into regularized
  softmax targets (`softmax(pair_returns / temperature)`) for
  `fork_pairwise_loss`.

- [Understanding the Success of Perfect Information Monte Carlo Sampling in Game Tree Search](https://ojs.aaai.org/index.php/AAAI/article/view/7562).
  Long, Sturtevant, Buro and Furtak, 2010. A search of sampled worlds, each
  treated as if the hidden cards were known, suffers strategy fusion, a
  term from Frank and Basin: it can choose different moves in worlds that
  the player cannot tell apart. It also overestimates the value of a
  position. It still plays well when hidden information comes out quickly
  and nearby outcomes agree.

  For jace, this paper is the case for a dealt-world search in Magic. It
  is also the reason the win chance should not learn from a search's root
  value, which such a search overestimates.

- [Information Set Monte Carlo Tree Search](https://eprints.whiterose.ac.uk/75048/).
  Cowling, Powley and Whitehouse, 2012. An information set is the set of
  positions a player cannot tell apart. Searching one tree over information
  sets, with a fresh deal on every pass, removes strategy fusion. This
  search beat separate dealt-world trees in Lord of the Rings: The
  Confrontation and only tied them in Dou Di Zhu.

  Decided against for jace's first search: it has to be built from scratch,
  and it tied the simpler search in the card game tested.

- [Ensemble Determinization in Monte Carlo Tree Search for the Imperfect Information Card Game Magic: The Gathering](https://eprints.whiterose.ac.uk/75050/).
  Cowling, Ward and Powley, 2012. An ensemble of searches over dealt worlds,
  with each move split into yes-or-no choices, played Magic competitively
  with an expert rule-based player on under one CPU second a move. The
  authors argue that Magic suits dealt worlds because hidden cards are
  revealed progressively.

  In jace, it is the main evidence for one tree for each dealt world, with
  each world's hidden cards dealt from the guess model's guess.

- [Scalable Decision Making for Games of Imperfect Information](https://arxiv.org/abs/2511.07312).
  Sokota et al., 2025. Ataraxos beat the most decorated human Stratego
  player. Its move network learns by self-play with a clipped update and
  two reverse-KL pulls. One pulls toward the policy that played the games.
  The other pulls toward a magnet policy, with a weight that falls over
  training. A belief network predicts the opponent's pieces, and it trains
  on the true hidden pieces from self-play games. Attention over time
  helped the belief network but not the policy.

  In jace, `jace.rl.losses.damped_policy_loss` implements this clipped update
  with dual reverse-KL pulls (`collection_kl` toward the behavior policy and
  `magnet_kl` toward `reference_probs`), paired with `jace.rl.select.top_fraction`
  advantage filtering and `jace.rl.schedules.DampedSchedule` (Table S7 curves).
  An earlier separate guess model was replaced by closed-decklist `v11`
  observations on the `ActorCritic` backbone. Batched counterfactual forks
  (`jace.forks.batch`) follow the short playout + regularized step pattern.

- [The Update-Equivalence Framework for Decision-Time Planning](https://arxiv.org/abs/2304.13138).
  Sokota et al., 2023. Planning at decision time can sample histories from
  a belief model in place of solving subgames. In Hanabi it matched
  public-belief search with about a hundredth of the search time.

  In jace, it is the theory behind the Ataraxos search above.

- [Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning](https://arxiv.org/abs/2206.15378).
  Perolat et al., 2022. DeepNash reached expert level at Stratego with
  self-play policy gradient and no search at play time. A pull toward a
  reference policy that is reset from time to time made it converge
  instead of cycling.

  In jace, it supports policy gradient as the player's base learner, with
  pulls toward other policies in its loss.

- [A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games](https://arxiv.org/abs/2206.05825).
  Sokota et al., 2022. Magnetic mirror descent regularizes each update
  toward a magnet policy. It solves for quantal response equilibria with
  linear convergence, and as self-play RL it was competitive with CFR in
  tabular games.

  In jace, `magnet_kl` in `damped_policy_loss` weighs the reverse-KL pull
  toward the full-support reference policy (`action_type_uniform` by default),
  and `DecaySchedule` or `DampedSchedule` lowers it over training.

- [Reevaluating Policy Gradient Methods for Imperfect-Information Games](https://arxiv.org/abs/2502.08938).
  Rudolph et al., 2025. Over more than 7,000 training runs on five games,
  methods based on fictitious play, double oracle and counterfactual regret
  did not beat generic policy gradient on exploitability.

  In jace, the player learns by policy gradient rather than by a regret or
  equilibrium method.

- [A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning](https://arxiv.org/abs/1711.00832).
  Lanctot et al., 2017. PSRO keeps a population of policies. A meta-solver
  picks the mix of opponents from a table of the payoffs between every pair
  of policies.

  Decided against: it needs that payoff table, and jace's player plays
  itself.

- [Combining Deep Reinforcement Learning and Search for Imperfect-Information Games](https://arxiv.org/abs/2007.13544).
  Brown, Bakhtin, Lerer and Gong, 2020. ReBeL searches over public belief
  states. A public belief state gives a probability for every hidden state,
  so the network's input grows with the number of hidden states. ReBeL is
  superhuman at heads-up no-limit hold'em.

  Decided against: a hold'em hand is one of 1,326, while an opponent's
  seven cards from a known 40-card deck are one of about 18.6 million. In
  limited, the opponent's decklist is hidden too.

- [Student of Games: A unified learning algorithm for both perfect and imperfect information games](https://arxiv.org/abs/2112.03178).
  Schmid et al., 2021. Growing-tree CFR with a learned value over public
  belief states plays chess, Go, poker and Scotland Yard.

  Decided against for the same reason as ReBeL: its public belief states
  also give a probability for every hidden state.

- [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164).
  Brown, Lerer, Gross and Sandholm, 2018. Deep CFR replaces tabular regrets
  with networks trained on sampled game-tree traversals, shown on heads-up
  limit hold'em.

  Decided against: the player learns by policy gradient, which Rudolph et
  al. found competitive.

- [Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680).
  Berner et al., 2019. OpenAI Five trained with PPO, playing its latest
  self 80% of the time and past versions 20%. Training on data from older
  parameters was slower, and reusing data even two or three times could
  halve its speed. It trained no drafting agent. Instead, it drafted heroes by
  searching over its win-probability head, which trained alongside the
  policy.

  In jace, every row of a complete-game rollout comes from one frozen version
  of the weights, so no data is stale. `jace.rl.pool.SnapshotPool` mixes
  one-seat games against past checkpoints (25% of updates by default).

- [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://doi.org/10.1038/s41586-019-1724-z).
  Vinyals et al., 2019. AlphaStar starts from supervised learning on human
  games and keeps a penalty for moving away from the supervised policy.
  During training only, its value function uses information from both
  players' views. Self-play alone "may chase cycles", so it trains against
  a league of past players and exploiters.

  In jace, `jace.rl.pool.SnapshotPool(matching="even_odds")` samples past
  checkpoints in proportion to `max(0.05, p * (1 - p))` using each snapshot's
  EMA win rate so self-play avoids cycling. An earlier separate hidden-card
  critic was replaced by the shared categorical WDL critic.

- [AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning](https://arxiv.org/abs/2308.03526).
  Mathieu et al., 2023. The agent reads units with a transformer and
  chooses a unit by comparing a query with each unit's embedding, so the
  number of choices can vary. For behaviour cloning, an agent with no
  memory did better than one with an LSTM.

  In jace, the drafter and the guess model score each card by a query's dot
  product with a projection of the card's embedding. The player has no
  memory of earlier decisions.

- [DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/abs/2106.06135).
  Zha et al., 2021. Self-play deep Monte Carlo trains a strong DouDizhu
  agent.

  In jace, it is a measure of how much experience a card-game agent uses,
  and the baseline PerfectDou is scored against.

## Policy-gradient training details

- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
  Schulman et al., 2015. GAE adds up the value's later temporal-difference
  errors, each weighted by lambda to the power of its distance. Lambda
  trades bias for variance.

  In jace, `jace.rl.credit.gae_wdl` computes undiscounted (`gamma=1.0`) GAE
  advantages with `lam_policy` and 3-class WDL lambda-returns with
  `lam_value`.

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
  Schulman et al., 2017. Clipping the probability ratio keeps one update
  close to the policy that collected the data, so several passes can reuse
  one batch.

  In jace, `jace.rl.losses.damped_policy_loss` is PPO's clipped loss with two
  reverse-KL pulls added, and `jace.rl.losses.ppo_loss` provides plain clipped
  PPO with an entropy bonus.

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561).
  Espeholt et al., 2018. V-trace corrects for actors that play with older
  weights than the learner by clipping importance weights.

  Decided against: jace's worker processes play every game of a round with
  the same weights, so no correction is needed.

- [What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990).
  Andrychowicz et al., 2020. Separate policy and value networks did better
  on four of five tasks, and going over the experience several times
  mattered.

  In jace, the win chance is its own network with its own optimizer, and
  the player's update makes four passes over each block of games by default.

- [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416).
  Cobbe, Hilton, Klimov and Schulman, 2020. Policy and value losses
  interfere in a shared network, and a value function tolerates much more
  reuse of its data than a policy does. One policy epoch was almost always
  near-optimal.

  Earlier jace prototypes used a separate value network. jace now uses the
  shared-trunk `ActorCritic` with 1 policy epoch per rollout by default, and
  `jace.forks.audit.build_batched_fork_aux_loss` reuses on-policy
  continuation states from counterfactual forks to supply extra WDL value
  supervision.

## Imperfect information and opponent modelling

- [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542).
  Pinto et al., 2017. The critic trains on the simulator's full state while
  the actor sees only its observations.

  Earlier jace prototypes used a separate `WinChanceModel` that read the
  opponent's true unseen cards during training. jace now uses a shared
  categorical WDL critic over closed-decklist observations.

- [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674).
  Baisero and Amato, 2021. A critic that reads only the true state gives a
  biased policy gradient. One that reads the agent's history as well as the
  state does not.

  In jace, the win chance reads everything the player reads of the board
  and its own deck, plus the hidden cards in place of the guess.

- [Contrasting Centralized and Decentralized Critics in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2102.04402).
  Lyu, Xiao, Daley and Amato, 2021. A critic with extra information gives
  the same expected update, with variance at least as large.

  In jace, whether the hidden cards make the advantages less noisy is not
  measured.

- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926).
  Foerster et al., 2017. COMA trains a separate critic on the global state
  during training only.

  In jace, it is one more precedent for a training-only critic with
  information the actors lack.

- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955).
  Yu et al., 2021. MAPPO's critic did best with agent-specific features
  plus global ones, and the global state alone lacked local information.

  In jace, the win chance reads the player's own view as well as the hidden
  cards, rather than the true state alone.

- [PerfectDou: Dominating DouDizhu with Perfect Information Distillation](https://arxiv.org/abs/2203.16406).
  Yang et al., 2022. Its value network reads the other players' hands in
  training, while the policy plays from its own view. Against DouZero it
  won 0.732 of its games, and the same agent without the hidden hands won
  0.717. The paper reports no variance.

  In jace, it is the closest card-game precedent for the win chance that
  sees hidden cards.

- [Suphx: Mastering Mahjong with Deep Reinforcement Learning](https://arxiv.org/abs/2003.13590).
  Li et al., 2020. Suphx starts from supervised learning on human game
  logs. Its oracle guiding first gives the policy the hidden tiles and
  then fades them out.

  In jace, the drafter, the deck builder and the deck score can start from
  checkpoints trained on human data. Oracle guiding was decided against:
  the player never sees hidden cards, and only the training-time win chance
  does.

- [Human-Level Competitive Pokémon via Scalable Offline Reinforcement Learning with Transformers](https://arxiv.org/abs/2504.04395).
  Grigsby et al., 2025. Metamon's transformer agents read up to 200 turns
  of a battle and ranked in the top 10% of active players.

  Decided against for jace's player: attention over a whole game costs the
  most to train. Instead, the guess model reads the recent events and its
  own earlier guesses, so what it inferred earlier in the game still
  informs each new guess.

## Learning from human data, and when each model learns

- [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835).
  Schmitt et al., 2018. A distillation pull toward a teacher speeds up a
  new agent, and tuning the pull's weight down lets the student surpass the
  teacher.

  In jace, the drafter's pull toward the human-trained drafter,
  `human_pull`, falls from 1 to 0 by `pull_ends`.

- [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://arxiv.org/abs/2206.11795).
  Baker et al., 2022. RL fine-tuning of a policy pretrained on human play
  kept a KL pull toward the pretrained policy, so the policy would not lose
  skills from pretraining. The pull's weight decays over training.

  In jace, `human_pull` decays over training for the same reason.

- [PIRLNav: Pretraining with Imitation and RL Finetuning for ObjectNav](https://arxiv.org/abs/2301.07302).
  Ramrakhya, Batra, Wijmans and Das, 2023. After behaviour-cloning
  pretraining, training the critic before the actor and ramping the
  learning rates improved RL fine-tuning.

  In jace, the drafter stays frozen while the player learns, then its rate
  ramps up.

- [Mastering the Game of No-Press Diplomacy via Human-Regularized Reinforcement Learning and Planning](https://arxiv.org/abs/2210.05492).
  Bakhtin et al., 2022. Self-play RL regularized toward a human imitation
  policy played well with humans, where self-play alone fell short.

  In jace, it is one precedent for the pull toward the human-trained
  drafter.

- [Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem](https://arxiv.org/abs/2402.02868).
  Wołczyk et al., 2024. A fine-tuned policy forgets what it knew on states
  it does not visit early in fine-tuning, and knowledge-retention methods
  prevent it.

  In jace, the drafter and the learning deck score are each pulled toward
  a frozen copy trained on human data, so they keep what that training
  taught them.

- [Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054).
  Kumar et al., 2022. Fine-tuning a whole pretrained model distorts its
  features. Training the head first, then everything, did better out of
  distribution.

  In jace, downstream deck models read a frozen `CardTable` from the
  pretrained card encoder, and the learning deck score trains with a KL pull
  toward the human-trained deck score.

- [Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms).
  Konda and Tsitsiklis, 1999. Actor-critic methods are two-time-scale
  algorithms: the critic learns faster than the actor.

  In jace, the learning deck score judges the builder's decks, as a critic
  judges an actor. It trains from the first iteration, and the builder
  learns only once the search reads that score.

- [A Finite Time Analysis of Two Time-Scale Actor Critic Methods](https://arxiv.org/abs/2005.01350).
  Wu, Zhang, Xu and Gu, 2020. Actor and critic both update at every step,
  the critic with the larger step size. An actor that moves too fast
  dominates the critic's error.

  jace orders its learners the same way: the learning deck score trains
  every iteration, at ten times the builder's learning rate.

- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500).
  Heusel et al., 2017. A generator and its judge, each with its own
  learning rate, converge when both learn throughout.

  In jace, it is another case of a judge that learns from the start.

- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477).
  Fujimoto, van Hoof and Meger, 2018. TD3 delays policy updates relative
  to the critic's.

  In jace, it shows the usual ordering, in which the actor waits for its
  judge. TD3 sets the wait by a fixed ratio of updates, not by a test of
  the actor's skill. jace's builder waits for a test of its judge instead:
  it learns only after the search has switched to the learning deck score.

- [What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret](https://arxiv.org/abs/2503.01491).
  Yuan et al., 2025. Pretraining the value model fixed a failure of PPO
  caused by the value's starting bias.

  In jace, it is one more case of a judge warmed up before the actor
  relies on it.

- [Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1702.08887).
  Foerster et al., 2017. When the other agents change, a learner that
  replays old experience learns from how they used to act. Decaying
  obsolete data, or marking its age, fixes replay.

  In jace, the learning deck score trains only on the decks of the last
  `--score-window` iterations, which recent versions of the player played.

- [The Primacy Bias in Deep Reinforcement Learning](https://arxiv.org/abs/2205.07802).
  Nikishin et al., 2022. Agents overfit their early experience and ignore
  later evidence.

  In jace, it is another reason the learning deck score trains on a recent
  window rather than on every game since the first iteration.

- [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685).
  Sohn et al., 2020. A pseudo-label is kept only when the model is
  confident, so a teacher is used only where it is reliable.

  In jace, the deck search reads the learning deck score only once that
  score's test loss beats the human-trained score's by a margin. The gate
  is on using a whole model, not on single examples.

- [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf).
  Bengio, Louradour, Collobert and Weston, 2009. A curriculum orders a
  learner's examples from easy to hard.

  In jace, the drafter's frozen start is not a curriculum: nothing gets
  easier, and the drafter waits until the player's games mean something.

## Card games, drafting and deck building

- [Mastering Strategy Card Game (Legends of Code and Magic) via End-to-End Policy and Optimistic Smooth Fictitious Play](https://arxiv.org/abs/2303.04096).
  Xi et al., 2023. The COG2022 winner uses one network for deck building
  and battle, trained on whole games. Trained on both stages together, the
  agent won 65.5% against a common opponent. Trained on the stages in turn,
  switching when the learner beat its past versions, it won 56.5% at the
  same compute. Each figure comes from one run. Sharing the card embedding
  between the stages helped.

  In jace, no model waits for a test of the player's skill, and every model
  reads the same card embedding. The player reads its own deck as card
  tokens, much as this agent pools the card embeddings of its deck.

- [Mastering Strategy Card Game (Hearthstone) with Improved Techniques](https://arxiv.org/abs/2303.05197).
  Xiao et al., 2023. One policy builds a 30-card deck and plays, with
  shared card embeddings. The learned hero choice collapsed to a single
  hero, so heroes were sampled instead.

  In jace, a learned choice can collapse early, so the builder and the
  drafter sample their choices and train with an entropy bonus.

- [Drafting in Collectible Card Games via Reinforcement Learning](https://www.sbgames.org/proceedings2020/ComputacaoFull/209690.pdf).
  Vieira, Tavares and Chaimowicz, 2020. A PPO agent learns to draft in
  Legends of Code and Magic. Its reward is +1, 0 or −1, the result of a
  fixed battle agent playing the drafted deck.

  In jace, the player that plays the drafted decks learns at the same time
  as the drafter. The drafter's reward is the win rate of its pool's best
  build.

- [AI solutions for drafting in Magic: the Gathering](https://arxiv.org/abs/2009.00655).
  Ward et al., 2020. On over 100,000 human drafts from Draftsim.com, a deep
  network imitated human picks better than heuristic and naive Bayes
  drafters.

  In jace, the drafter first learns to imitate human picks, from 17Lands
  data.

- [Q-DeckRec: A Fast Deck Recommendation System for Collectible Card Games](https://arxiv.org/abs/1806.09771).
  Chen et al., 2018. A deck search policy is learned by Q-learning, with
  win rates from simulated games between fixed greedy players. The authors
  name the fixed player as the method's limit.

  In jace, the learning deck score trains on the games of recent versions
  of the player, so its score means "in the hands of the current player".

- [Two-Step Reinforcement Learning for Multistage Strategy Card Game](https://arxiv.org/abs/2311.17305).
  Godlewski and Sawicki, 2023. Agents train first on a simplified game,
  then on the full one, with separate agents for separate decisions. The
  paper does not compare this with joint training.

  In jace, it adds no evidence for or against staged training.

- [Causal Reinforcement Learning for Complex Card Games: A Magic The Gathering Benchmark](https://arxiv.org/abs/2605.06066).
  da Costa Cunha et al., 2026. The paper builds a Magic environment with
  five fixed Standard archetypes, where masked PPO beats random play.

  For jace, it shows that published RL agents for Magic gameplay play fixed
  decks and are not tested on cards they have never seen.

- [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857).
  Arjona-Medina et al., 2018. RUDDER learns to split a delayed reward over
  the steps that caused it.

  In jace, credit for each pick is not built: all of a seat's picks share
  one reward, its best build's win rate. Credit for each pick would need
  a method like RUDDER.

- [RyanSaxe/mtg](https://github.com/RyanSaxe/mtg). RyanSaxe. A transformer
  drafter and deck builder trained on 17Lands data. Its README reports
  Mythic rank on Arena in 23 drafts at a 66% win rate.

  In jace, the drafter is a PyTorch rewrite of its design. It reads a whole
  draft as one sequence. Its pretraining weights each pick by the player's
  skill, the event's result and the pick's place in its round. The drafter
  is scored by top-1, top-2 and top-3 agreement with human picks. Unlike
  RyanSaxe/mtg, jace's drafter has no output layer with one slot per card,
  and its card vectors are not built from 17Lands statistics.

- [RyanSaxe/CubeCobraML](https://github.com/RyanSaxe/CubeCobraML). RyanSaxe.
  One shared encoder serves a drafter, a deck builder and a recommender, with
  cards as numeric indexes.

  In jace, every model reads a card through a vector built from the card's
  content, never one learned for its index, so a new card has a vector.

## Sets and subsets

- [Deep Sets](https://arxiv.org/abs/1703.06114). Zaheer et al., 2017. The
  paper gives the form a network needs to be invariant or equivariant to
  the order of its inputs.

  In jace, the deck builder reads a pool as a set of card tokens and scores
  every card, and the deck score reads one token per distinct card with its
  count.

- [Learning Neural Set Functions Under the Optimal Subset Oracle](https://arxiv.org/abs/2203.01693).
  Ou et al., 2022. EquiVSet learns per-element scores from pairs of a set
  and its best subset.

  In jace, it is the deck builder's shape. The builder's own set encoder
  reads the pool and scores every card in it, and the builder learns from
  pairs of a human pool and the deck built from it.

- [Order Matters: Sequence to sequence for sets](https://arxiv.org/abs/1511.06391).
  Vinyals, Bengio and Kudlur, 2015. The order chosen for a set's elements
  changes what a sequence model learns.

  In jace, `builds_from_decks` puts each human deck's spells and nonbasic
  lands in a random order, with a spell last, for teacher forcing.

- [Reparameterizable Subset Sampling via Continuous Relaxations](https://arxiv.org/abs/1901.10517).
  Xie and Ermon, 2019. Adding Gumbel noise to scores and keeping the top k
  draws k items without replacement, each in proportion to its weight.

  In jace, the builder draws its cards with the Gumbel-max trick, and a
  round's draws are Plackett-Luce over the copies left.

- [POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://arxiv.org/abs/2010.16011).
  Kwon et al., 2020. Many rollouts of one problem share a baseline, their
  mean reward, which gives low-variance REINFORCE.

  In jace, each build's reward is its win rate minus the mean win rate of
  its pool's builds that played.

## Card representation

- [Generalization to New Actions in Reinforcement Learning](https://arxiv.org/abs/2011.01928).
  Jain, Szot and Lim, 2020. An agent that sees each action through a
  learned representation can choose actions it never trained on.

  In jace, no model has an output layer with one slot per card. The drafter
  and the guess model score each candidate card by a query's dot product
  with a projection of its embedding. The deck builder scores each card of
  a pool from that card's own encoding.

- [Learning Action Representations for Reinforcement Learning](https://arxiv.org/abs/1902.00183).
  Chandak et al., 2019. Embedding a large action set lets a policy act in
  the embedding space.

  In jace, it is the same reason cards are scored by their embeddings.

- [Grounding Language for Transfer in Deep Reinforcement Learning](https://arxiv.org/abs/1708.00133).
  Narasimhan, Barzilay and Jaakkola, 2017. Text descriptions of entities
  let a policy transfer to new domains.

  Decided against: jace encodes each card from its `managym` engine program
  tokens instead of its printed text, since the program defines the card's
  exact mechanics in the engine's own vocabulary.

- [RTFM: Generalising to Novel Environment Dynamics via Reading](https://arxiv.org/abs/1910.08210).
  Zhong, Rocktäschel and Grefenstette, 2019. Generated games and manuals
  force an agent to read rather than memorize, and it generalizes to new
  dynamics.

  In jace, a card's vector depends only on its engine program, which is
  known before anyone plays it, and no card has an identity of its own.

- [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.07393).
  Hanjie, Zhong and Narasimhan, 2021. EMMA attends from each entity to its
  entry in a text manual and generalizes to unseen games.

  Decided against: jace builds each card's vector from its engine program
  tokens rather than a text manual embedding.

- [A Survey of Zero-shot Generalisation in Deep Reinforcement Learning](https://arxiv.org/abs/2111.09794).
  Kirk, Zhang, Grefenstette and Rocktäschel, 2021. Generalization is
  tested on held-out members of a family of environments.

  In jace, the models have not been tested on cards from other sets. Such
  a test would hold out whole cards or sets.

- [A Survey of Reinforcement Learning Informed by Natural Language](https://arxiv.org/abs/1906.03926).
  Luketina et al., 2019. It surveys agents that use language, including
  agents that read descriptions of entities.

  Decided against: jace reads each card through its structured engine
  program instead of natural-language rules text.

- [DraftFM: A FoundationModel for Day-Zero Drafting in Magic: The Gathering](https://arxiv.org/abs/2608.19568).
  Ward, 2026. Every card enters the model as a frozen vector of its printed
  record: structured features and a fixed text embedding. The model has no
  card or set identities and no usage statistics. Trained on 149 million
  picks from 29 sets, it agreed with 50.8%, 60.4% and 56.7% of human picks
  on three sets it never saw. Its code is at [brianward92/mtga](https://github.com/brianward92/mtga).

  In jace, the card encoder reads each card's engine program alone, with no
  17Lands statistics and no fixed per-card IDs. DraftFM is a point of
  comparison, not a target.

- [Learning With Generalised Card Representations for "Magic: The Gathering"](https://arxiv.org/abs/2407.05879).
  Bertram, Fürnkranz and Müller, 2024. The choice of card representation
  barely changes how well a model predicts human picks of known cards. For
  unseen cards it changes that a lot, and their model predicts 55% of those
  picks.

  In jace, accuracy on cards seen in training cannot choose the encoder.
  The test needs cards held out of training.

- [Cardsformer: Grounding Language to Learn a Generalizable Policy in Hearthstone](https://ebooks.iospress.nl/doi/10.3233/FAIA230581).
  Xia et al., 2023. A model first learns from recorded games to predict
  how card descriptions change the game state, then a policy trains on top.
  It plays well with cards it never trained on.

  Decided against for now: jace has no move-by-move records of human games
  to learn the first stage from.

- [Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited](https://arxiv.org/abs/2303.13835).
  Yuan et al., 2023. On new items a model with a learned ID per item fails,
  0.0125 HR@10 against 0.5899 for a text-based one. On known items a frozen
  text encoder with a trained head came close to IDs.

  In jace, a card's vector is built from its engine program rather than a
  learned per-card ID (replacing an earlier frozen text encoder).

- [Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization](https://arxiv.org/abs/2507.19473).
  Pembek, Fatkulin, Klenitskiy and Vasilev, 2025. A per-item table started
  from content and then trained freely did worse on new items: 0.435 HR@10,
  against 0.509 for frozen content plus a trainable offset with a bounded
  norm.

  In jace, no card has trainable weights of its own.

- [DropoutNet: Addressing Cold Start in Recommender Systems](https://proceedings.neurips.cc/paper/2017/hash/dbd22ba3bd0df8f385bdac3e9f8be207-Abstract.html).
  Volkovs, Yu and Poutanen, 2017. Zeroing the usage input for a random
  share of items in training teaches a model to score from content alone.

  Decided against: the model would still take 17Lands statistics as an
  input, and a new set has none. jace's card vectors use no statistics.

## Engine, data and tools

- [loopflowstudio/manabot](https://github.com/loopflowstudio/manabot).
  loopflowstudio, MIT licence, version 0.2.3. Its engine, managym, is a Rust
  rules engine built for RL: it steps many games in a batch, clones a game
  exactly and shows each player only the cards that player may see.
  manabot adds the board encoding, scripted opponents, rating math and a
  PPO rule adapted from Ataraxos.

  In jace, games run on `managym` with `manabot`'s `ActorCritic`,
  `ObservationSpace`, and rollout collector behind a narrow import seam
  (`jace.data.catalog`, `jace.rl.loop`, `jace.collect`, `jace.eval`,
  `jace.forks`, `jace.storage.gameplay`, and `jace.train`), while all RL
  algorithms live in `jace.rl`. Evaluation
  scores checkpoints against `managym`'s `Heuristic` and random players, and
  `jace.cards` reads each card's engine program through `jace.data.catalog`.

- [XMage](https://github.com/magefree/mage), [Forge](https://github.com/Card-Forge/forge)
  and [phase-rs](https://github.com/phase-rs/phase). magefree, Card-Forge
  and phase-rs. Open Magic engines that each implement more than 32,000
  cards.

  Decided against: none has a built-in RL interface. managym steps batched
  games, clones a game exactly and shows each player only the cards that
  player may see, and self-play needs all three.

- [17Lands public datasets](https://www.17lands.com/public_datasets).
  17Lands, Creative Commons Attribution 4.0. Draft and game records from
  Arena for each set, with per-card statistics, which the
  [metric definitions](https://www.17lands.com/metrics_definitions) page
  defines. First-pick data is missing for some sets.

  In jace, `jace.data` reads the drafts and decks. Of the statistics it
  keeps only average pick position, for a baseline picker. The Avatar
  first pick is restored as history and never trained or scored on.

- [MTG Arena draft](https://magic.wizards.com/en/mtgarena/draft). Wizards of
  the Coast. The page describes 14-card draft packs and events that end at
  7 wins or 3 losses.

  In jace, each seat of a `TLA_LIMITED` draft opens three boosters of 14
  cards and picks every card.

- [spells](https://github.com/oelarnes/spells). oelarnes, MIT licence,
  version 0.18 or later. It downloads 17Lands files, converts them to
  Parquet and adds MTGJSON card data.

  In jace, `uv run spells add TLA` downloads a set, and `read_set` reads it.

- [MTGJSON](https://mtgjson.com/). MTGJSON. Card data for every Magic
  card, including rules text.

  In jace, it supplies set card attributes (rarity, land type, and mana
  value) through spells.

- [bstaple1/MTGA_Draft_17Lands](https://github.com/bstaple1/MTGA_Draft_17Lands).
  bstaple1, MIT licence. A draft overlay whose README notes that Arena's log
  does not list the first pick until after the second.

  In jace, it explains why Avatar has no first-pick rows.

- [solean/p1p1](https://github.com/solean/p1p1). solean. Its README lists Avatar
  among the 17Lands exports that lack the opening pick, and notes that the
  second pick's pool contains the card taken first.

  In jace, `jace.data` recovers the first pick from the second pick's pool.

- [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  sentence-transformers, Apache 2.0. A small sentence embedder that gives
  384 numbers per text.

  Decided against: jace unified its card encoder on `managym` engine program
  tokens alone and removed frozen text embeddings.

- [Hugging Face Transformers](https://github.com/huggingface/transformers).
  Hugging Face, Apache 2.0, version 5.22.2 or later.

  Decided against: removed alongside `all-MiniLM-L6-v2` when the card
  encoder moved to engine program tokens alone.

- [PyTorch](https://pytorch.org/). The PyTorch Foundation, version 2.10 or
  later.

  In jace, every model and loss is written in it.

- Three PyTorch forum threads on typing `nn.Module.__call__`:
  [How to correctly typecheck nn.Module.\_\_call\_\_](https://discuss.pytorch.org/t/how-to-correctly-typecheck-nn-module-call/105821)
  (gchaperon, 2020),
  [how to get correct typing from nn.Module.\_\_call\_\_ ?](https://discuss.pytorch.org/t/how-to-get-correct-typing-from-nn-module-call/171577)
  (RmZeta2718, 2023) and
  [Why not type annotate Module to be Tensor->Tensor?](https://discuss.pytorch.org/t/why-not-type-annotate-module-to-be-tensor-tensor/170350)
  (Daniel-Chin, 2023). Calling a module returns a value the type checker
  cannot see as a tensor.

  In jace, code calls a module as `layer(x)`, as ordinary PyTorch code
  does, and the strict ty check ignores its `unsound-return-statement`
  rule. Where a call needs `forward`'s types or keyword arguments, the code
  calls `forward` itself.

- [uv](https://docs.astral.sh/uv/). Astral.

  In jace, every install and command runs through it.

- [RyanSaxe/byor](https://github.com/RyanSaxe/byor). RyanSaxe, Apache 2.0. It
  manages code checks as rules and scripts.

  In jace, the pre-commit hook and CI run byor's checks, among them
  [ruff](https://github.com/astral-sh/ruff), [ty](https://github.com/astral-sh/ty)
  and [ast-grep](https://github.com/ast-grep/ast-grep). This repository's
  own rules are in `.byor/rules/project/`.

- Experiment trackers: [TensorBoard](https://github.com/tensorflow/tensorboard),
  [MLflow](https://github.com/mlflow/mlflow), [Weights & Biases](https://github.com/wandb/wandb),
  [Aim](https://github.com/aimhubio/aim), [trackio](https://github.com/gradio-app/trackio),
  [DVCLive](https://github.com/treeverse/dvclive) and [ClearML](https://github.com/clearml/clearml).

  Decided against: with any of them, jace would still define a run's
  lineage and state itself, and plain files can be read without a service.
  `jace.storage.gameplay` writes Academy-native gameplay runs and
  `jace.storage.pretrain` writes file-locked pretraining runs as folders of
  plain files.
