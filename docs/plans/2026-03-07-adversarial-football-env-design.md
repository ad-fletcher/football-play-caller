# Adversarial Football Play-Calling Environment Design

## Overview

Two-agent adversarial RL environment where an offensive coordinator LLM and defensive coordinator LLM play against each other over the course of a football drive. Built on OpenEnv 0.2.1 as a single composite environment.

## Architecture: Single Composite Environment (Approach A)

One OpenEnv `FootballDriveEnv` with a composite `GameAction` containing both `OffenseAction` and `DefenseAction`. The environment resolves plays using the existing `PlayOutcomeModel` (two-stage: outcome classifier + quantile regressors). Training runs two separate REINFORCE loops with batch baselines, updating each model alternately.

- Single HF Spaces deployment
- PlayOutcomeModel already takes both offensive and defensive features
- Two-agent logic lives in the training orchestration

## Action Space

### Offense (Hierarchical)

**Level 1 — Formation:** `SHOTGUN`, `SINGLEBACK`, `EMPTY`, `I_FORM`, `PISTOL`, `JUMBO`, `WILDCAT` (7 choices)

**Level 2 — Play Type:** `run`, `pass`, `play_action`, `punt`, `field_goal` (5 choices)
- `isDropback` derived automatically (pass/play_action → true, run → false)

**Level 3 — Conditional:**
- If pass/play_action: `designedPass` (13 categories: deep/medium/short/screen × left/middle/right + none) + `receiverAlignment` (11 categories: 1x0 through 4x1)
- If run: `pff_runConceptPrimary` (13 categories: INSIDE ZONE, OUTSIDE ZONE, POWER, COUNTER, DRAW, etc.)
- If punt/field_goal: no sub-choices

**Contradiction penalties (-3):**
- punt/field_goal on 1st-3rd down
- designedPass specified on a run
- pff_runConceptPrimary specified on a pass
- Invalid formation + play type combos not seen in data

### Defense (Hierarchical)

**Level 1 — Formation:** 10 categories (3-4, 4-3, Nickel variants, Dime variants, 5-2, etc.)

**Level 2 — Man/Zone:** `Man` or `Zone`

**Level 3 — Coverage:** Constrained by man/zone choice:
- Man → Cover-0, Cover-1, Cover-1 Double, 2-Man
- Zone → Cover-2, Cover-3, Cover-4, Cover-6, Quarters, etc.

**Level 4 — Pass Rushers:** Integer, valid range depends on formation

## Observation Space

### Shared Base (both agents)

| Field | Type | Description |
|-------|------|-------------|
| down | int (1-4) | Current down |
| yardsToGo | int | Yards to first down |
| absoluteYardlineNumber | int (1-99) | Field position (1=own end zone, 99=opponent end zone) |
| quarter | int (1-4) | Current quarter |
| gameClock_seconds | int | Seconds remaining in quarter |
| score_diff | int | Offense score minus defense score |
| is_third_down_long | bool | down == 3 and yardsToGo >= 7 |
| red_zone | bool | absoluteYardlineNumber >= 80 |

### Asymmetric History (current drive only)

- **Offense sees** `defense_history`: list of defense's previous plays (defFormation, pff_passCoverage, pff_manZone, passRushers)
- **Defense sees** `offense_history`: list of offense's previous plays (offenseFormation, playType, designedPass/pff_runConceptPrimary)

### Post-Play Result (both see)
- `last_play_yards`: float
- `last_play_result`: normal, touchdown, interception, fumble_lost

## Game Logic

### Drive Start (randomized)
- `absoluteYardlineNumber`: sampled from real drive start distributions (typically 20-35)
- `quarter`: uniform 1-4
- `gameClock_seconds`: uniform within quarter (0-900)
- `score_diff`: sampled from real game score differentials
- Always 1st and 10

### Per-Play Flow
1. Both agents receive observation (asymmetric views)
2. Offense outputs hierarchical action
3. Defense outputs hierarchical action
4. Both bundled into composite GameAction
5. env.step(composite_action):
   - Validate actions → apply contradiction penalties if needed
   - Feed features into PlayOutcomeModel.predict() → (outcome, yards)
   - Update game state: field position, down/distance, clock (-40s per play)
   - Append to history, build new observation
6. Return observation with per-play rewards

### Drive End Conditions (done=True)

| Condition | Offense Reward | Defense Reward |
|-----------|---------------|----------------|
| Touchdown (field_position >= 100) | +7 | -7 |
| Field Goal made (probabilistic) | +3 | -3 |
| Field Goal missed | 0 | +3 |
| Punt (4th down choice) | 0 | +1 |
| Turnover on downs | -2 | +4 |
| Interception | -2 | +5 |
| Fumble lost | -2 | +5 |
| Safety (behind own goal line) | -2 | +2 |

### Field Goal Probability by Distance
- < 30 yards: ~95%
- 30-39: ~85%
- 40-49: ~75%
- 50-54: ~60%
- 55+: ~30%

## Reward Structure

### Offense (per-play + drive-end)
- Per play: `yardsGained * 0.1`
- First down conversion: `+1`
- Drive-end: see table above
- Contradiction: `-3`

### Defense (hybrid per-play + drive-end)
- Per play: `-yardsGained * 0.1`
- Drive-end: see table above
- Contradiction: `-3`

## Pydantic Models

```python
class OffenseAction(BaseModel):
    offenseFormation: str      # 7 choices
    playType: str              # run, pass, play_action, punt, field_goal
    designedPass: str = "none"
    receiverAlignment: str = "none"
    pff_runConceptPrimary: str = "none"

class DefenseAction(BaseModel):
    defFormation: str          # 10 choices
    pff_manZone: str           # Man, Zone
    pff_passCoverage: str      # constrained by manZone
    passRushers: int           # constrained by formation

class GameAction(Action):      # inherits from openenv Action
    offense: OffenseAction
    defense: DefenseAction

class PlayRecord(BaseModel):
    offenseFormation: str = ""
    playType: str = ""
    designedPass: str = ""
    pff_runConceptPrimary: str = ""
    defFormation: str = ""
    pff_passCoverage: str = ""
    pff_manZone: str = ""
    passRushers: int = 0

class GameObservation(Observation):  # inherits done, reward
    down: int
    yardsToGo: int
    absoluteYardlineNumber: int
    quarter: int
    gameClock_seconds: int
    score_diff: int
    is_third_down_long: bool
    red_zone: bool
    last_play_yards: float = 0.0
    last_play_result: str = ""
    offense_history: list[PlayRecord] = []
    defense_history: list[PlayRecord] = []
```

## Training: REINFORCE with Batch Baseline

Not using TRL's GRPOTrainer — two-agent coordination breaks its single-agent assumptions. Instead, custom REINFORCE with batch mean as baseline.

### Episode Collection
```python
def collect_episodes(env, off_model, def_model, n_episodes):
    offense_episodes = []  # (prompt, response, reward)
    defense_episodes = []

    for _ in range(n_episodes):
        obs = env.reset()
        drive_off, drive_def = [], []

        while not obs.done:
            off_prompt = format_offense_obs(obs)
            def_prompt = format_defense_obs(obs)
            off_response = offense_model.generate(off_prompt)
            def_response = defense_model.generate(def_prompt)
            obs = env.step(GameAction(offense=off_response, defense=def_response))
            drive_off.append((off_prompt, off_response, obs.offense_reward))
            drive_def.append((def_prompt, def_response, obs.defense_reward))

        # Add drive-end bonus to last play
        drive_off[-1] = (*drive_off[-1][:2], drive_off[-1][2] + drive_end_offense(obs))
        drive_def[-1] = (*drive_def[-1][:2], drive_def[-1][2] + drive_end_defense(obs))

        offense_episodes.extend(drive_off)
        defense_episodes.extend(drive_def)

    return offense_episodes, defense_episodes
```

### Policy Update
```python
def reinforce_update(model, tokenizer, optimizer, episodes):
    prompts, responses, rewards = zip(*episodes)
    rewards_t = torch.tensor(rewards)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    for prompt, response, advantage in zip(prompts, responses, advantages):
        log_prob = get_log_prob(model, prompt, response)
        loss = -(advantage * log_prob)
        loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```

### Training Loop
```python
for round in range(num_rounds):
    off_data, def_data = collect_episodes(env, off_model, def_model, batch_size)
    reinforce_update(off_model, off_tokenizer, off_optimizer, off_data)
    reinforce_update(def_model, def_tokenizer, def_optimizer, def_data)
```

## Existing Assets to Reuse
- `football_env/play_outcome_model.py` — two-stage outcome prediction (19 features, ~17ms/play)
- `data/outcome_classifier.joblib` + 20 quantile regressor joblib files
- `data/label_encoders.json` — valid categories for all categorical features
- `data/model_features.json` — feature metadata
- `football_env/server/app.py` — OpenEnv server scaffold (needs update)
- `football_env/client.py` — client scaffold (needs update)
- `train_grpo.py` — v1 training script (replace internals, keep infrastructure)
