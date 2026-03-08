# Football Play-Calling RL Environment

## Project Context
OpenEnv Hackathon (March 7-8, 2026) at Shack15, SF. Build an RL environment where an LLM acts as an NFL offensive coordinator, choosing formation + play type each down. Outcomes sampled from real NFL data.

- **Hackathon Track**: Statement 3.1 (World Modeling — Professional Tasks), Secondary: Statement 2 (Long-Horizon Planning)
- **Must deploy**: HF Spaces using OpenEnv 0.2.1
- **Must include**: TRL/Unsloth GRPO training script in Colab
- **Must include**: 1-minute demo video uploaded to YouTube
- **Repo must be public** (open source requirement)
- **All work must be new** — no prior work allowed

## Schedule
- **Sat Mar 7**: Doors 9AM, Hacking 11:30AM, Lunch 1PM, Dinner 6PM, Doors close 10PM
- **Sun Mar 8**: Doors 9AM, Submissions due 1PM, First judging 1:15PM, Finals 3PM, Winners 4PM

## Judging Criteria
- Environment Innovation (40%) — novel, creative, challenging?
- Storytelling (30%) — clear explanation, engaging demo?
- Training Script Showing Reward Improvement (20%) — observable training progress?
- Reward & Training Pipeline Setup (10%) — coherent reward logic?

## Prizes
- 1st: $15,000 | 2nd: $9,000 | 3rd: $6,000
- Partner sub-problem prizes: $10,000 each

## Submission
- Submit at: cerebralvalley.ai/e/openenv-hackathon-sf/hackathon/submit
- Upload 1-min YouTube demo video
- Show training script in Colab (Unsloth or HF TRL)

## Data Source
Kaggle NFL Big Data Bowl 2025: `data/` directory
- Downloaded from: https://www.kaggle.com/datasets/marriottgiftmumba/nfl-big-data-bowl-2025
- Key file: `data/plays.csv` (16,124 plays, 50 columns, 136 games, weeks 1-9 of 2022 season)
- Tracking data: `data/tracking_week_1.csv` through `tracking_week_9.csv` (~7.7GB total)
- Derived: `data/defensive_formations.csv` (13 categories, classified from player positions at snap)

## Key Data Columns (plays.csv)
- `isDropback` — True=pass, False=run
- `playAction` — True/False
- `offenseFormation` — 7 categories: SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL, JUMBO, WILDCAT
- `passResult` — C (complete), I (incomplete), S (sack), R (scramble), IN (interception)
- `passLength` — numeric air yards
- `passLocationType` — INSIDE_BOX, OUTSIDE_LEFT, OUTSIDE_RIGHT
- `rushLocationType` — INSIDE_LEFT, INSIDE_RIGHT, OUTSIDE_LEFT, OUTSIDE_RIGHT
- `yardsGained` — the outcome/reward
- `pff_passCoverage` — Cover-1, Cover-2, Cover-3, Quarters, etc.
- `pff_manZone` — Man, Zone, Other
- `pff_runConceptPrimary` — OUTSIDE ZONE, INSIDE ZONE, POWER, COUNTER, etc.
- `down`, `yardsToGo`, `absoluteYardlineNumber`, `quarter`, `gameClock`

## Derived Labels (in notebook)
### Play Types (19 categories, `plays['playType']`)
Short/Mid/Deep Pass × Left/Middle/Right, Run Inside/Outside × Left/Right, Sack, Scramble, QB Spike/Kneel/Sneak
- Short pass: 0-9 air yards, Mid: 10-19, Deep: 20+

### Defensive Formations (13 categories, `data/defensive_formations.csv`)
Nickel (4-2-5), Nickel (3-3-5), Nickel (2-4-5), 3-4, 4-3, Dime (2-3-6), Dime (4-1-6), Dime (3-2-6), 5-2, 5-3 Heavy, 6-2 Goal Line, 4-4, Quarter
- Classified by counting DL/LB/DB roster positions on field at snap

## Action Space (15 actions)
0-2: SHOTGUN run/pass/play_action
3-5: UNDER_CENTER run/pass/play_action
6-8: PISTOL run/pass/play_action
9-10: I_FORM run/pass
11-12: SINGLEBACK run/pass
13: PUNT (4th down only)
14: FIELD_GOAL (4th down, in range)

## Observation Space
down, distance, field_position, quarter, game_clock, score_diff, defense_formation, defenders_in_box, legal_actions, game_phase, done, reward

## Lookup Table Design
- Key: `(down, distance_bucket, field_zone, formation, play_type)`
- Value: list of raw yardsGained values → sample with `random.choice()`
- Distance buckets: short(1-3), medium(4-7), long(8-13), very_long(14+)
- Field zones: own_deep(0-20), own_territory(20-40), midfield(40-60), opp_territory(60-80), red_zone(80-100)
- Sparse fallback: exact → drop field_zone → drop formation → play_type only (min 10 plays)

## Reward
- Raw `yardsGained` per play
- Turnover: -7 (sampled probabilistically per bucket)
- Stretch: drive-end bonuses (+7 TD, +3 FG)

## OpenEnv Framework
- Install: `pip install openenv-core` (or from GitHub for latest)
- Scaffold: `openenv init football_env`
- Models inherit from `openenv.core.env_server.types.Action` and `Observation` (Pydantic)
- Environment extends `openenv.core.env_server.interfaces.Environment`
- Observation base class has `done` and `reward` built in
- Server: `create_app(EnvironmentClass, ActionClass, ObservationClass, env_name=...)` — pass class, NOT instance
- Client: extend `EnvClient`, implement `_step_payload()`, `_parse_result()`, `_parse_state()`
- Deploy: `openenv push --repo-id NAME` to HF Spaces
- CLI: `openenv serve` (local), `openenv build` (Docker), `openenv validate`
- GRPO training: format obs as text prompt, parse LLM output to action, use TRL GRPOTrainer
- Full reference: see `memory/openenv.md`

## Dev Environment
- Python 3.12, venv at `venv/`
- Jupyter kernel registered as "Python (hack)"
- ffmpeg at `/opt/homebrew/bin/ffmpeg` (needed for matplotlib animations)

## Files
- `nfl_tracking.ipynb` — data exploration notebook with visualizations
- `process_nfl_data.ipynb` — lookup table builder (in progress)
- `data/` — NFL dataset CSVs
- `data/defensive_formations.csv` — derived defensive formation labels
- `football_env/` — RL environment (to be built)
