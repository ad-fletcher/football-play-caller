---
title: Football Play-Calling Environment
emoji: 🏈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Football Play-Calling RL Environment

Adversarial NFL play-calling environment where two LLM agents compete: an **offensive coordinator** picks formation + play type, and a **defensive coordinator** chooses formation + coverage. Outcomes are resolved by a two-stage ML model trained on real 2022 NFL tracking data (Big Data Bowl 2025).

Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) 0.2.1.

## Quick Start

```python
from football_drive import FootballDriveClient

async with FootballDriveClient(base_url="https://YOUR-SPACE.hf.space") as client:
    obs = await client.reset()
    print(f"1st & 10 at the {obs.absoluteYardlineNumber}")
```

## Action Space (Adversarial)

**Offense:** formation (7) + play type (5) + pass design or run concept
**Defense:** formation (10) + man/zone + coverage + pass rushers

## Observation

Down, distance, field position, quarter, clock, score differential, play history, per-agent rewards.

## Reward

- **Offense:** yards×0.1 + first down (+1) + TD (+7) / FG (+3) / turnover (-2)
- **Defense:** -yards×0.1 + turnover (+5) + punt (+1) / TD (-7) / FG (-3)
- **Contradiction penalty:** -3 for invalid action combos
