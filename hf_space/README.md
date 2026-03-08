---
title: Football Play-Calling Environment
emoji: 🏈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

NFL play-calling RL environment. An LLM acts as an offensive coordinator — given a defensive formation, it picks a formation and play type. Outcomes are sampled from real 2022 NFL data (Big Data Bowl 2025).

## API

**Reset** — `POST /reset`
```json
{}
```

**Step** — `POST /step`
```json
{"action": {"formation": "SHOTGUN", "play_type": "pass"}}
```

**Valid formations:** SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL

**Valid play types:** run, pass, play_action
