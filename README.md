# Football Play-Calling RL Environment

An [OpenEnv](https://github.com/meta-pytorch/openenv) environment that simulates NFL offensive play-calling. The agent acts as an offensive coordinator, choosing a formation and play type each down. Play outcomes (yards gained, turnovers) are sampled from a lookup table built from real NFL data (16,000+ plays from the 2022 season). Built for the OpenEnv Hackathon, March 2026.

## Run the Environment Locally

```bash
cd football_env
pip install -e .
openenv serve
```

Or with uvicorn directly:

```bash
uvicorn football_env.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Run Training

See the Colab training notebook for GRPO training with TRL/Unsloth.

## Action Space

| ID | Formation | Play Type |
|----|-----------|-----------|
| 0 | SHOTGUN | Run |
| 1 | SHOTGUN | Pass |
| 2 | SHOTGUN | Play Action |
| 3 | UNDER_CENTER | Run |
| 4 | UNDER_CENTER | Pass |
| 5 | UNDER_CENTER | Play Action |
| 6 | PISTOL | Run |
| 7 | PISTOL | Pass |
| 8 | PISTOL | Play Action |
| 9 | I_FORM | Run |
| 10 | I_FORM | Pass |
| 11 | SINGLEBACK | Run |
| 12 | SINGLEBACK | Pass |
| 13 | PUNT | (4th down only) |
| 14 | FIELD_GOAL | (4th down, in range) |
