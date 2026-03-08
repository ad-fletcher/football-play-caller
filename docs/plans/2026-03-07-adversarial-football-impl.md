# Adversarial Football Environment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-play, single-agent football environment with a full-drive, two-agent adversarial environment using the PlayOutcomeModel, and build a custom REINFORCE training script for both agents.

**Architecture:** Single OpenEnv composite environment (`FootballDriveEnv`) where each `step()` takes a `GameAction` containing both `OffenseAction` and `DefenseAction`. The environment simulates a full drive (multiple plays until score/turnover/punt). Training uses custom REINFORCE with batch baseline, alternating updates between offense and defense models.

**Tech Stack:** OpenEnv 0.2.1, Pydantic, PlayOutcomeModel (joblib/sklearn), PyTorch, Unsloth, Transformers

**Key Reference Files:**
- Design doc: `docs/plans/2026-03-07-adversarial-football-env-design.md`
- Existing env: `football_env/server/environment.py`
- Existing models: `football_env/models.py`
- Play outcome model: `football_env/play_outcome_model.py`
- Label encoders: `data/label_encoders.json`
- Model features: `data/model_features.json`
- v1 training: `train_grpo.py`

---

### Task 1: New Pydantic Models

**Files:**
- Modify: `football_env/models.py` (replace entirely)

**Step 1: Write the new models**

Replace `football_env/models.py` with:

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from openenv.core.env_server.types import Action, Observation


class OffenseAction(BaseModel):
    offenseFormation: str = Field(..., description="SHOTGUN|SINGLEBACK|EMPTY|I_FORM|PISTOL|JUMBO|WILDCAT")
    playType: str = Field(..., description="run|pass|play_action|punt|field_goal")
    designedPass: str = Field(default="none", description="Pass route: deep/medium/short/screen x left/middle/right")
    receiverAlignment: str = Field(default="none", description="Receiver alignment: 1x0 through 4x1")
    pff_runConceptPrimary: str = Field(default="none", description="Run concept: INSIDE ZONE, OUTSIDE ZONE, POWER, etc.")


class DefenseAction(BaseModel):
    defFormation: str = Field(..., description="3-4|4-3|Nickel (4-2-5)|Nickel (3-3-5)|etc.")
    pff_manZone: str = Field(..., description="Man|Zone")
    pff_passCoverage: str = Field(..., description="Cover-0|Cover-1|Cover-2|Cover-3|etc.")
    passRushers: int = Field(..., description="Number of pass rushers", ge=1, le=8)


class GameAction(Action):
    offense: OffenseAction
    defense: DefenseAction


class PlayRecord(BaseModel):
    """Record of one play for drive history."""
    offenseFormation: str = ""
    playType: str = ""
    designedPass: str = ""
    pff_runConceptPrimary: str = ""
    defFormation: str = ""
    pff_passCoverage: str = ""
    pff_manZone: str = ""
    passRushers: int = 0
    yardsGained: float = 0.0
    result: str = ""


class GameObservation(Observation):
    # Inherited from Observation: done (bool), reward (Optional[float])
    down: int = Field(default=1)
    yardsToGo: int = Field(default=10)
    absoluteYardlineNumber: int = Field(default=25)
    quarter: int = Field(default=1)
    gameClock_seconds: int = Field(default=900)
    score_diff: int = Field(default=0)
    is_third_down_long: bool = Field(default=False)
    red_zone: bool = Field(default=False)
    last_play_yards: float = Field(default=0.0)
    last_play_result: str = Field(default="")
    offense_history: List[PlayRecord] = Field(default_factory=list)
    defense_history: List[PlayRecord] = Field(default_factory=list)
    offense_reward: float = Field(default=0.0)
    defense_reward: float = Field(default=0.0)
    drive_result: str = Field(default="", description="Empty until drive ends: touchdown|field_goal_made|field_goal_missed|punt|turnover_on_downs|interception|fumble_lost|safety")
```

**Step 2: Verify models import and validate**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "from football_env.models import GameAction, GameObservation, OffenseAction, DefenseAction, PlayRecord; print('Models OK')"`
Expected: `Models OK`

**Step 3: Commit**

```bash
git add football_env/models.py
git commit -m "feat: new Pydantic models for adversarial two-agent environment"
```

---

### Task 2: Action Validation Module

**Files:**
- Create: `football_env/validation.py`

**Step 1: Write the validation module**

This module checks for contradictions in both offense and defense actions and returns penalty amounts.

```python
"""Action validation with contradiction detection for offense and defense."""

# Valid values from data/label_encoders.json
VALID_OFFENSE_FORMATIONS = ["SHOTGUN", "SINGLEBACK", "EMPTY", "I_FORM", "PISTOL", "JUMBO", "WILDCAT"]
VALID_PLAY_TYPES = ["run", "pass", "play_action", "punt", "field_goal"]
VALID_DESIGNED_PASS = [
    "deep_left", "deep_middle", "deep_right",
    "medium_left", "medium_middle", "medium_right",
    "short_left", "short_middle", "short_right",
    "screen_left", "screen_middle", "screen_right",
    "none",
]
VALID_RECEIVER_ALIGNMENT = ["1x0", "1x1", "2x0", "2x1", "2x2", "3x0", "3x1", "3x2", "3x3", "4x1", "unknown"]
VALID_RUN_CONCEPTS = [
    "COUNTER", "DRAW", "FB RUN", "INSIDE ZONE", "MAN", "OUTSIDE ZONE",
    "POWER", "PULL LEAD", "SNEAK", "TRAP", "TRICK", "UNDEFINED", "none",
]

VALID_DEF_FORMATIONS = [
    "3-4", "4-3", "5-2", "Dime (2-3-6)", "Dime (3-2-6)", "Dime (4-1-6)",
    "Nickel (2-4-5)", "Nickel (3-3-5)", "Nickel (4-2-5)", "Other",
]
VALID_MAN_ZONE = ["Man", "Zone"]
VALID_MAN_COVERAGES = ["Cover-0", "Cover-1", "Cover-1 Double", "2-Man"]
VALID_ZONE_COVERAGES = [
    "Cover-2", "Cover-3", "Cover-3 Cloud Left", "Cover-3 Cloud Right",
    "Cover-3 Double Cloud", "Cover-3 Seam", "Cover-6 Right", "Cover 6-Left",
    "Quarters", "Bracket", "Goal Line", "Miscellaneous", "Prevent", "Red Zone",
]
ALL_VALID_COVERAGES = VALID_MAN_COVERAGES + VALID_ZONE_COVERAGES + ["none"]

# Pass rusher ranges by formation (min, max)
RUSHER_RANGES = {
    "3-4": (3, 5),
    "4-3": (4, 6),
    "5-2": (5, 7),
    "Dime (2-3-6)": (2, 4),
    "Dime (3-2-6)": (3, 4),
    "Dime (4-1-6)": (4, 5),
    "Nickel (2-4-5)": (2, 5),
    "Nickel (3-3-5)": (3, 5),
    "Nickel (4-2-5)": (4, 6),
    "Other": (3, 6),
}

CONTRADICTION_PENALTY = -3.0


def validate_offense(offense_action, down: int) -> tuple[float, list[str]]:
    """Validate offensive action. Returns (penalty, list_of_violations)."""
    penalty = 0.0
    violations = []

    # Invalid formation
    if offense_action.offenseFormation not in VALID_OFFENSE_FORMATIONS:
        penalty += CONTRADICTION_PENALTY
        violations.append(f"Invalid formation: {offense_action.offenseFormation}")

    # Invalid play type
    if offense_action.playType not in VALID_PLAY_TYPES:
        penalty += CONTRADICTION_PENALTY
        violations.append(f"Invalid play type: {offense_action.playType}")

    # Punt/FG on 1st-3rd down
    if offense_action.playType in ("punt", "field_goal") and down < 4:
        penalty += CONTRADICTION_PENALTY
        violations.append(f"{offense_action.playType} on down {down} (not 4th down)")

    # Pass details on a run
    if offense_action.playType == "run":
        if offense_action.designedPass != "none":
            penalty += CONTRADICTION_PENALTY
            violations.append(f"designedPass='{offense_action.designedPass}' on a run")
        if offense_action.receiverAlignment != "none":
            penalty += CONTRADICTION_PENALTY
            violations.append(f"receiverAlignment='{offense_action.receiverAlignment}' on a run")

    # Run concept on a pass
    if offense_action.playType in ("pass", "play_action"):
        if offense_action.pff_runConceptPrimary != "none":
            penalty += CONTRADICTION_PENALTY
            violations.append(f"runConcept='{offense_action.pff_runConceptPrimary}' on a pass")

    # Invalid sub-choices
    if offense_action.playType in ("pass", "play_action"):
        if offense_action.designedPass not in VALID_DESIGNED_PASS:
            penalty += CONTRADICTION_PENALTY
            violations.append(f"Invalid designedPass: {offense_action.designedPass}")
        if offense_action.receiverAlignment not in VALID_RECEIVER_ALIGNMENT:
            penalty += CONTRADICTION_PENALTY
            violations.append(f"Invalid receiverAlignment: {offense_action.receiverAlignment}")

    if offense_action.playType == "run":
        if offense_action.pff_runConceptPrimary not in VALID_RUN_CONCEPTS:
            penalty += CONTRADICTION_PENALTY
            violations.append(f"Invalid runConcept: {offense_action.pff_runConceptPrimary}")

    return penalty, violations


def validate_defense(defense_action) -> tuple[float, list[str]]:
    """Validate defensive action. Returns (penalty, list_of_violations)."""
    penalty = 0.0
    violations = []

    # Invalid formation
    if defense_action.defFormation not in VALID_DEF_FORMATIONS:
        penalty += CONTRADICTION_PENALTY
        violations.append(f"Invalid def formation: {defense_action.defFormation}")

    # Invalid man/zone
    if defense_action.pff_manZone not in VALID_MAN_ZONE:
        penalty += CONTRADICTION_PENALTY
        violations.append(f"Invalid manZone: {defense_action.pff_manZone}")

    # Coverage doesn't match man/zone
    if defense_action.pff_manZone == "Man":
        if defense_action.pff_passCoverage not in VALID_MAN_COVERAGES:
            penalty += CONTRADICTION_PENALTY
            violations.append(f"Coverage '{defense_action.pff_passCoverage}' is not a Man coverage")
    elif defense_action.pff_manZone == "Zone":
        if defense_action.pff_passCoverage not in VALID_ZONE_COVERAGES:
            penalty += CONTRADICTION_PENALTY
            violations.append(f"Coverage '{defense_action.pff_passCoverage}' is not a Zone coverage")

    # Pass rushers out of range for formation
    if defense_action.defFormation in RUSHER_RANGES:
        lo, hi = RUSHER_RANGES[defense_action.defFormation]
        if not (lo <= defense_action.passRushers <= hi):
            penalty += CONTRADICTION_PENALTY
            violations.append(f"{defense_action.passRushers} rushers invalid for {defense_action.defFormation} (valid: {lo}-{hi})")

    return penalty, violations
```

**Step 2: Verify validation imports**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "from football_env.validation import validate_offense, validate_defense; print('Validation OK')"`
Expected: `Validation OK`

**Step 3: Write tests for validation**

Create `tests/test_validation.py`:

```python
"""Tests for action validation logic."""
import pytest
from football_env.models import OffenseAction, DefenseAction
from football_env.validation import validate_offense, validate_defense, CONTRADICTION_PENALTY


class TestOffenseValidation:
    def test_valid_pass_play(self):
        action = OffenseAction(
            offenseFormation="SHOTGUN", playType="pass",
            designedPass="short_middle", receiverAlignment="3x1",
        )
        penalty, violations = validate_offense(action, down=1)
        assert penalty == 0.0
        assert violations == []

    def test_valid_run_play(self):
        action = OffenseAction(
            offenseFormation="I_FORM", playType="run",
            pff_runConceptPrimary="POWER",
        )
        penalty, violations = validate_offense(action, down=2)
        assert penalty == 0.0
        assert violations == []

    def test_punt_on_4th_down_valid(self):
        action = OffenseAction(offenseFormation="SHOTGUN", playType="punt")
        penalty, violations = validate_offense(action, down=4)
        assert penalty == 0.0

    def test_punt_on_2nd_down_penalty(self):
        action = OffenseAction(offenseFormation="SHOTGUN", playType="punt")
        penalty, violations = validate_offense(action, down=2)
        assert penalty == CONTRADICTION_PENALTY
        assert len(violations) == 1

    def test_pass_details_on_run_penalty(self):
        action = OffenseAction(
            offenseFormation="SHOTGUN", playType="run",
            designedPass="deep_left", receiverAlignment="3x1",
            pff_runConceptPrimary="INSIDE ZONE",
        )
        penalty, violations = validate_offense(action, down=1)
        # Two violations: designedPass on run + receiverAlignment on run
        assert penalty == CONTRADICTION_PENALTY * 2

    def test_run_concept_on_pass_penalty(self):
        action = OffenseAction(
            offenseFormation="SHOTGUN", playType="pass",
            designedPass="short_middle", receiverAlignment="3x1",
            pff_runConceptPrimary="INSIDE ZONE",
        )
        penalty, violations = validate_offense(action, down=1)
        assert penalty == CONTRADICTION_PENALTY

    def test_invalid_formation(self):
        action = OffenseAction(offenseFormation="INVALID", playType="run", pff_runConceptPrimary="POWER")
        penalty, violations = validate_offense(action, down=1)
        assert penalty == CONTRADICTION_PENALTY


class TestDefenseValidation:
    def test_valid_man_coverage(self):
        action = DefenseAction(
            defFormation="Nickel (4-2-5)", pff_manZone="Man",
            pff_passCoverage="Cover-1", passRushers=4,
        )
        penalty, violations = validate_defense(action)
        assert penalty == 0.0
        assert violations == []

    def test_valid_zone_coverage(self):
        action = DefenseAction(
            defFormation="4-3", pff_manZone="Zone",
            pff_passCoverage="Cover-3", passRushers=4,
        )
        penalty, violations = validate_defense(action)
        assert penalty == 0.0

    def test_man_coverage_with_zone_scheme_penalty(self):
        action = DefenseAction(
            defFormation="4-3", pff_manZone="Man",
            pff_passCoverage="Cover-3", passRushers=4,
        )
        penalty, violations = validate_defense(action)
        assert penalty == CONTRADICTION_PENALTY

    def test_rushers_out_of_range(self):
        action = DefenseAction(
            defFormation="Dime (2-3-6)", pff_manZone="Zone",
            pff_passCoverage="Cover-2", passRushers=7,
        )
        penalty, violations = validate_defense(action)
        assert penalty == CONTRADICTION_PENALTY

    def test_invalid_formation(self):
        action = DefenseAction(
            defFormation="INVALID", pff_manZone="Zone",
            pff_passCoverage="Cover-3", passRushers=4,
        )
        penalty, violations = validate_defense(action)
        assert penalty == CONTRADICTION_PENALTY
```

**Step 4: Run tests**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -m pytest tests/test_validation.py -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add football_env/validation.py tests/test_validation.py
git commit -m "feat: action validation with contradiction detection for offense and defense"
```

---

### Task 3: Drive Environment Core

**Files:**
- Modify: `football_env/server/environment.py` (replace entirely)

**Step 1: Write the new environment**

Replace `football_env/server/environment.py` with the full drive simulation. This is the largest single file.

```python
"""Adversarial football drive environment.

Single drive per episode. Two agents (offense + defense) submit actions each play.
PlayOutcomeModel resolves outcomes. Drive ends on score, turnover, or punt.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from ..models import GameAction, GameObservation, PlayRecord
from ..validation import validate_offense, validate_defense
from ..play_outcome_model import PlayOutcomeModel


# Field goal make probability by distance (yards from goal line to kick spot is ~17 + distance)
FG_PROBABILITY = [
    (30, 0.95),   # < 30 yards
    (40, 0.85),   # 30-39
    (50, 0.75),   # 40-49
    (55, 0.60),   # 50-54
    (100, 0.30),  # 55+
]

# Realistic drive start positions (yards from own end zone)
DRIVE_START_POSITIONS = list(range(15, 40))  # 15-39, weighted toward 20-30
DRIVE_START_WEIGHTS = [1 if y < 20 or y > 35 else 3 for y in DRIVE_START_POSITIONS]

SCORE_DIFF_RANGE = range(-21, 22)  # -21 to +21


class FootballDriveEnvironment(Environment):
    def __init__(self):
        self._model = PlayOutcomeModel()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._game_state = {}
        self._offense_history = []
        self._defense_history = []

    def reset(self, seed=None, **kwargs) -> GameObservation:
        if seed is not None:
            random.seed(seed)

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._offense_history = []
        self._defense_history = []

        # Randomized start
        field_pos = random.choices(DRIVE_START_POSITIONS, weights=DRIVE_START_WEIGHTS, k=1)[0]
        quarter = random.randint(1, 4)
        clock = random.randint(60, 900)
        score_diff = random.choice(SCORE_DIFF_RANGE)

        self._game_state = {
            "down": 1,
            "yardsToGo": 10,
            "absoluteYardlineNumber": field_pos,
            "quarter": quarter,
            "gameClock_seconds": clock,
            "score_diff": score_diff,
            "first_down_marker": field_pos + 10,  # yard line needed for first down
        }

        return self._build_observation(done=False, offense_reward=0.0, defense_reward=0.0)

    def step(self, action: GameAction, **kwargs) -> GameObservation:
        self._state.step_count += 1
        gs = self._game_state

        # Validate actions
        off_penalty, off_violations = validate_offense(action.offense, gs["down"])
        def_penalty, def_violations = validate_defense(action.defense)

        offense_reward = off_penalty
        defense_reward = def_penalty

        # Handle punt
        if action.offense.playType == "punt" and gs["down"] == 4:
            self._append_history(action, 0.0, "punt")
            return self._build_observation(
                done=True, offense_reward=offense_reward + 0.0,
                defense_reward=defense_reward + 1.0,
                last_play_yards=0.0, last_play_result="punt",
                drive_result="punt",
            )

        # Handle field goal attempt
        if action.offense.playType == "field_goal" and gs["down"] == 4:
            fg_distance = 100 - gs["absoluteYardlineNumber"] + 17  # snap + hold
            made = self._attempt_field_goal(fg_distance)
            result = "field_goal_made" if made else "field_goal_missed"
            off_r = 3.0 if made else 0.0
            def_r = -3.0 if made else 3.0
            self._append_history(action, 0.0, result)
            return self._build_observation(
                done=True, offense_reward=offense_reward + off_r,
                defense_reward=defense_reward + def_r,
                last_play_yards=0.0, last_play_result=result,
                drive_result=result,
            )

        # If there are violations but it's not a special play, still run the play
        # (agent gets penalized but play resolves with defaults for invalid fields)

        # Derive isDropback
        is_dropback = action.offense.playType in ("pass", "play_action")

        # Call PlayOutcomeModel
        outcome, yards = self._model.predict(
            quarter=gs["quarter"],
            down=gs["down"],
            yardsToGo=gs["yardsToGo"],
            gameClock_seconds=gs["gameClock_seconds"],
            absoluteYardlineNumber=gs["absoluteYardlineNumber"],
            isDropback=is_dropback,
            passRushers=action.defense.passRushers,
            score_diff=gs["score_diff"],
            offenseFormation=action.offense.offenseFormation,
            playType=action.offense.playType,
            defFormation=action.defense.defFormation,
            pff_passCoverage=action.defense.pff_passCoverage,
            pff_manZone=action.defense.pff_manZone,
            designedPass=action.offense.designedPass,
            receiverAlignment=action.offense.receiverAlignment,
            pff_runConceptPrimary=action.offense.pff_runConceptPrimary,
        )

        yards = round(yards, 1)

        # Check for turnovers
        if outcome == "interception":
            self._append_history(action, yards, "interception")
            return self._build_observation(
                done=True, offense_reward=offense_reward + -2.0,
                defense_reward=defense_reward + 5.0,
                last_play_yards=yards, last_play_result="interception",
                drive_result="interception",
            )

        if outcome == "fumble_lost":
            self._append_history(action, yards, "fumble_lost")
            return self._build_observation(
                done=True, offense_reward=offense_reward + -2.0,
                defense_reward=defense_reward + 5.0,
                last_play_yards=yards, last_play_result="fumble_lost",
                drive_result="fumble_lost",
            )

        # Normal play or touchdown outcome from model
        new_field_pos = gs["absoluteYardlineNumber"] + yards

        # Safety check (tackled behind own goal line)
        if new_field_pos <= 0:
            self._append_history(action, yards, "safety")
            return self._build_observation(
                done=True, offense_reward=offense_reward + -2.0,
                defense_reward=defense_reward + 2.0,
                last_play_yards=yards, last_play_result="safety",
                drive_result="safety",
            )

        # Touchdown check
        if new_field_pos >= 100 or outcome == "touchdown":
            self._append_history(action, yards, "touchdown")
            return self._build_observation(
                done=True, offense_reward=offense_reward + 7.0,
                defense_reward=defense_reward + -7.0,
                last_play_yards=yards, last_play_result="touchdown",
                drive_result="touchdown",
            )

        # Per-play rewards (yards gained)
        offense_reward += yards * 0.1
        defense_reward += -yards * 0.1

        # Update game state
        gs["absoluteYardlineNumber"] = int(round(new_field_pos))

        # First down check
        if new_field_pos >= gs["first_down_marker"]:
            offense_reward += 1.0  # First down bonus
            gs["down"] = 1
            gs["yardsToGo"] = 10
            gs["first_down_marker"] = min(gs["absoluteYardlineNumber"] + 10, 100)
        else:
            gs["down"] += 1
            gs["yardsToGo"] = int(round(gs["first_down_marker"] - new_field_pos))

        # Turnover on downs
        if gs["down"] > 4:
            self._append_history(action, yards, "turnover_on_downs")
            return self._build_observation(
                done=True, offense_reward=offense_reward + -2.0,
                defense_reward=defense_reward + 4.0,
                last_play_yards=yards, last_play_result="turnover_on_downs",
                drive_result="turnover_on_downs",
            )

        # Clock
        gs["gameClock_seconds"] = max(0, gs["gameClock_seconds"] - 40)

        self._append_history(action, yards, "normal")

        return self._build_observation(
            done=False, offense_reward=offense_reward, defense_reward=defense_reward,
            last_play_yards=yards, last_play_result="normal",
        )

    def _build_observation(
        self, done: bool, offense_reward: float, defense_reward: float,
        last_play_yards: float = 0.0, last_play_result: str = "",
        drive_result: str = "",
    ) -> GameObservation:
        gs = self._game_state
        return GameObservation(
            done=done,
            reward=offense_reward,  # OpenEnv base uses this; we also expose per-agent
            down=gs["down"],
            yardsToGo=gs["yardsToGo"],
            absoluteYardlineNumber=gs["absoluteYardlineNumber"],
            quarter=gs["quarter"],
            gameClock_seconds=gs["gameClock_seconds"],
            score_diff=gs["score_diff"],
            is_third_down_long=(gs["down"] == 3 and gs["yardsToGo"] >= 7),
            red_zone=(gs["absoluteYardlineNumber"] >= 80),
            last_play_yards=last_play_yards,
            last_play_result=last_play_result,
            offense_history=list(self._offense_history),
            defense_history=list(self._defense_history),
            offense_reward=offense_reward,
            defense_reward=defense_reward,
            drive_result=drive_result,
        )

    def _append_history(self, action: GameAction, yards: float, result: str):
        self._offense_history.append(PlayRecord(
            offenseFormation=action.offense.offenseFormation,
            playType=action.offense.playType,
            designedPass=action.offense.designedPass,
            pff_runConceptPrimary=action.offense.pff_runConceptPrimary,
            yardsGained=yards,
            result=result,
        ))
        self._defense_history.append(PlayRecord(
            defFormation=action.defense.defFormation,
            pff_passCoverage=action.defense.pff_passCoverage,
            pff_manZone=action.defense.pff_manZone,
            passRushers=action.defense.passRushers,
            yardsGained=yards,
            result=result,
        ))

    def _attempt_field_goal(self, distance: int) -> bool:
        for threshold, prob in FG_PROBABILITY:
            if distance < threshold:
                return random.random() < prob
        return random.random() < 0.30

    @property
    def state(self) -> State:
        return self._state
```

**Step 2: Verify environment imports**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "from football_env.server.environment import FootballDriveEnvironment; print('Environment OK')"`
Expected: `Environment OK`

**Step 3: Commit**

```bash
git add football_env/server/environment.py
git commit -m "feat: full-drive adversarial environment with PlayOutcomeModel"
```

---

### Task 4: Update Server App and Client

**Files:**
- Modify: `football_env/server/app.py`
- Modify: `football_env/client.py`
- Modify: `football_env/__init__.py`

**Step 1: Update app.py**

```python
import uvicorn
from openenv.core.env_server import create_app
from .environment import FootballDriveEnvironment
from ..models import GameAction, GameObservation

app = create_app(FootballDriveEnvironment, GameAction, GameObservation, env_name="football_drive")


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

**Step 2: Update client.py**

```python
from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import GameAction, GameObservation, OffenseAction, DefenseAction


class FootballDriveClient(EnvClient[GameAction, GameObservation, State]):
    def _step_payload(self, action: GameAction) -> Dict[str, Any]:
        return {
            "offense": action.offense.model_dump(),
            "defense": action.defense.model_dump(),
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[GameObservation]:
        obs_data = payload.get("observation", payload)
        obs = GameObservation(**obs_data)
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
```

**Step 3: Update __init__.py**

```python
from .models import GameAction, GameObservation, OffenseAction, DefenseAction, PlayRecord
from .client import FootballDriveClient
```

**Step 4: Verify imports**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "from football_env import GameAction, GameObservation, FootballDriveClient; print('Package OK')"`
Expected: `Package OK`

**Step 5: Commit**

```bash
git add football_env/server/app.py football_env/client.py football_env/__init__.py
git commit -m "feat: update server app and client for adversarial environment"
```

---

### Task 5: Environment Integration Test

**Files:**
- Create: `tests/test_environment.py`

**Step 1: Write integration tests**

```python
"""Integration tests for the adversarial football drive environment."""
import pytest
from football_env.models import GameAction, OffenseAction, DefenseAction
from football_env.server.environment import FootballDriveEnvironment


@pytest.fixture
def env():
    return FootballDriveEnvironment()


def make_pass_action():
    return GameAction(
        offense=OffenseAction(
            offenseFormation="SHOTGUN", playType="pass",
            designedPass="short_middle", receiverAlignment="3x1",
        ),
        defense=DefenseAction(
            defFormation="Nickel (4-2-5)", pff_manZone="Zone",
            pff_passCoverage="Cover-3", passRushers=4,
        ),
    )


def make_run_action():
    return GameAction(
        offense=OffenseAction(
            offenseFormation="I_FORM", playType="run",
            pff_runConceptPrimary="POWER",
        ),
        defense=DefenseAction(
            defFormation="4-3", pff_manZone="Man",
            pff_passCoverage="Cover-1", passRushers=4,
        ),
    )


def make_punt_action():
    return GameAction(
        offense=OffenseAction(offenseFormation="SHOTGUN", playType="punt"),
        defense=DefenseAction(
            defFormation="4-3", pff_manZone="Zone",
            pff_passCoverage="Cover-3", passRushers=4,
        ),
    )


def make_fg_action():
    return GameAction(
        offense=OffenseAction(offenseFormation="SHOTGUN", playType="field_goal"),
        defense=DefenseAction(
            defFormation="4-3", pff_manZone="Zone",
            pff_passCoverage="Cover-3", passRushers=4,
        ),
    )


class TestReset:
    def test_reset_returns_valid_observation(self, env):
        obs = env.reset(seed=42)
        assert obs.done is False
        assert obs.down == 1
        assert obs.yardsToGo == 10
        assert 15 <= obs.absoluteYardlineNumber <= 39
        assert 1 <= obs.quarter <= 4
        assert obs.offense_history == []
        assert obs.defense_history == []

    def test_reset_with_seed_is_deterministic(self, env):
        obs1 = env.reset(seed=123)
        obs2 = env.reset(seed=123)
        assert obs1.absoluteYardlineNumber == obs2.absoluteYardlineNumber
        assert obs1.quarter == obs2.quarter
        assert obs1.score_diff == obs2.score_diff


class TestStep:
    def test_step_returns_observation(self, env):
        env.reset(seed=42)
        obs = env.step(make_pass_action())
        assert isinstance(obs.offense_reward, float)
        assert isinstance(obs.defense_reward, float)
        assert obs.last_play_result != ""

    def test_history_accumulates(self, env):
        env.reset(seed=42)
        obs = env.step(make_pass_action())
        if not obs.done:
            assert len(obs.offense_history) == 1
            assert len(obs.defense_history) == 1
            obs2 = env.step(make_run_action())
            if not obs2.done:
                assert len(obs2.offense_history) == 2


class TestDriveEnd:
    def test_punt_ends_drive_on_4th_down(self, env):
        env.reset(seed=42)
        # Advance to 4th down
        env._game_state["down"] = 4
        obs = env.step(make_punt_action())
        assert obs.done is True
        assert obs.drive_result == "punt"
        assert obs.defense_reward >= 1.0  # defense gets +1

    def test_punt_penalized_on_1st_down(self, env):
        env.reset(seed=42)
        # Should get contradiction penalty but still process as punt on non-4th
        obs = env.step(make_punt_action())
        # On 1st down, punt gets contradiction penalty
        assert obs.offense_reward < 0

    def test_field_goal_ends_drive_on_4th_down(self, env):
        env.reset(seed=42)
        env._game_state["down"] = 4
        env._game_state["absoluteYardlineNumber"] = 80  # close range
        obs = env.step(make_fg_action())
        assert obs.done is True
        assert obs.drive_result in ("field_goal_made", "field_goal_missed")


class TestFullDrive:
    def test_drive_eventually_ends(self, env):
        """Run plays until drive ends (should always terminate)."""
        env.reset(seed=42)
        obs = env.reset(seed=42)
        max_plays = 50
        for i in range(max_plays):
            if obs.done:
                break
            action = make_pass_action()
            # On 4th down, punt to avoid infinite loop
            if obs.down == 4:
                action = make_punt_action()
            obs = env.step(action)
        assert obs.done is True, f"Drive did not end after {max_plays} plays"

    def test_drive_collects_rewards(self, env):
        """Verify rewards are returned each step."""
        obs = env.reset(seed=42)
        total_off = 0.0
        total_def = 0.0
        for _ in range(50):
            if obs.done:
                break
            action = make_pass_action() if obs.down < 4 else make_punt_action()
            obs = env.step(action)
            total_off += obs.offense_reward
            total_def += obs.defense_reward
        # At least some non-zero rewards should exist
        assert total_off != 0.0 or total_def != 0.0


class TestContradictions:
    def test_offense_contradiction_penalty(self, env):
        env.reset(seed=42)
        # Run play with pass details = contradiction
        bad_action = GameAction(
            offense=OffenseAction(
                offenseFormation="SHOTGUN", playType="run",
                designedPass="deep_left",  # contradiction!
                pff_runConceptPrimary="INSIDE ZONE",
            ),
            defense=DefenseAction(
                defFormation="4-3", pff_manZone="Zone",
                pff_passCoverage="Cover-3", passRushers=4,
            ),
        )
        obs = env.step(bad_action)
        assert obs.offense_reward < 0  # should have penalty

    def test_defense_contradiction_penalty(self, env):
        env.reset(seed=42)
        bad_action = GameAction(
            offense=OffenseAction(
                offenseFormation="SHOTGUN", playType="pass",
                designedPass="short_middle", receiverAlignment="3x1",
            ),
            defense=DefenseAction(
                defFormation="4-3", pff_manZone="Man",
                pff_passCoverage="Cover-3",  # Cover-3 is zone, not man!
                passRushers=4,
            ),
        )
        obs = env.step(bad_action)
        assert obs.defense_reward < 0  # should have penalty
```

**Step 2: Run tests**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -m pytest tests/test_environment.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/test_environment.py
git commit -m "test: integration tests for adversarial drive environment"
```

---

### Task 6: Prompt Formatting for LLM Agents

**Files:**
- Create: `football_env/prompts.py`

**Step 1: Write prompt formatters**

These format observations into text prompts for the offense and defense LLMs, and parse their JSON responses back into actions.

```python
"""Prompt formatting and response parsing for offense and defense LLMs."""
import json
from .models import GameObservation, OffenseAction, DefenseAction
from .validation import (
    VALID_OFFENSE_FORMATIONS, VALID_PLAY_TYPES,
    VALID_DESIGNED_PASS, VALID_RECEIVER_ALIGNMENT, VALID_RUN_CONCEPTS,
    VALID_DEF_FORMATIONS, VALID_MAN_ZONE, VALID_MAN_COVERAGES, VALID_ZONE_COVERAGES,
    RUSHER_RANGES,
)


OFFENSE_SYSTEM_PROMPT = """\
You are an NFL offensive coordinator. Given the game situation and the defense's tendencies, call a play.

Respond with ONLY a JSON object. Your formation and play type determine what other fields are required:

For pass or play_action:
{"offenseFormation": "<FORM>", "playType": "pass", "designedPass": "<ROUTE>", "receiverAlignment": "<ALIGN>"}

For run:
{"offenseFormation": "<FORM>", "playType": "run", "pff_runConceptPrimary": "<CONCEPT>"}

For punt (4th down only):
{"offenseFormation": "<FORM>", "playType": "punt"}

For field_goal (4th down only):
{"offenseFormation": "<FORM>", "playType": "field_goal"}

Valid formations: """ + ", ".join(VALID_OFFENSE_FORMATIONS) + """
Valid pass routes: """ + ", ".join([d for d in VALID_DESIGNED_PASS if d != "none"]) + """
Valid alignments: """ + ", ".join([r for r in VALID_RECEIVER_ALIGNMENT if r != "unknown"]) + """
Valid run concepts: """ + ", ".join([r for r in VALID_RUN_CONCEPTS if r not in ("none", "UNDEFINED")])


DEFENSE_SYSTEM_PROMPT = """\
You are an NFL defensive coordinator. Given the game situation and the offense's tendencies, call a defense.

Respond with ONLY a JSON object:
{"defFormation": "<FORM>", "pff_manZone": "Man|Zone", "pff_passCoverage": "<COVERAGE>", "passRushers": <N>}

Valid formations: """ + ", ".join(VALID_DEF_FORMATIONS) + """
Man coverages: """ + ", ".join(VALID_MAN_COVERAGES) + """
Zone coverages: """ + ", ".join(VALID_ZONE_COVERAGES) + """
Pass rushers must be in valid range for your formation."""


def format_offense_obs(obs: GameObservation) -> list[dict]:
    """Format observation as chat messages for the offense LLM."""
    situation = (
        f"{_down_str(obs.down)} & {obs.yardsToGo} at the {_field_pos_str(obs.absoluteYardlineNumber)}. "
        f"Q{obs.quarter}, {_clock_str(obs.gameClock_seconds)}. "
        f"{'Trailing' if obs.score_diff < 0 else 'Leading' if obs.score_diff > 0 else 'Tied'}"
        f"{' by ' + str(abs(obs.score_diff)) if obs.score_diff != 0 else ''}."
    )

    if obs.defense_history:
        tendencies = _summarize_defense_history(obs.defense_history)
        situation += f"\n\nDefense tendencies this drive:\n{tendencies}"

    if obs.last_play_result:
        situation += f"\n\nLast play: {obs.last_play_result} for {obs.last_play_yards} yards."

    return [
        {"role": "system", "content": OFFENSE_SYSTEM_PROMPT},
        {"role": "user", "content": situation},
    ]


def format_defense_obs(obs: GameObservation) -> list[dict]:
    """Format observation as chat messages for the defense LLM."""
    situation = (
        f"Offense has {_down_str(obs.down)} & {obs.yardsToGo} at the {_field_pos_str(obs.absoluteYardlineNumber)}. "
        f"Q{obs.quarter}, {_clock_str(obs.gameClock_seconds)}. "
        f"{'They trail' if obs.score_diff < 0 else 'They lead' if obs.score_diff > 0 else 'Tied'}"
        f"{' by ' + str(abs(obs.score_diff)) if obs.score_diff != 0 else ''}."
    )

    if obs.offense_history:
        tendencies = _summarize_offense_history(obs.offense_history)
        situation += f"\n\nOffense tendencies this drive:\n{tendencies}"

    if obs.last_play_result:
        situation += f"\n\nLast play: {obs.last_play_result} for {obs.last_play_yards} yards."

    return [
        {"role": "system", "content": DEFENSE_SYSTEM_PROMPT},
        {"role": "user", "content": situation},
    ]


def parse_offense_response(text: str) -> OffenseAction:
    """Parse LLM text output into OffenseAction. Returns defaults for unparseable input."""
    parsed = _extract_json(text)
    if parsed is None:
        return OffenseAction(offenseFormation="SHOTGUN", playType="pass", designedPass="short_middle", receiverAlignment="3x1")
    return OffenseAction(
        offenseFormation=parsed.get("offenseFormation", "SHOTGUN"),
        playType=parsed.get("playType", "pass"),
        designedPass=parsed.get("designedPass", "none"),
        receiverAlignment=parsed.get("receiverAlignment", "none"),
        pff_runConceptPrimary=parsed.get("pff_runConceptPrimary", "none"),
    )


def parse_defense_response(text: str) -> DefenseAction:
    """Parse LLM text output into DefenseAction. Returns defaults for unparseable input."""
    parsed = _extract_json(text)
    if parsed is None:
        return DefenseAction(defFormation="4-3", pff_manZone="Zone", pff_passCoverage="Cover-3", passRushers=4)
    return DefenseAction(
        defFormation=parsed.get("defFormation", "4-3"),
        pff_manZone=parsed.get("pff_manZone", "Zone"),
        pff_passCoverage=parsed.get("pff_passCoverage", "Cover-3"),
        passRushers=int(parsed.get("passRushers", 4)),
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError):
        return None


def _down_str(down: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(down, f"{down}th")


def _field_pos_str(yard_line: int) -> str:
    if yard_line <= 50:
        return f"own {yard_line}"
    return f"opponent {100 - yard_line}"


def _clock_str(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def _summarize_defense_history(history) -> str:
    lines = []
    for i, play in enumerate(history, 1):
        lines.append(f"  Play {i}: {play.defFormation}, {play.pff_passCoverage} ({play.pff_manZone}), {play.passRushers} rushers → {play.result} ({play.yardsGained} yds)")
    return "\n".join(lines)


def _summarize_offense_history(history) -> str:
    lines = []
    for i, play in enumerate(history, 1):
        detail = play.designedPass if play.playType != "run" else play.pff_runConceptPrimary
        lines.append(f"  Play {i}: {play.offenseFormation} {play.playType} ({detail}) → {play.result} ({play.yardsGained} yds)")
    return "\n".join(lines)
```

**Step 2: Verify import**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "from football_env.prompts import format_offense_obs, format_defense_obs, parse_offense_response, parse_defense_response; print('Prompts OK')"`
Expected: `Prompts OK`

**Step 3: Commit**

```bash
git add football_env/prompts.py
git commit -m "feat: prompt formatting and response parsing for offense/defense LLMs"
```

---

### Task 7: Training Script — Episode Collection

**Files:**
- Create: `train_adversarial.py`

**Step 1: Write the training script**

This is the full training script with episode collection and REINFORCE updates. It replaces `train_grpo.py` conceptually.

```python
"""
Adversarial REINFORCE Training: Offense vs Defense
===================================================
Two Qwen2.5-1.5B-Instruct models (LoRA) trained adversarially.
Offense learns to score, defense learns to stop them.

Custom REINFORCE with batch baseline (not GRPOTrainer).

Install:
  pip install unsloth torch transformers peft datasets matplotlib
"""

import json
import os
import pathlib
import random
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_RANK = 32
OUTPUT_DIR = "checkpoints/adversarial"

# Training
EPISODES_PER_ROUND = 64       # drives per collection round
NUM_ROUNDS = 100               # total training rounds
LEARNING_RATE = 1e-5
MAX_COMPLETION_LENGTH = 150
TEMPERATURE = 0.7

# Unsloth
LOAD_4BIT = True
FAST_INFERENCE = True
GPU_MEMORY_UTIL = 0.4

SAVE_EVERY = 10  # save every N rounds


# ──────────────────────────────────────────────
# Episode Collection
# ──────────────────────────────────────────────
def collect_episodes(env, off_model, off_tokenizer, def_model, def_tokenizer, n_episodes, device):
    """Roll out full drives, collect (prompt_ids, response_ids, reward) per agent."""
    from football_env.models import GameAction
    from football_env.prompts import (
        format_offense_obs, format_defense_obs,
        parse_offense_response, parse_defense_response,
    )

    offense_episodes = []
    defense_episodes = []

    for ep in range(n_episodes):
        obs = env.reset()
        drive_off = []
        drive_def = []

        while not obs.done:
            # Format prompts
            off_messages = format_offense_obs(obs)
            def_messages = format_defense_obs(obs)

            # Generate offense action
            off_text = off_tokenizer.apply_chat_template(off_messages, tokenize=False, add_generation_prompt=True)
            off_ids = off_tokenizer(off_text, return_tensors="pt").to(device)
            with torch.no_grad():
                off_out = off_model.generate(
                    **off_ids, max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=TEMPERATURE, do_sample=True,
                )
            off_response_ids = off_out[0][off_ids["input_ids"].shape[1]:]
            off_response_text = off_tokenizer.decode(off_response_ids, skip_special_tokens=True)

            # Generate defense action
            def_text = def_tokenizer.apply_chat_template(def_messages, tokenize=False, add_generation_prompt=True)
            def_ids = def_tokenizer(def_text, return_tensors="pt").to(device)
            with torch.no_grad():
                def_out = def_model.generate(
                    **def_ids, max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=TEMPERATURE, do_sample=True,
                )
            def_response_ids = def_out[0][def_ids["input_ids"].shape[1]:]
            def_response_text = def_tokenizer.decode(def_response_ids, skip_special_tokens=True)

            # Parse actions
            off_action = parse_offense_response(off_response_text)
            def_action = parse_defense_response(def_response_text)

            # Step environment
            composite = GameAction(offense=off_action, defense=def_action)
            obs = env.step(composite)

            # Store (prompt_text, response_text, reward)
            drive_off.append((off_text, off_response_text, obs.offense_reward))
            drive_def.append((def_text, def_response_text, obs.defense_reward))

        offense_episodes.extend(drive_off)
        defense_episodes.extend(drive_def)

    return offense_episodes, defense_episodes


# ──────────────────────────────────────────────
# REINFORCE Update
# ──────────────────────────────────────────────
def get_log_prob(model, tokenizer, prompt_text, response_text, device):
    """Compute log probability of response given prompt."""
    full_text = prompt_text + response_text
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = prompt_ids["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**full_ids)

    # Get logits for response tokens only
    logits = outputs.logits[0, prompt_len - 1:-1, :]  # shifted by 1
    response_token_ids = full_ids["input_ids"][0, prompt_len:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
    return token_log_probs.sum()


def reinforce_update(model, tokenizer, optimizer, episodes, device):
    """REINFORCE with batch mean baseline."""
    if not episodes:
        return 0.0

    prompts, responses, rewards = zip(*episodes)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    model.train()
    total_loss = 0.0

    for prompt, response, advantage in zip(prompts, responses, advantages):
        # Re-enable gradients for log prob computation
        full_text = prompt + response
        full_ids = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        outputs = model(**full_ids)
        logits = outputs.logits[0, prompt_len - 1:-1, :]
        response_token_ids = full_ids["input_ids"][0, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        log_prob = token_log_probs.sum()

        loss = -(advantage * log_prob)
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / len(episodes)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
@dataclass
class TrainingLog:
    rounds: list = field(default_factory=list)
    offense_rewards: list = field(default_factory=list)
    defense_rewards: list = field(default_factory=list)
    offense_losses: list = field(default_factory=list)
    defense_losses: list = field(default_factory=list)
    drive_results: list = field(default_factory=list)

    def log_round(self, round_num, off_episodes, def_episodes, off_loss, def_loss, drive_results):
        self.rounds.append(round_num)
        off_rewards = [r for _, _, r in off_episodes]
        def_rewards = [r for _, _, r in def_episodes]
        self.offense_rewards.append(float(np.mean(off_rewards)))
        self.defense_rewards.append(float(np.mean(def_rewards)))
        self.offense_losses.append(off_loss)
        self.defense_losses.append(def_loss)
        self.drive_results.append(drive_results)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump({
                "rounds": self.rounds,
                "offense_rewards": self.offense_rewards,
                "defense_rewards": self.defense_rewards,
                "offense_losses": self.offense_losses,
                "defense_losses": self.defense_losses,
                "drive_results": self.drive_results,
            }, f, indent=2)

        # Plot reward curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(self.rounds, self.offense_rewards, "b-", alpha=0.5, label="Offense")
        ax.plot(self.rounds, self.defense_rewards, "r-", alpha=0.5, label="Defense")
        w = min(10, len(self.rounds))
        if w > 1:
            off_ma = np.convolve(self.offense_rewards, np.ones(w)/w, mode="valid")
            def_ma = np.convolve(self.defense_rewards, np.ones(w)/w, mode="valid")
            ax.plot(self.rounds[w-1:], off_ma, "b-", lw=2, label=f"Off MA-{w}")
            ax.plot(self.rounds[w-1:], def_ma, "r-", lw=2, label=f"Def MA-{w}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Adversarial Training: Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self.rounds, self.offense_losses, "b-", alpha=0.5, label="Offense Loss")
        ax.plot(self.rounds, self.defense_losses, "r-", alpha=0.5, label="Defense Loss")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
        plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load models ──
    from unsloth import FastLanguageModel

    print(f"Loading offense model: {MODEL_NAME}...")
    off_model, off_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
        fast_inference=FAST_INFERENCE,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
    )
    off_model = FastLanguageModel.get_peft_model(
        off_model, r=LORA_RANK, lora_alpha=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    print(f"Loading defense model: {MODEL_NAME}...")
    def_model, def_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
        fast_inference=FAST_INFERENCE,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
    )
    def_model = FastLanguageModel.get_peft_model(
        def_model, r=LORA_RANK, lora_alpha=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=43,
    )

    # ── Optimizers ──
    off_optimizer = AdamW(off_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    def_optimizer = AdamW(def_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # ── Environment ──
    from football_env.server.environment import FootballDriveEnvironment
    env = FootballDriveEnvironment()

    # ── Training loop ──
    log = TrainingLog()
    print(f"Starting adversarial training: {NUM_ROUNDS} rounds, {EPISODES_PER_ROUND} episodes/round")

    for round_num in range(1, NUM_ROUNDS + 1):
        # Collect episodes
        off_model.eval()
        def_model.eval()
        off_episodes, def_episodes = collect_episodes(
            env, off_model, off_tokenizer, def_model, def_tokenizer,
            EPISODES_PER_ROUND, device,
        )

        # Count drive results
        drive_results = {}
        for obs_hist in [off_episodes]:
            # Last entry per drive has the drive-end reward baked in
            pass  # drive results tracked via env observation

        # Update offense
        off_loss = reinforce_update(off_model, off_tokenizer, off_optimizer, off_episodes, device)

        # Update defense
        def_loss = reinforce_update(def_model, def_tokenizer, def_optimizer, def_episodes, device)

        # Log
        off_mean = np.mean([r for _, _, r in off_episodes])
        def_mean = np.mean([r for _, _, r in def_episodes])
        log.log_round(round_num, off_episodes, def_episodes, off_loss, def_loss, {})
        print(f"Round {round_num}/{NUM_ROUNDS} | Off reward: {off_mean:.3f} | Def reward: {def_mean:.3f} | Off loss: {off_loss:.4f} | Def loss: {def_loss:.4f} | Episodes: {len(off_episodes)} plays")

        # Save checkpoints
        if round_num % SAVE_EVERY == 0:
            print(f"  Saving checkpoint at round {round_num}...")
            off_model.save_lora(os.path.join(OUTPUT_DIR, f"offense_lora_r{round_num}"))
            def_model.save_lora(os.path.join(OUTPUT_DIR, f"defense_lora_r{round_num}"))
            log.save(OUTPUT_DIR)

    # Final save
    print("Saving final models...")
    off_model.save_lora(os.path.join(OUTPUT_DIR, "offense_lora_final"))
    def_model.save_lora(os.path.join(OUTPUT_DIR, "defense_lora_final"))
    log.save(OUTPUT_DIR)
    print(f"Training complete! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script parses (don't run training)**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -c "import ast; ast.parse(open('train_adversarial.py').read()); print('Script parses OK')"`
Expected: `Script parses OK`

**Step 3: Commit**

```bash
git add train_adversarial.py
git commit -m "feat: adversarial REINFORCE training script for offense vs defense"
```

---

### Task 8: Smoke Test — Full Pipeline

**Files:**
- Create: `tests/test_smoke.py`

This test verifies the full pipeline works end-to-end without GPU (using the environment directly with random/hardcoded actions).

**Step 1: Write smoke test**

```python
"""Smoke test: full drive simulation with prompt formatting."""
from football_env.server.environment import FootballDriveEnvironment
from football_env.models import GameAction, OffenseAction, DefenseAction
from football_env.prompts import (
    format_offense_obs, format_defense_obs,
    parse_offense_response, parse_defense_response,
)


def test_full_pipeline_smoke():
    """Simulate a drive with prompt formatting and response parsing."""
    env = FootballDriveEnvironment()
    obs = env.reset(seed=42)

    plays = 0
    while not obs.done and plays < 30:
        # Format prompts (verify they don't crash)
        off_prompt = format_offense_obs(obs)
        def_prompt = format_defense_obs(obs)

        assert len(off_prompt) == 2  # system + user
        assert len(def_prompt) == 2

        # Simulate LLM responses (hardcoded valid JSON)
        if obs.down == 4:
            off_response = '{"offenseFormation": "SHOTGUN", "playType": "punt"}'
        else:
            off_response = '{"offenseFormation": "SHOTGUN", "playType": "pass", "designedPass": "short_middle", "receiverAlignment": "3x1"}'
        def_response = '{"defFormation": "Nickel (4-2-5)", "pff_manZone": "Zone", "pff_passCoverage": "Cover-3", "passRushers": 4}'

        off_action = parse_offense_response(off_response)
        def_action = parse_defense_response(def_response)

        action = GameAction(offense=off_action, defense=def_action)
        obs = env.step(action)
        plays += 1

    assert obs.done is True
    assert obs.drive_result != ""
    print(f"Drive ended after {plays} plays: {obs.drive_result}")


def test_parse_garbage_input():
    """Verify parsers handle garbage gracefully."""
    off = parse_offense_response("this is not json at all")
    assert off.offenseFormation == "SHOTGUN"  # default fallback

    def_ = parse_defense_response("")
    assert def_.defFormation == "4-3"  # default fallback
```

**Step 2: Run smoke test**

Run: `cd /Users/anthonyfletcher/Projects/hack && python -m pytest tests/test_smoke.py -v -s`
Expected: All tests pass, prints drive result

**Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: smoke test for full adversarial pipeline"
```

---

### Task 9: Update HF Space Deployment Files

**Files:**
- Modify: `hf_space/` directory to match new environment

**Step 1: Copy updated files to hf_space/**

```bash
cp football_env/models.py hf_space/models.py
cp football_env/validation.py hf_space/validation.py
cp football_env/play_outcome_model.py hf_space/play_outcome_model.py
cp football_env/server/environment.py hf_space/environment.py
cp football_env/server/app.py hf_space/app.py
```

Note: `hf_space/app.py` imports will need adjusting since it's flat (no package structure). The environment.py imports will need to be changed from relative to direct imports.

**Step 2: Fix imports in hf_space files**

In `hf_space/environment.py`, change:
```python
from ..models import GameAction, GameObservation, PlayRecord
from ..validation import validate_offense, validate_defense
from ..play_outcome_model import PlayOutcomeModel
```
to:
```python
from models import GameAction, GameObservation, PlayRecord
from validation import validate_offense, validate_defense
from play_outcome_model import PlayOutcomeModel
```

In `hf_space/app.py`, change:
```python
from .environment import FootballDriveEnvironment
from ..models import GameAction, GameObservation
```
to:
```python
from environment import FootballDriveEnvironment
from models import GameAction, GameObservation
```

**Step 3: Copy model data files**

```bash
cp data/outcome_classifier.joblib hf_space/
cp data/yards_q*.joblib hf_space/
cp data/model_features.json hf_space/
cp data/label_encoders.json hf_space/
```

**Step 4: Update hf_space/requirements.txt**

```
openenv-core>=0.2.1
uvicorn>=0.30.0
pydantic>=2.0
scikit-learn>=1.3
joblib>=1.3
numpy>=1.24
```

**Step 5: Commit**

```bash
git add hf_space/
git commit -m "feat: update HF Space deployment for adversarial environment"
```

---

### Task 10: Update CLAUDE.md and Memory

**Files:**
- Modify: `CLAUDE.md`
- Modify: `memory/MEMORY.md`

**Step 1: Update CLAUDE.md**

Add to the Action Space section:

```
## Action Space (Adversarial — Two Agents)

### Offense Action (Hierarchical)
Level 1: offenseFormation — 7 choices (SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL, JUMBO, WILDCAT)
Level 2: playType — run, pass, play_action, punt, field_goal
Level 3: designedPass + receiverAlignment (pass) or pff_runConceptPrimary (run)

### Defense Action (Hierarchical)
Level 1: defFormation — 10 choices
Level 2: pff_manZone — Man, Zone
Level 3: pff_passCoverage — constrained by man/zone
Level 4: passRushers — constrained by formation

### Composite
GameAction = OffenseAction + DefenseAction (submitted together each step)
```

**Step 2: Update the Reward section**

```
## Reward (Adversarial)
### Offense: per-play yards*0.1 + first down +1 + drive-end (+7 TD, +3 FG, -2 turnover, 0 punt)
### Defense: per-play -yards*0.1 + drive-end (-7 TD, -3 FG, +5 turnover, +1 punt)
### Contradiction penalty: -3 for invalid action combos (both agents)
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for adversarial environment"
```

---

## Task Dependency Order

```
Task 1 (Models) → Task 2 (Validation) → Task 3 (Environment) → Task 4 (App/Client)
                                                                      ↓
Task 5 (Env Tests) ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
       ↓
Task 6 (Prompts) → Task 7 (Training Script) → Task 8 (Smoke Test)
                                                      ↓
                                               Task 9 (HF Space)
                                                      ↓
                                               Task 10 (Docs)
```

Tasks 1-4 are the critical path. Tasks 5-8 verify correctness. Tasks 9-10 are deployment/docs.
