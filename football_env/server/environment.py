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
