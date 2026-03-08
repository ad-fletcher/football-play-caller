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
