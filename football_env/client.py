from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import FootballAction, FootballObservation


class FootballEnv(EnvClient[FootballAction, FootballObservation, State]):
    def _step_payload(self, action: FootballAction) -> Dict[str, Any]:
        return {"formation": action.formation, "play_type": action.play_type}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FootballObservation]:
        obs_data = payload.get("observation", payload)
        obs = FootballObservation(**obs_data)
        return StepResult(observation=obs, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
