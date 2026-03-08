from typing import List
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class FootballAction(Action):
    formation: str = Field(..., description="Offensive formation: SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL")
    play_type: str = Field(..., description="Play type: run, pass, play_action")


class FootballObservation(Observation):
    down: int = Field(default=1)
    distance: int = Field(default=10)
    field_position: int = Field(default=25, description="Yards from own endzone")
    defense_formation: str = Field(default="")
    play_result: str = Field(default="", description="Human-readable result after step")
    yards_gained: int = Field(default=0)
    valid_formations: List[str] = Field(default_factory=list)
    valid_play_types: List[str] = Field(default_factory=list)
