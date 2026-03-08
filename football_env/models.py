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
