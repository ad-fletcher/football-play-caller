import json
import pathlib
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import FootballAction, FootballObservation


class FootballEnvironment(Environment):
    VALID_FORMATIONS = ["SHOTGUN", "SINGLEBACK", "EMPTY", "I_FORM", "PISTOL"]
    VALID_PLAY_TYPES = ["run", "pass", "play_action"]

    def __init__(self):
        data_path = pathlib.Path(__file__).parent / "lookup_table.json"
        with open(data_path) as f:
            data = json.load(f)
        self.lookup = data["lookup"]
        self.defense_names = list(data["defense_distribution"].keys())
        self.defense_weights = list(data["defense_distribution"].values())
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._defense = ""

    def reset(self, seed=None, **kwargs) -> FootballObservation:
        if seed is not None:
            random.seed(seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._defense = random.choices(self.defense_names, weights=self.defense_weights, k=1)[0]
        return FootballObservation(
            done=False,
            reward=0.0,
            down=1,
            distance=10,
            field_position=25,
            defense_formation=self._defense,
            play_result=f"1st & 10 from your own 25. Defense is in {self._defense}.",
            yards_gained=0,
            valid_formations=self.VALID_FORMATIONS,
            valid_play_types=self.VALID_PLAY_TYPES,
        )

    def step(self, action: FootballAction, **kwargs) -> FootballObservation:
        self._state.step_count += 1
        formation = action.formation
        play_type = action.play_type

        if formation not in self.VALID_FORMATIONS or play_type not in self.VALID_PLAY_TYPES:
            return FootballObservation(
                done=True,
                reward=-10.0,
                play_result=f"Invalid action: {formation} {play_type}",
                defense_formation=self._defense,
                valid_formations=self.VALID_FORMATIONS,
                valid_play_types=self.VALID_PLAY_TYPES,
            )

        key = f"{formation}|{play_type}|{self._defense}"

        if key not in self.lookup:
            fallback_keys = [k for k in self.lookup if k.startswith(f"{formation}|{play_type}|")]
            if fallback_keys:
                all_entries = []
                for fk in fallback_keys:
                    b = self.lookup[fk]
                    for i in range(len(b["rewards"])):
                        all_entries.append((b["rewards"][i], b["yards"][i], b["results"][i]))
                reward, yards, pass_result = random.choice(all_entries)
                result_desc = f"{play_type} from {formation} for {yards} yards."
            else:
                reward, yards = 0.0, 0
                result_desc = f"No data for {formation} {play_type}"
        else:
            bucket = self.lookup[key]
            idx = random.randrange(len(bucket["rewards"]))
            reward = bucket["rewards"][idx]
            pass_result = bucket["results"][idx]
            yards = bucket["yards"][idx]

            if pass_result == "IN":
                result_desc = f"INTERCEPTION! Pass from {formation} picked off. ({yards} yards on the play)"
            elif pass_result == "S":
                result_desc = f"SACK! QB taken down for {yards} yards."
            elif pass_result == "I":
                result_desc = f"Incomplete pass from {formation}. 0 yards."
            elif pass_result in ("C", "R"):
                result_desc = f"{'Pass' if pass_result == 'C' else 'Scramble'} from {formation} for {yards} yards."
            else:
                result_desc = f"{'Run' if play_type == 'run' else 'Play-action run'} from {formation} for {yards} yards."

        return FootballObservation(
            done=True,
            reward=float(reward),
            down=1,
            distance=10,
            field_position=25,
            defense_formation=self._defense,
            play_result=result_desc,
            yards_gained=int(yards),
            valid_formations=self.VALID_FORMATIONS,
            valid_play_types=self.VALID_PLAY_TYPES,
        )

    @property
    def state(self) -> State:
        return self._state
