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
