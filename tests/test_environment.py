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
