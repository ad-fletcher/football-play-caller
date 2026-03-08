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
