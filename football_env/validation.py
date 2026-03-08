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
