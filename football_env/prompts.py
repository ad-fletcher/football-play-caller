"""Prompt formatting and response parsing for offense and defense LLMs."""
import json
from .models import GameObservation, OffenseAction, DefenseAction
from .validation import (
    VALID_OFFENSE_FORMATIONS, VALID_PLAY_TYPES,
    VALID_DESIGNED_PASS, VALID_RECEIVER_ALIGNMENT, VALID_RUN_CONCEPTS,
    VALID_DEF_FORMATIONS, VALID_MAN_ZONE, VALID_MAN_COVERAGES, VALID_ZONE_COVERAGES,
    RUSHER_RANGES,
)


OFFENSE_SYSTEM_PROMPT = """\
You are an NFL offensive coordinator. Given the game situation and the defense's tendencies, call a play.

Respond with ONLY a JSON object. Your formation and play type determine what other fields are required:

For pass or play_action:
{"offenseFormation": "<FORM>", "playType": "pass", "designedPass": "<ROUTE>", "receiverAlignment": "<ALIGN>"}

For run:
{"offenseFormation": "<FORM>", "playType": "run", "pff_runConceptPrimary": "<CONCEPT>"}

For punt (4th down only):
{"offenseFormation": "<FORM>", "playType": "punt"}

For field_goal (4th down only):
{"offenseFormation": "<FORM>", "playType": "field_goal"}

Valid formations: """ + ", ".join(VALID_OFFENSE_FORMATIONS) + """
Valid pass routes: """ + ", ".join([d for d in VALID_DESIGNED_PASS if d != "none"]) + """
Valid alignments: """ + ", ".join([r for r in VALID_RECEIVER_ALIGNMENT if r != "unknown"]) + """
Valid run concepts: """ + ", ".join([r for r in VALID_RUN_CONCEPTS if r not in ("none", "UNDEFINED")])


DEFENSE_SYSTEM_PROMPT = """\
You are an NFL defensive coordinator. Given the game situation and the offense's tendencies, call a defense.

Respond with ONLY a JSON object:
{"defFormation": "<FORM>", "pff_manZone": "Man|Zone", "pff_passCoverage": "<COVERAGE>", "passRushers": <N>}

Valid formations: """ + ", ".join(VALID_DEF_FORMATIONS) + """
Man coverages: """ + ", ".join(VALID_MAN_COVERAGES) + """
Zone coverages: """ + ", ".join(VALID_ZONE_COVERAGES) + """
Pass rushers must be in valid range for your formation."""


def format_offense_obs(obs: GameObservation) -> list[dict]:
    """Format observation as chat messages for the offense LLM."""
    situation = (
        f"{_down_str(obs.down)} & {obs.yardsToGo} at the {_field_pos_str(obs.absoluteYardlineNumber)}. "
        f"Q{obs.quarter}, {_clock_str(obs.gameClock_seconds)}. "
        f"{'Trailing' if obs.score_diff < 0 else 'Leading' if obs.score_diff > 0 else 'Tied'}"
        f"{' by ' + str(abs(obs.score_diff)) if obs.score_diff != 0 else ''}."
    )

    if obs.defense_history:
        tendencies = _summarize_defense_history(obs.defense_history)
        situation += f"\n\nDefense tendencies this drive:\n{tendencies}"

    if obs.last_play_result:
        situation += f"\n\nLast play: {obs.last_play_result} for {obs.last_play_yards} yards."

    return [
        {"role": "system", "content": OFFENSE_SYSTEM_PROMPT},
        {"role": "user", "content": situation},
    ]


def format_defense_obs(obs: GameObservation) -> list[dict]:
    """Format observation as chat messages for the defense LLM."""
    situation = (
        f"Offense has {_down_str(obs.down)} & {obs.yardsToGo} at the {_field_pos_str(obs.absoluteYardlineNumber)}. "
        f"Q{obs.quarter}, {_clock_str(obs.gameClock_seconds)}. "
        f"{'They trail' if obs.score_diff < 0 else 'They lead' if obs.score_diff > 0 else 'Tied'}"
        f"{' by ' + str(abs(obs.score_diff)) if obs.score_diff != 0 else ''}."
    )

    if obs.offense_history:
        tendencies = _summarize_offense_history(obs.offense_history)
        situation += f"\n\nOffense tendencies this drive:\n{tendencies}"

    if obs.last_play_result:
        situation += f"\n\nLast play: {obs.last_play_result} for {obs.last_play_yards} yards."

    return [
        {"role": "system", "content": DEFENSE_SYSTEM_PROMPT},
        {"role": "user", "content": situation},
    ]


def parse_offense_response(text: str) -> OffenseAction:
    """Parse LLM text output into OffenseAction. Returns defaults for unparseable input."""
    parsed = _extract_json(text)
    if parsed is None:
        return OffenseAction(offenseFormation="SHOTGUN", playType="pass", designedPass="short_middle", receiverAlignment="3x1")
    return OffenseAction(
        offenseFormation=parsed.get("offenseFormation", "SHOTGUN"),
        playType=parsed.get("playType", "pass"),
        designedPass=parsed.get("designedPass", "none"),
        receiverAlignment=parsed.get("receiverAlignment", "none"),
        pff_runConceptPrimary=parsed.get("pff_runConceptPrimary", "none"),
    )


def parse_defense_response(text: str) -> DefenseAction:
    """Parse LLM text output into DefenseAction. Returns defaults for unparseable input."""
    parsed = _extract_json(text)
    if parsed is None:
        return DefenseAction(defFormation="4-3", pff_manZone="Zone", pff_passCoverage="Cover-3", passRushers=4)
    return DefenseAction(
        defFormation=parsed.get("defFormation", "4-3"),
        pff_manZone=parsed.get("pff_manZone", "Zone"),
        pff_passCoverage=parsed.get("pff_passCoverage", "Cover-3"),
        passRushers=int(parsed.get("passRushers", 4)),
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError):
        return None


def _down_str(down: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(down, f"{down}th")


def _field_pos_str(yard_line: int) -> str:
    if yard_line <= 50:
        return f"own {yard_line}"
    return f"opponent {100 - yard_line}"


def _clock_str(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def _summarize_defense_history(history) -> str:
    lines = []
    for i, play in enumerate(history, 1):
        lines.append(f"  Play {i}: {play.defFormation}, {play.pff_passCoverage} ({play.pff_manZone}), {play.passRushers} rushers → {play.result} ({play.yardsGained} yds)")
    return "\n".join(lines)


def _summarize_offense_history(history) -> str:
    lines = []
    for i, play in enumerate(history, 1):
        detail = play.designedPass if play.playType != "run" else play.pff_runConceptPrimary
        lines.append(f"  Play {i}: {play.offenseFormation} {play.playType} ({detail}) → {play.result} ({play.yardsGained} yds)")
    return "\n".join(lines)
