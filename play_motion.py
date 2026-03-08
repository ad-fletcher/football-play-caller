"""
play_motion.py — Generate motion trail tuples for all 22 players based on play data.

Each trail is a tuple:
    ((start_x, start_y), (end_x, end_y), role, side, is_key_actor)

- side: "offense" or "defense"
- is_key_actor: True for ball carrier on run, target WR + QB on pass, all rushers on pass
"""

_FIELD_MIN_X = 0.0
_FIELD_MAX_X = 120.0
_FIELD_MIN_Y = 0.0
_FIELD_MAX_Y = 53.3
_CY = 26.65  # field center y


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _cx(x: float) -> float:
    return _clamp(x, _FIELD_MIN_X, _FIELD_MAX_X)


def _cy(y: float) -> float:
    return _clamp(y, _FIELD_MIN_Y, _FIELD_MAX_Y)


# ---------------------------------------------------------------------------
# Pass route parsing
# ---------------------------------------------------------------------------

def _parse_designed_pass(designed_pass: str) -> tuple[float, str]:
    """Return (depth_yards, y_direction) for the primary route.
    y_direction: 'left', 'right', or 'center'
    """
    dp = (designed_pass or "").lower()
    if "deep_right" in dp:
        return 20.0, "right"
    if "deep_left" in dp:
        return 20.0, "left"
    if "screen_middle" in dp:
        return 3.0, "center"
    if "screen_right" in dp:
        return 3.0, "right"
    if "screen_left" in dp:
        return 3.0, "left"
    if "short_middle" in dp:
        return 8.0, "center"
    if "short_right" in dp:
        return 8.0, "right"
    if "short_left" in dp:
        return 8.0, "left"
    if "mid_right" in dp:
        return 14.0, "right"
    if "mid_left" in dp:
        return 14.0, "left"
    if "mid_middle" in dp:
        return 14.0, "center"
    # default
    return 10.0, "center"


def _pick_target_wr(wr_positions: list[tuple], y_direction: str) -> tuple | None:
    """Pick the WR/TE closest to the route direction."""
    if not wr_positions:
        return None
    if y_direction == "right":
        # right side = larger y values (y > _CY)
        candidates = [p for p in wr_positions if p[1] >= _CY]
        if candidates:
            return max(candidates, key=lambda p: p[1])
        return min(wr_positions, key=lambda p: abs(p[1] - _CY))
    if y_direction == "left":
        candidates = [p for p in wr_positions if p[1] <= _CY]
        if candidates:
            return min(candidates, key=lambda p: p[1])
        return min(wr_positions, key=lambda p: abs(p[1] - _CY))
    # center
    return min(wr_positions, key=lambda p: abs(p[1] - _CY))


# ---------------------------------------------------------------------------
# Run concept parsing
# ---------------------------------------------------------------------------

def _parse_run_concept(run_concept: str) -> tuple[float, float]:
    """Return (dx, dy) displacement for the RB on a run play."""
    rc = (run_concept or "").upper()
    if "OUTSIDE ZONE" in rc:
        return 6.0, -4.0   # wide right
    if "INSIDE ZONE" in rc:
        return 5.0, 0.0    # straight ahead
    if "POWER" in rc or "MAN" in rc:
        return 5.0, 1.5    # slight gap left
    if "TRAP" in rc:
        return 5.0, 1.0
    if "COUNTER" in rc:
        return 5.0, -2.0
    # default
    return 5.0, 0.0


# ---------------------------------------------------------------------------
# Offense helpers
# ---------------------------------------------------------------------------

_OL_ROLES = {"C", "LG", "RG", "LT", "RT"}
_RECEIVER_ROLES = {"WR1", "WR2", "WR3", "WR4", "TE", "TE1", "TE2"}


def _pass_offense_trails(play: dict, positions: list[tuple]) -> list[tuple]:
    """Generate offense motion trails for a pass play (including play_action)."""
    depth, y_direction = _parse_designed_pass(play.get("off_designedPass", ""))
    is_screen = depth <= 3.0

    # Collect receivers (WRs and TEs) for targeting
    receivers = [(x, y, role) for x, y, role in positions if role in _RECEIVER_ROLES]
    target = _pick_target_wr(receivers, y_direction)

    trails = []
    for x, y, role in positions:
        start = (x, y)

        if role == "QB":
            # QB drops back 6 yards
            end = (_cx(x - 6), _cy(y))
            trails.append((start, end, role, "offense", True))

        elif role in _OL_ROLES:
            # OL: minimal pocket protection movement
            # Guards/tackles shift slightly outward (~1 yd)
            if role == "LG" or role == "LT":
                end = (_cx(x), _cy(y - 1.0))
            elif role == "RG" or role == "RT":
                end = (_cx(x), _cy(y + 1.0))
            else:  # C
                end = (_cx(x), _cy(y))
            trails.append((start, end, role, "offense", False))

        elif role == "RB":
            if is_screen:
                # Screen: RB releases to flat forward
                flat_y = _cy(y + 3.0) if y_direction == "right" else _cy(y - 3.0)
                end = (_cx(x + 3), flat_y)
            else:
                # Pass pro / chip block: slight shift to one side
                end = (_cx(x - 1), _cy(y - 2.0))
            trails.append((start, end, role, "offense", False))

        elif role == "FB":
            # FB stays in to block or runs short route
            end = (_cx(x), _cy(y))
            trails.append((start, end, role, "offense", False))

        else:
            # WR or TE
            if target and role == target[2] and (x, y) == (target[0], target[1]):
                # Primary target: run the designed route
                if y_direction == "right":
                    y_shift = -5.0  # toward right sideline
                elif y_direction == "left":
                    y_shift = 5.0   # toward left sideline
                else:
                    y_shift = 0.0
                end = (_cx(x + depth), _cy(y + y_shift))
                trails.append((start, end, role, "offense", True))
            else:
                # Other receivers: clearing routes (shorter, opposite direction)
                clearing_depth = depth * 0.5 + 2.0
                if y_direction == "right":
                    clear_y_shift = 3.0  # run away from target side
                elif y_direction == "left":
                    clear_y_shift = -3.0
                else:
                    # for center routes, receivers run out to the sides
                    clear_y_shift = 4.0 if y > _CY else -4.0
                end = (_cx(x + clearing_depth), _cy(y + clear_y_shift))
                trails.append((start, end, role, "offense", False))

    return trails


def _run_offense_trails(play: dict, positions: list[tuple]) -> list[tuple]:
    """Generate offense motion trails for a run play."""
    dx, dy = _parse_run_concept(play.get("off_runConcept", ""))

    trails = []
    for x, y, role in positions:
        start = (x, y)

        if role == "RB":
            end = (_cx(x + dx), _cy(y + dy))
            trails.append((start, end, role, "offense", True))

        elif role == "FB":
            # FB leads block
            end = (_cx(x + dx - 1), _cy(y + dy))
            trails.append((start, end, role, "offense", False))

        elif role == "QB":
            # QB hands off or bootleg
            if dx > 0:
                end = (_cx(x + 1), _cy(y))
            else:
                end = (_cx(x - 1), _cy(y))
            trails.append((start, end, role, "offense", False))

        elif role in _OL_ROLES:
            # OL drives block in run direction (x+2, slight y shift toward gap)
            y_shift = dy * 0.2  # subtle lateral movement
            end = (_cx(x + 2), _cy(y + y_shift))
            trails.append((start, end, role, "offense", False))

        else:
            # WRs/TEs: run forward to block downfield
            end = (_cx(x + 3), _cy(y))
            trails.append((start, end, role, "offense", False))

    return trails


def _fg_punt_offense_trails(positions: list[tuple]) -> list[tuple]:
    """Generate offense trails for field goal or punt."""
    trails = []
    for x, y, role in positions:
        start = (x, y)
        end = (_cx(x + 1), _cy(y))
        trails.append((start, end, role, "offense", False))
    return trails


# ---------------------------------------------------------------------------
# Defense helpers
# ---------------------------------------------------------------------------

def _split_defense_by_type(positions: list[tuple]) -> tuple[list, list, list, list, list]:
    """Split defense positions into DL, LBs, CBs, SS, FS lists."""
    dl, lb, cb, ss, fs = [], [], [], [], []
    for pos in positions:
        role = pos[2]
        if role.startswith("DL"):
            dl.append(pos)
        elif role.startswith("LB"):
            lb.append(pos)
        elif role in ("CB1", "CB2", "NCB") or role.startswith("CB"):
            cb.append(pos)
        elif role == "SS":
            ss.append(pos)
        elif role == "FS":
            fs.append(pos)
        else:
            # Extra DBs or other roles — treat as CB
            cb.append(pos)
    return dl, lb, cb, ss, fs


def _pass_defense_trails(play: dict, positions: list[tuple]) -> list[tuple]:
    """Generate defense motion trails for a pass play."""
    def_rushers = int(play.get("def_rushers", 4))
    man_zone = (play.get("def_manZone") or "Zone").strip()
    is_zone = man_zone.lower() != "man"

    dl, lb, cb, ss, fs = _split_defense_by_type(positions)

    # Determine which players rush
    rushers_assigned = 0
    rushing_roles: set[str] = set()

    # First, DL rush
    for pos in dl:
        if rushers_assigned < def_rushers:
            rushing_roles.add(pos[2])
            rushers_assigned += 1

    # If we need more rushers, pull from LBs (LB1 first)
    for pos in lb:
        if rushers_assigned < def_rushers:
            rushing_roles.add(pos[2])
            rushers_assigned += 1

    trails = []

    for x, y, role in positions:
        start = (x, y)

        if role in rushing_roles:
            # Rusher: charge toward LOS (smaller x)
            end = (_cx(x - 5), _cy(y))
            trails.append((start, end, role, "defense", True))

        elif role.startswith("LB"):
            # Non-rushing LB
            if is_zone:
                end = (_cx(x + 2), _cy(y))  # drop to hook/curl zone
            else:
                # Man: track nearest WR — stay put (simplified)
                end = (_cx(x), _cy(y))
            trails.append((start, end, role, "defense", False))

        elif role.startswith("CB") or role == "NCB":
            if is_zone:
                end = (_cx(x + 3), _cy(y))  # drop back in zone
            else:
                # Man: hold position (tracking WR)
                end = (_cx(x), _cy(y))
            trails.append((start, end, role, "defense", False))

        elif role == "SS":
            if is_zone:
                # Drop back, drift toward flat
                end = (_cx(x + 3), _cy(y - 2.0))
            else:
                end = (_cx(x), _cy(y))
            trails.append((start, end, role, "defense", False))

        elif role == "FS":
            if is_zone:
                end = (_cx(x + 5), _cy(y))  # drop deep
            else:
                end = (_cx(x + 2), _cy(y))  # drift toward nearest deep WR
            trails.append((start, end, role, "defense", False))

        else:
            # Any unclassified role: minimal movement
            end = (_cx(x), _cy(y))
            trails.append((start, end, role, "defense", False))

    return trails


def _run_defense_trails(play: dict, positions: list[tuple]) -> list[tuple]:
    """Generate defense motion trails for a run play."""
    _, dy = _parse_run_concept(play.get("off_runConcept", ""))
    # Flow direction: if dy > 0 defense flows right (y increases), else left
    flow_y = 2.0 if dy >= 0 else -2.0

    dl, lb, cb, ss, fs = _split_defense_by_type(positions)

    trails = []
    for x, y, role in positions:
        start = (x, y)

        if role.startswith("DL"):
            # DL push into backfield
            end = (_cx(x - 2), _cy(y))
            trails.append((start, end, role, "defense", False))

        elif role.startswith("LB"):
            # LBs flow toward run direction
            end = (_cx(x - 2), _cy(y + flow_y))
            trails.append((start, end, role, "defense", False))

        else:
            # CBs, SS, FS come up toward LOS
            end = (_cx(x - 3), _cy(y))
            trails.append((start, end, role, "defense", False))

    return trails


def _fg_punt_defense_trails(positions: list[tuple]) -> list[tuple]:
    """Generate defense trails for field goal or punt."""
    trails = []
    for x, y, role in positions:
        start = (x, y)
        end = (_cx(x - 1), _cy(y))
        trails.append((start, end, role, "defense", False))
    return trails


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_motion_trails(
    play: dict,
    off_positions: list[tuple],
    def_positions: list[tuple],
) -> list[tuple]:
    """
    Generate motion trails for all 22 players based on play data.

    Args:
        play: dict with keys:
            off_playType, off_designedPass, off_runConcept,
            off_formation, def_formation, def_manZone, def_coverage,
            def_rushers, yards, result, drive_result, yardline
        off_positions: list of 11 (x, y, role) tuples from get_offense_positions()
        def_positions: list of 11 (x, y, role) tuples from get_defense_positions()

    Returns:
        List of exactly len(off_positions) + len(def_positions) tuples, each:
            ((start_x, start_y), (end_x, end_y), role, side, is_key_actor)
    """
    play_type = (play.get("off_playType") or "pass").lower()

    # --- Offense trails ---
    if play_type in ("pass", "play_action"):
        off_trails = _pass_offense_trails(play, off_positions)
    elif play_type == "run":
        off_trails = _run_offense_trails(play, off_positions)
    else:
        # field_goal, punt, or unknown
        off_trails = _fg_punt_offense_trails(off_positions)

    # --- Defense trails ---
    if play_type in ("pass", "play_action"):
        def_trails = _pass_defense_trails(play, def_positions)
    elif play_type == "run":
        def_trails = _run_defense_trails(play, def_positions)
    else:
        def_trails = _fg_punt_defense_trails(def_positions)

    return off_trails + def_trails


def get_ball_flight(play, trails):
    """Return ((qb_end_x, qb_end_y), (wr_end_x, wr_end_y)) for pass plays, else None."""
    if play.get("off_playType") not in ("pass", "play_action"):
        return None
    qb_end = None
    wr_end = None
    for start, end, role, side, is_key in trails:
        if side == "offense" and role == "QB":
            qb_end = end
        if side == "offense" and is_key and role not in ("QB", "RB", "FB", "C", "LG", "RG", "LT", "RT"):
            wr_end = end
    if qb_end is None or wr_end is None:
        return None
    return (qb_end, wr_end)
