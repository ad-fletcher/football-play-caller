"""
Offense formation position templates for 11v11 player visualization.

Each formation is a list of 11 (x_offset, y, role) tuples where:
  - x_offset: yards relative to ball_x (negative = behind line of scrimmage)
  - y: absolute field y-coordinate (0-53.3, center is 26.65)
  - role: string label for the player (QB, RB, FB, C, LG, RG, LT, RT, TE, WR1-WR4)

Line of scrimmage is at x_offset=0. OL lines up at x_offset=0 (on the ball).
Backfield players have negative x_offset (behind LOS).
WRs typically line up at x_offset=0 (at LOS) spread wide.
"""

# Field center y-coordinate
_CY = 26.65

# OL spacing: C at center, guards 1.5yd out, tackles 3yd out
_OL = [
    (0, _CY,        "C"),
    (0, _CY - 1.5,  "LG"),
    (0, _CY + 1.5,  "RG"),
    (0, _CY - 3.0,  "LT"),
    (0, _CY + 3.0,  "RT"),
]

# Formation templates: list of 11 (x_offset, y, role) tuples
OFFENSE_FORMATIONS = {
    # SHOTGUN: QB 5yd back, RB offset beside QB, 5 OL, TE on right, 3 WRs spread
    "SHOTGUN": _OL + [
        (-5,  _CY,        "QB"),
        (-4,  _CY + 4.0,  "RB"),
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 10.0, "WR1"),  # split wide left
        ( 0,  _CY + 12.0, "WR2"),  # split wide right
        ( 0,  _CY - 18.0, "WR3"),  # far wide left
    ],

    # SINGLEBACK: QB under center, RB 6yd back, 5 OL, TE on right, 3 WRs
    "SINGLEBACK": _OL + [
        (-1,  _CY,        "QB"),
        (-6,  _CY,        "RB"),
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 10.0, "WR1"),
        ( 0,  _CY + 12.0, "WR2"),
        ( 0,  _CY - 18.0, "WR3"),
    ],

    # I_FORM: QB under center, FB 4yd back, RB 7yd back, 5 OL, TE, 2 WRs
    "I_FORM": _OL + [
        (-1,  _CY,        "QB"),
        (-4,  _CY,        "FB"),
        (-7,  _CY,        "RB"),
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 10.0, "WR1"),
        ( 0,  _CY + 12.0, "WR2"),
    ],

    # EMPTY: QB in shotgun, no RB, 5 OL, 5 receivers spread across field
    "EMPTY": _OL + [
        (-5,  _CY,        "QB"),
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 10.0, "WR1"),
        ( 0,  _CY + 12.0, "WR2"),
        ( 0,  _CY - 18.0, "WR3"),
        ( 0,  _CY + 20.0, "WR4"),
    ],

    # PISTOL: QB 3yd back, RB 6yd directly behind QB, 5 OL, TE, 3 WRs
    "PISTOL": _OL + [
        (-3,  _CY,        "QB"),
        (-6,  _CY,        "RB"),
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 10.0, "WR1"),
        ( 0,  _CY + 12.0, "WR2"),
        ( 0,  _CY - 18.0, "WR3"),
    ],

    # JUMBO: 5 OL, 2 TEs flanking OL, QB under center, FB, RB, 1 WR
    "JUMBO": [
        ( 0,  _CY,        "C"),
        ( 0,  _CY - 1.5,  "LG"),
        ( 0,  _CY + 1.5,  "RG"),
        ( 0,  _CY - 3.0,  "LT"),
        ( 0,  _CY + 3.0,  "RT"),
        ( 0,  _CY - 4.5,  "TE1"),  # left of LT
        ( 0,  _CY + 4.5,  "TE2"),  # right of RT
        (-1,  _CY,        "QB"),
        (-4,  _CY,        "FB"),
        (-6,  _CY,        "RB"),
        ( 0,  _CY - 12.0, "WR1"),
    ],

    # WILDCAT: RB takes snap, QB wide left as receiver, 5 OL, TE, 3 WRs
    "WILDCAT": [
        ( 0,  _CY,        "C"),
        ( 0,  _CY - 1.5,  "LG"),
        ( 0,  _CY + 1.5,  "RG"),
        ( 0,  _CY - 3.0,  "LT"),
        ( 0,  _CY + 3.0,  "RT"),
        (-3,  _CY,        "RB"),    # RB takes the snap
        (-1,  _CY - 8.0,  "QB"),   # QB lined up wide left as receiver
        ( 0,  _CY + 5.5,  "TE"),
        ( 0,  _CY - 12.0, "WR1"),
        ( 0,  _CY + 12.0, "WR2"),
        ( 0,  _CY - 20.0, "WR3"),
    ],
}


# ---------------------------------------------------------------------------
# Defense formation templates
# ---------------------------------------------------------------------------
# Each template is a list of 11 (x_offset, y, role) tuples where:
#   - x_offset: positive yards ahead of the line of scrimmage (defensive side)
#   - y: absolute field y-coordinate (0-53.3, center is 26.65)
#   - role: string label (DL1-DL4, LB1-LB3, CB1, CB2, SS, FS)

DEFENSE_FORMATIONS = {
    # 4-3: 4 DL at +1, 3 LB at +4, 2 CB wide at +5, SS at +8, FS at +12
    "4-3": [
        (+1, _CY - 4.5,  "DL1"),
        (+1, _CY - 1.5,  "DL2"),
        (+1, _CY + 1.5,  "DL3"),
        (+1, _CY + 4.5,  "DL4"),
        (+4, _CY - 5.0,  "LB1"),
        (+4, _CY,        "LB2"),
        (+4, _CY + 5.0,  "LB3"),
        (+5,  8.0,       "CB1"),   # wide left corner
        (+5, 45.3,       "CB2"),   # wide right corner
        (+8, _CY + 3.0,  "SS"),
        (+12, _CY,       "FS"),
    ],

    # 3-4: 3 DL at +1, 4 LB (2 outside +3, 2 inside +4), 2 CB wide, SS, FS
    "3-4": [
        (+1, _CY - 3.0,  "DL1"),
        (+1, _CY,        "DL2"),
        (+1, _CY + 3.0,  "DL3"),
        (+3, _CY - 7.0,  "LB1"),   # outside LB left
        (+4, _CY - 2.5,  "LB2"),   # inside LB left
        (+4, _CY + 2.5,  "LB3"),   # inside LB right
        (+3, _CY + 7.0,  "LB4"),   # outside LB right
        (+5,  8.0,       "CB1"),
        (+5, 45.3,       "CB2"),
        (+8, _CY + 3.0,  "SS"),
        (+12, _CY,       "FS"),
    ],

    # Nickel (4-2-5): 4 DL at +1, 2 LB at +4, NCB in slot, 2 wide CB, SS, FS
    "Nickel (4-2-5)": [
        (+1, _CY - 4.5,  "DL1"),
        (+1, _CY - 1.5,  "DL2"),
        (+1, _CY + 1.5,  "DL3"),
        (+1, _CY + 4.5,  "DL4"),
        (+4, _CY - 3.5,  "LB1"),
        (+4, _CY + 3.5,  "LB2"),
        (+5,  8.0,       "CB1"),   # wide left corner
        (+5, 45.3,       "CB2"),   # wide right corner
        (+4, _CY,        "NCB"),   # nickel CB in slot (counts as CB for DB total)
        (+8, _CY + 3.0,  "SS"),
        (+12, _CY,       "FS"),
    ],
}


def _parse_defense_counts(formation: str) -> tuple[int, int, int]:
    """
    Parse DL/LB/DB counts from a formation string.

    Examples:
        "Dime (2-3-6)" -> (2, 3, 6)
        "Nickel (4-2-5)" -> (4, 2, 5)
        "5-2" -> (5, 2, 4)   [db = 11 - dl - lb]
        "4-3" -> (4, 3, 4)
        "3-4" -> (3, 4, 4)
        Unknown -> (4, 3, 4)
    """
    import re

    # Try to match parenthesized form: "Label (dl-lb-db)"
    m = re.search(r'\((\d+)-(\d+)-(\d+)\)', formation)
    if m:
        dl, lb, db = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return dl, lb, db

    # Try bare "dl-lb" form (e.g. "4-3", "3-4", "5-2")
    m = re.fullmatch(r'(\d+)-(\d+)', formation.strip())
    if m:
        dl, lb = int(m.group(1)), int(m.group(2))
        db = 11 - dl - lb
        return dl, lb, max(db, 0)

    # Default
    return 4, 3, 4


def _generate_defense_positions(dl: int, lb: int, db: int) -> list[tuple]:
    """
    Algorithmically generate 11 defense positions from unit counts.

    DL spread across y=[18,35] at x_offset=+1
    LB spread across y=[15,38] at x_offset=+4
    DBs: CB1 at y=8, CB2 at y=45 (wide), then SS, FS, then extra DBs in middle
    """
    positions = []

    # DL
    if dl == 1:
        ys_dl = [_CY]
    else:
        ys_dl = [18.0 + (35.0 - 18.0) * i / (dl - 1) for i in range(dl)]
    for i, y in enumerate(ys_dl):
        positions.append((+1, y, f"DL{i + 1}"))

    # LB
    if lb == 1:
        ys_lb = [_CY]
    else:
        ys_lb = [15.0 + (38.0 - 15.0) * i / (lb - 1) for i in range(lb)]
    for i, y in enumerate(ys_lb):
        positions.append((+4, y, f"LB{i + 1}"))

    # DBs
    # First two are always wide CBs; remaining fill SS, FS, then extra DBs
    db_positions = []
    if db >= 1:
        db_positions.append((+5, 8.0, "CB1"))
    if db >= 2:
        db_positions.append((+5, 45.3, "CB2"))
    if db >= 3:
        db_positions.append((+8, _CY + 3.0, "SS"))
    if db >= 4:
        db_positions.append((+12, _CY, "FS"))
    # Any remaining DBs placed in middle at x_offset=+6
    extra_db_count = db - 4
    if extra_db_count > 0:
        ys_extra = [_CY - 4 + 8 * i / max(extra_db_count - 1, 1)
                    for i in range(extra_db_count)] if extra_db_count > 1 else [_CY]
        for i, y in enumerate(ys_extra):
            db_positions.append((+6, y, f"DB{i + 4 + 1}"))

    positions.extend(db_positions)
    return positions


def get_defense_positions(formation: str, ball_x: float) -> list[tuple[float, float, str]]:
    """
    Return 11 player positions for the given defensive formation.

    Args:
        formation: Formation name (e.g. "4-3", "Nickel (4-2-5)"). Falls back to
                   algorithmic generation for unknown formations.
        ball_x: Absolute x-coordinate of the ball on the field (0-120).

    Returns:
        List of 11 (x, y, role) tuples with absolute field coordinates.
        x is clamped to [0, 120], y is clamped to [0, 53.3].
    """
    # Try exact match first, then case-insensitive
    template = DEFENSE_FORMATIONS.get(formation)
    if template is None:
        for key, val in DEFENSE_FORMATIONS.items():
            if key.lower() == formation.lower():
                template = val
                break

    if template is None:
        dl, lb, db = _parse_defense_counts(formation)
        template = _generate_defense_positions(dl, lb, db)

    positions = []
    for x_offset, y, role in template:
        x = max(0.0, min(120.0, ball_x + x_offset))
        y = max(0.0, min(53.3, y))
        positions.append((x, y, role))

    return positions


def get_offense_positions(formation: str, ball_x: float) -> list[tuple[float, float, str]]:
    """
    Return 11 player positions for the given offensive formation.

    Args:
        formation: Formation name (e.g. "SHOTGUN", "I_FORM"). Case-insensitive.
                   Falls back to SHOTGUN if unknown.
        ball_x: Absolute x-coordinate of the ball on the field (0-120).

    Returns:
        List of 11 (x, y, role) tuples with absolute field coordinates.
        x is clamped to [0, 120], y is clamped to [0, 53.3].
    """
    key = formation.upper()
    template = OFFENSE_FORMATIONS.get(key, OFFENSE_FORMATIONS["SHOTGUN"])

    positions = []
    for x_offset, y, role in template:
        x = max(0.0, min(120.0, ball_x + x_offset))
        y = max(0.0, min(53.3, y))
        positions.append((x, y, role))

    return positions
