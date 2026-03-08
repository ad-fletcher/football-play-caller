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
