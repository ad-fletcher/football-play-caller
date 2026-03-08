from formations import get_offense_positions

def test_shotgun_returns_11_players():
    positions = get_offense_positions("SHOTGUN", ball_x=60)
    assert len(positions) == 11
    roles = [p[2] for p in positions]
    assert "QB" in roles
    assert "RB" in roles
    assert len([r for r in roles if r.startswith("WR")]) >= 2

def test_shotgun_qb_behind_los():
    positions = get_offense_positions("SHOTGUN", ball_x=60)
    qb = [p for p in positions if p[2] == "QB"][0]
    assert qb[0] < 60  # QB is behind line of scrimmage

def test_unknown_formation_still_returns_11():
    positions = get_offense_positions("UNKNOWN_FORMATION", ball_x=60)
    assert len(positions) == 11

def test_positions_within_field_bounds():
    for form in ["SHOTGUN", "SINGLEBACK", "I_FORM", "EMPTY", "PISTOL", "JUMBO", "WILDCAT"]:
        positions = get_offense_positions(form, ball_x=60)
        for x, y, role in positions:
            assert 0 <= x <= 120, f"{form} {role} x={x} out of bounds"
            assert 0 <= y <= 53.3, f"{form} {role} y={y} out of bounds"
