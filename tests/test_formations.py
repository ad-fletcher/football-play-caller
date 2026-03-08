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


from formations import get_defense_positions

def test_43_returns_11_players():
    positions = get_defense_positions("4-3", ball_x=60)
    assert len(positions) == 11

def test_43_has_correct_unit_counts():
    positions = get_defense_positions("4-3", ball_x=60)
    roles = [p[2] for p in positions]
    assert len([r for r in roles if r.startswith("DL")]) == 4
    assert len([r for r in roles if r.startswith("LB")]) == 3
    assert len([r for r in roles if r in ("CB1", "CB2", "FS", "SS")]) == 4

def test_defense_ahead_of_los():
    positions = get_defense_positions("4-3", ball_x=60)
    for x, y, role in positions:
        assert x >= 60, f"{role} at x={x} should be ahead of LOS at 60"

def test_algorithmic_dime_236():
    positions = get_defense_positions("Dime (2-3-6)", ball_x=60)
    assert len(positions) == 11
    roles = [p[2] for p in positions]
    dl_count = len([r for r in roles if r.startswith("DL")])
    lb_count = len([r for r in roles if r.startswith("LB")])
    db_count = len([r for r in roles if r.startswith(("CB", "FS", "SS", "DB"))])
    assert dl_count == 2
    assert lb_count == 3
    assert db_count == 6

def test_unknown_defense_returns_11():
    positions = get_defense_positions("UNKNOWN", ball_x=60)
    assert len(positions) == 11

def test_defense_within_field_bounds():
    for form in ["4-3", "3-4", "Nickel (4-2-5)", "Dime (2-3-6)", "5-2"]:
        positions = get_defense_positions(form, ball_x=60)
        for x, y, role in positions:
            assert 0 <= x <= 120, f"{form} {role} x={x} out of bounds"
            assert 0 <= y <= 53.3, f"{form} {role} y={y} out of bounds"
