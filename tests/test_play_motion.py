from formations import get_offense_positions, get_defense_positions
from play_motion import get_motion_trails, get_ball_flight

def _make_play(play_type="pass", designed_pass="short_middle", run_concept="none",
               off_formation="SHOTGUN", def_formation="4-3",
               def_manZone="Zone", def_rushers=4, yards=5.0):
    return {
        "off_formation": off_formation,
        "off_playType": play_type,
        "off_designedPass": designed_pass,
        "off_runConcept": run_concept,
        "def_formation": def_formation,
        "def_manZone": def_manZone,
        "def_coverage": "Cover-3",
        "def_rushers": def_rushers,
        "yards": yards,
        "result": "normal",
        "drive_result": None,
        "yardline": 50,
    }

def test_pass_play_returns_22_trails():
    play = _make_play(play_type="pass")
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    assert len(trails) == 22

def test_trail_tuple_structure():
    play = _make_play()
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    for trail in trails:
        start, end, role, side, is_key = trail
        assert len(start) == 2  # (x, y)
        assert len(end) == 2
        assert isinstance(role, str)
        assert side in ("offense", "defense")
        assert isinstance(is_key, bool)

def test_run_play_rb_moves_forward():
    play = _make_play(play_type="run", designed_pass="none",
                      run_concept="INSIDE ZONE", yards=4.0)
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    rb_trails = [t for t in trails if t[2] == "RB" and t[3] == "offense"]
    assert len(rb_trails) == 1
    assert rb_trails[0][1][0] > rb_trails[0][0][0]  # RB end x > start x
    assert rb_trails[0][4] is True  # RB is key actor on run

def test_pass_play_qb_drops_back():
    play = _make_play(play_type="pass")
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    qb_trails = [t for t in trails if t[2] == "QB"]
    assert len(qb_trails) == 1
    assert qb_trails[0][1][0] < qb_trails[0][0][0]  # QB drops back

def test_pass_rushers_move_toward_qb():
    play = _make_play(play_type="pass", def_rushers=4)
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    # Rushers should move toward LOS (smaller x)
    dl_trails = [t for t in trails if t[2].startswith("DL") and t[3] == "defense"]
    for t in dl_trails:
        assert t[1][0] < t[0][0], f"{t[2]} should rush toward LOS"

def test_pass_play_has_ball_flight():
    play = _make_play(play_type="pass", designed_pass="short_middle", yards=8.0)
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    flight = get_ball_flight(play, trails)
    assert flight is not None
    start, end = flight
    assert len(start) == 2
    assert len(end) == 2

def test_run_play_no_ball_flight():
    play = _make_play(play_type="run", run_concept="INSIDE ZONE")
    ball_x = play["yardline"] + 10
    off_pos = get_offense_positions(play["off_formation"], ball_x)
    def_pos = get_defense_positions(play["def_formation"], ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)
    flight = get_ball_flight(play, trails)
    assert flight is None
