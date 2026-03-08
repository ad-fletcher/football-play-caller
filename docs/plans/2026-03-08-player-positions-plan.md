# Player Positions & Motion Trails — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 11v11 player position dots and playbook-style motion trails to the Gradio demo replay app.

**Architecture:** Two new modules (`formations.py` for position templates, `play_motion.py` for trail generation) integrated into the existing `draw_play_state()` function in `demo_replay.py`. Matplotlib-only, no new dependencies, no UI changes.

**Tech Stack:** Python, matplotlib (existing)

---

### Task 1: Offense Formation Templates

**Files:**
- Create: `formations.py`
- Test: `tests/test_formations.py`

**Step 1: Write the failing test**

```python
# tests/test_formations.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_formations.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'formations'`

**Step 3: Write the implementation**

Create `formations.py` with:

- `OFFENSE_FORMATIONS` dict with hand-crafted templates for SHOTGUN, SINGLEBACK, I_FORM, EMPTY, PISTOL. Each value is a list of 11 `(x_offset, y, role)` tuples. `x_offset` is relative to ball_x (negative = behind LOS). `y` is absolute field y-coordinate (0-53.3, center is 26.65).
  - SHOTGUN: QB at -5, RB offset at -4, 5 OL at 0, TE at 0 on edge, 3 WRs split wide
  - SINGLEBACK: QB under center at -1, RB at -6, 5 OL at 0, TE at 0, 3 WRs
  - I_FORM: QB under center at -1, FB at -4, RB at -6, 5 OL at 0, TE at 0, 2 WRs
  - EMPTY: QB at -5, no RB near QB, 5 OL at 0, 5 receivers spread
  - PISTOL: QB at -3, RB at -5 directly behind, 5 OL at 0, TE, 3 WRs
- JUMBO maps to modified SINGLEBACK (extra TE replaces a WR)
- WILDCAT maps to modified SHOTGUN (RB takes snap, QB out wide)
- `get_offense_positions(formation: str, ball_x: float) -> list[tuple[float, float, str]]` — looks up template, applies ball_x offset, returns absolute coordinates. Unknown formations fall back to SHOTGUN.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_formations.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add formations.py tests/test_formations.py
git commit -m "feat: add offense formation position templates"
```

---

### Task 2: Defense Formation Templates

**Files:**
- Modify: `formations.py`
- Modify: `tests/test_formations.py`

**Step 1: Write the failing test**

```python
# append to tests/test_formations.py
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_formations.py::test_43_returns_11_players -v`
Expected: FAIL — `ImportError: cannot import name 'get_defense_positions'`

**Step 3: Write the implementation**

Add to `formations.py`:

- `DEFENSE_FORMATIONS` dict with hand-crafted templates for 4-3, 3-4, Nickel (4-2-5). Each has 11 `(x_offset, y, role)` tuples. `x_offset` is positive (ahead of LOS).
  - 4-3: 4 DL at +1 spread across, 3 LB at +4, 2 CB wide at +5, FS at +12, SS at +8
  - 3-4: 3 DL at +1, 4 LB at +3-4, 2 CB wide at +5, FS at +12, SS at +8
  - Nickel (4-2-5): 4 DL at +1, 2 LB at +4, 3 CB at +5, FS at +12, SS at +8
- `_parse_defense_counts(formation: str) -> (int, int, int)` — extracts DL/LB/DB counts from strings like "Dime (2-3-6)" or "5-2". Returns (dl, lb, db) where db = 11 - dl - lb.
- `_generate_defense_positions(dl, lb, db) -> list[tuple]` — algorithmically spaces players: DL at +1 evenly across center, LB at +4 evenly, DBs split into CB (wide, +5), safeties (deep, +10-12).
- `get_defense_positions(formation: str, ball_x: float) -> list[tuple[float, float, str]]` — looks up template or generates algorithmically. Applies ball_x offset.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_formations.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add formations.py tests/test_formations.py
git commit -m "feat: add defense formation templates with algorithmic fallback"
```

---

### Task 3: Play Motion Trails

**Files:**
- Create: `play_motion.py`
- Create: `tests/test_play_motion.py`

**Step 1: Write the failing test**

```python
# tests/test_play_motion.py
from formations import get_offense_positions, get_defense_positions
from play_motion import get_motion_trails

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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_play_motion.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'play_motion'`

**Step 3: Write the implementation**

Create `play_motion.py` with:

- `get_motion_trails(play, off_positions, def_positions) -> list[tuple]`
  - Returns list of `((start_x, start_y), (end_x, end_y), role, side, is_key_actor)`
  - `side` is "offense" or "defense"
  - `is_key_actor` is True for ball carrier (run), target WR + QB (pass), rushers

- Internal logic:
  - **`_pass_offense_trails(play, positions)`**: QB drops back 5-7 yds. Parse `designedPass` (e.g. "short_middle") to get depth (short=8, mid=15, deep=25) and direction (left/middle/right → y offset). Target WR (closest to route direction) gets route trail, marked as key actor. Other WRs get clearing routes (opposite direction, shorter). RB shifts to flat or stays for pass pro. OL shifts outward 1-2 yds (pocket).
  - **`_pass_defense_trails(play, positions)`**: First N DL (where N = `def_rushers`) rush toward QB (move x toward LOS by 5-6 yds). If `def_rushers` > DL count, some LBs also rush. Zone: LBs/DBs drop back into zones (deep third, hook/curl areas). Man: DBs trail toward nearest WR endpoint.
  - **`_run_offense_trails(play, positions)`**: Parse `off_runConcept` for direction. RB trails through gap or around edge (key actor). OL all shift toward run direction (2-3 yds). WRs move downfield to block (3-4 yds forward).
  - **`_run_defense_trails(play, positions)`**: DL push into gaps (1-2 yds toward LOS). LBs flow toward run direction. DBs come up toward LOS (2-3 yds).
  - **`_fg_punt_trails(play, positions)`**: Minimal movement — everyone shifts 1 yd forward/back.
  - Trail lengths scaled by `abs(yards)` with a multiplier (capped at reasonable distances so trails don't go off-field).

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_play_motion.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add play_motion.py tests/test_play_motion.py
git commit -m "feat: add play motion trail generation"
```

---

### Task 4: Ball Flight Line (Pass Plays)

**Files:**
- Modify: `play_motion.py`
- Modify: `tests/test_play_motion.py`

**Step 1: Write the failing test**

```python
# append to tests/test_play_motion.py
from play_motion import get_ball_flight

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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_play_motion.py::test_pass_play_has_ball_flight -v`
Expected: FAIL — `ImportError: cannot import name 'get_ball_flight'`

**Step 3: Write the implementation**

Add to `play_motion.py`:

- `get_ball_flight(play, trails) -> tuple | None` — For pass/play_action plays, returns `((qb_end_x, qb_end_y), (wr_end_x, wr_end_y))` by finding QB and key-actor WR trail endpoints. Returns None for non-pass plays.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_play_motion.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add play_motion.py tests/test_play_motion.py
git commit -m "feat: add ball flight line for pass plays"
```

---

### Task 5: Integrate into demo_replay.py

**Files:**
- Modify: `demo_replay.py:123-215` (the `draw_play_state` function)

**Step 1: Write a smoke test**

```python
# tests/test_demo_rendering.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from demo_replay import draw_play_state

def test_draw_play_state_with_players():
    play = {
        "yardline": 50, "yardsToGo": 10, "down": 1, "quarter": 1,
        "clock": 900, "score_diff": 0,
        "off_formation": "SHOTGUN", "off_playType": "pass",
        "off_designedPass": "short_middle", "off_runConcept": "none",
        "off_receiverAlignment": "3x1",
        "def_formation": "4-3", "def_manZone": "Zone",
        "def_coverage": "Cover-3", "def_rushers": 4,
        "result": "normal", "yards": 6.0, "drive_result": None,
        "offense_reward": 0.6, "defense_reward": -0.6,
        "off_violations": [], "def_violations": [],
    }
    fig = draw_play_state(play, prev_yardline=50)
    assert fig is not None
    # Check that there are more artists than a bare field
    ax = fig.axes[0]
    # Player dots (scatter) + trails (annotations) should add content
    assert len(ax.collections) >= 1  # at least the player scatter plots
    plt.close(fig)

def test_draw_run_play():
    play = {
        "yardline": 30, "yardsToGo": 7, "down": 2, "quarter": 2,
        "clock": 450, "score_diff": -7,
        "off_formation": "PISTOL", "off_playType": "run",
        "off_designedPass": "none", "off_runConcept": "OUTSIDE ZONE",
        "off_receiverAlignment": "none",
        "def_formation": "4-3", "def_manZone": "Zone",
        "def_coverage": "Cover-3", "def_rushers": 4,
        "result": "normal", "yards": 4.0, "drive_result": None,
        "offense_reward": 0.4, "defense_reward": -0.4,
        "off_violations": [], "def_violations": [],
    }
    fig = draw_play_state(play, prev_yardline=30)
    assert fig is not None
    plt.close(fig)
```

**Step 2: Run test to verify current behavior (baseline)**

Run: `python -m pytest tests/test_demo_rendering.py -v`
Expected: PASS (existing function works, just no player dots yet — but test is loose enough to pass)

**Step 3: Integrate formations and trails into draw_play_state**

Modify `draw_play_state()` in `demo_replay.py` to:

1. Add imports at top: `from formations import get_offense_positions, get_defense_positions` and `from play_motion import get_motion_trails, get_ball_flight`
2. After drawing the field and LOS lines, before the yards-gained arrow section:
   - Call `get_offense_positions(play["off_formation"], ball_x)` and `get_defense_positions(play["def_formation"], ball_x)`
   - Call `get_motion_trails(play, off_pos, def_pos)`
   - Draw motion trails: for each trail, `ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha), zorder=3)`. Blue for offense, red for defense. Key actors: lw=2.5, alpha=0.7. Others: lw=1.5, alpha=0.35.
   - Call `get_ball_flight(play, trails)` — if not None, draw dashed white line from QB to WR endpoint: `ax.plot([qb_x, wr_x], [qb_y, wr_y], '--', color='white', lw=1.5, alpha=0.6, zorder=4)`
   - Draw player dots: `ax.scatter(xs, ys, c=color, s=150, zorder=6, edgecolors='white', linewidths=1)` — blue for offense, red for defense
   - Draw role labels: `ax.text(x, y, role, fontsize=5, color='white', ha='center', va='center', fontweight='bold', zorder=7)`
3. Remove the old football diamond marker (replaced by QB position dot)
4. Move formation text labels to top/bottom margins to avoid overlap with player dots

**Step 4: Run tests**

Run: `python -m pytest tests/test_demo_rendering.py tests/test_formations.py tests/test_play_motion.py -v`
Expected: All PASS

**Step 5: Visual check**

Run: `python demo_replay.py` and verify in browser at localhost:7860

**Step 6: Commit**

```bash
git add demo_replay.py tests/test_demo_rendering.py
git commit -m "feat: integrate player positions and motion trails into replay"
```

---

### Task 6: Polish and Edge Cases

**Files:**
- Modify: `formations.py`
- Modify: `play_motion.py`
- Modify: `demo_replay.py`

**Step 1: Handle edge-of-field positions**

Ensure all player positions and trail endpoints are clamped to valid field coordinates (x: 0-120, y: 0-53.3). Add clamping in `get_offense_positions`, `get_defense_positions`, and `get_motion_trails`.

**Step 2: Handle field_goal and punt play types**

Add minimal formation handling — FG/punt formations show a tight cluster, minimal trails.

**Step 3: Test with all plays in the actual data**

```python
# append to tests/test_demo_rendering.py
import json

def test_all_actual_plays_render():
    """Smoke test: every play in the dataset renders without error."""
    with open("checkpoints/adversarial/phase_play_by_play.json") as f:
        plays = json.load(f)
    for i, play in enumerate(plays):
        prev_yl = plays[i-1]["yardline"] if i > 0 and plays[i-1].get("drive") == play.get("drive") else None
        fig = draw_play_state(play, prev_yardline=prev_yl)
        assert fig is not None
        plt.close(fig)
```

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add formations.py play_motion.py demo_replay.py tests/test_demo_rendering.py
git commit -m "feat: polish player rendering, handle edge cases"
```
