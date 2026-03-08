# Player Positions & Motion Trails — Design Doc

**Date:** 2026-03-08
**Goal:** Add 11v11 player position dots and playbook-style motion trails to the Gradio demo replay app.

## Decisions

- **Style:** Static playbook diagrams with motion trail arrows (no animation)
- **Players:** Full 11v11 both sides
- **Movement:** Both offense and defense show trails
- **Rendering:** Matplotlib only, no new dependencies
- **Formation templates:** Hand-craft common formations, generate the rest algorithmically
- **UI changes:** None — same buttons, layout, and callbacks

## Architecture

### New Files

**`formations.py`** — Formation position templates + algorithmic fallback.

- `OFFENSE_FORMATIONS` dict: hand-crafted templates for SHOTGUN, SINGLEBACK, I_FORM, EMPTY, PISTOL. Each is a list of 11 `(x_offset, y, role)` tuples relative to LOS.
- `DEFENSE_FORMATIONS` dict: hand-crafted templates for 4-3, 3-4, Nickel (4-2-5).
- `get_offense_positions(formation, ball_x) -> [(x, y, role)]` — returns absolute field coordinates.
- `get_defense_positions(formation, ball_x) -> [(x, y, role)]` — returns absolute field coordinates. Parses DL-LB-DB counts from formation string for unknown formations.
- JUMBO/WILDCAT use modified SINGLEBACK/SHOTGUN templates.

**`play_motion.py`** — Trail generation from play data.

- `get_motion_trails(play_data, off_positions, def_positions) -> List[(start_xy, end_xy, role, side, is_key_actor)]`
- Pass plays: QB drops back, OL pockets, target WR runs route from `designedPass`, other WRs run clearing routes, RB chips/swings. Rushers push toward QB, coverage drops based on `def_manZone`.
- Run plays: RB trails toward `off_runConcept` direction, OL blocks toward run direction, WRs block downfield. DL pushes gaps, LBs flow, DBs come up.
- Punt/FG: minimal movement.
- Trail length roughly scaled to yards gained.

### Modified Files

**`demo_replay.py`** — `draw_play_state()` updated to:

1. Call `get_offense_positions()` and `get_defense_positions()` for pre-snap dots.
2. Call `get_motion_trails()` for trail arrows.
3. Render motion trails as semi-transparent arrows (blue offense, red defense, alpha 0.4; key actors alpha 0.7).
4. Render player dots as colored circles with role label text.
5. For pass plays, render dashed ball-flight line from QB to catch point.

### Rendering Layers (zorder)

1. Field (existing)
2. Motion trails (new)
3. Ball flight dashed line (new, pass only)
4. Player dots at pre-snap positions (new)
5. LOS / first-down markers (existing)
6. Info labels (existing)

### Visual Style

- Offense dots: blue circles, size ~150
- Defense dots: red circles, size ~150
- Role labels: small white text centered on dots
- Offense trails: blue arrows, alpha 0.4
- Defense trails: red arrows, alpha 0.4
- Key actor trails: thicker, alpha 0.7

## Not In Scope

- Real NFL tracking data integration
- Animated frame-by-frame playback
- Interactive hover/click on players
- Jersey numbers or player names
