"""
NFL Play-Calling AI — Adversarial Football Drive Replay

Gradio app that replays saved play-by-play data from trained adversarial
offense vs defense LLM models. Shows a football field with ball movement,
formations, and play results animated.

No model inference — pure replay of eval_phases.py output.
"""

import json
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gradio as gr
from formations import get_offense_positions, get_defense_positions
from play_motion import get_motion_trails, get_ball_flight

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "adversarial", "phase_play_by_play.json")
with open(DATA_PATH) as f:
    ALL_PLAYS = json.load(f)

# Group plays by (checkpoint, drive)
def _group_drives(plays):
    drives = {}
    for p in plays:
        key = (p["checkpoint"], p["drive"])
        drives.setdefault(key, []).append(p)
    return drives

ALL_DRIVES = _group_drives(ALL_PLAYS)

# Build checkpoint list and per-checkpoint drive lists
CHECKPOINTS = []
seen = set()
for p in ALL_PLAYS:
    if p["checkpoint"] not in seen:
        CHECKPOINTS.append(p["checkpoint"])
        seen.add(p["checkpoint"])

CHECKPOINT_LABELS = {
    "round_10": "Phase 0 — Early Training (Round 10)",
    "round_30": "Phase 1 — Offense Trains (Round 30)",
    "round_50": "Phase 2 — Defense Trains (Round 50)",
    "round_70": "Phase 3 — Co-Adaptation (Round 70)",
    "final": "Final Models",
}


def get_drives_for_checkpoint(cp):
    drives = {}
    for (c, d), plays in ALL_DRIVES.items():
        if c == cp:
            drives[d] = plays
    return drives


def drive_label(drive_plays):
    n = len(drive_plays)
    last = drive_plays[-1]
    result = last.get("drive_result") or "ongoing"
    total_yds = sum(p["yards"] for p in drive_plays)
    return f"Drive {drive_plays[0]['drive']}: {n} plays \u2192 {result} ({total_yds:.0f} yds)"


# ---------------------------------------------------------------------------
# Football field drawing
# ---------------------------------------------------------------------------
def draw_field(ax):
    """Draw a football field on the given axes."""
    ax.set_facecolor("#2e7d32")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    # Yard lines every 5 yards
    for x in range(10, 111, 5):
        lw = 2 if x % 10 == 0 else 0.5
        ax.axvline(x, color="white", linewidth=lw, alpha=0.7)

    # End zones
    ax.axvspan(0, 10, color="#1b5e20", alpha=0.5)
    ax.axvspan(110, 120, color="#1b5e20", alpha=0.5)

    # End zone labels
    ax.text(5, 26.65, "END\nZONE", fontsize=14, color="white",
            ha="center", va="center", fontweight="bold", alpha=0.6, rotation=90)
    ax.text(115, 26.65, "END\nZONE", fontsize=14, color="white",
            ha="center", va="center", fontweight="bold", alpha=0.6, rotation=-90)

    # Yard numbers
    for yard in range(10, 100, 10):
        num = yard if yard <= 50 else 100 - yard
        ax.text(yard + 10, 5, str(num), fontsize=16, color="white",
                ha="center", va="center", fontweight="bold", alpha=0.5)
        ax.text(yard + 10, 48.3, str(num), fontsize=16, color="white",
                ha="center", va="center", fontweight="bold", alpha=0.5, rotation=180)

    # Hash marks
    for x in range(10, 111):
        ax.plot([x, x], [0.5, 1.5], color="white", linewidth=0.5)
        ax.plot([x, x], [51.8, 52.8], color="white", linewidth=0.5)
        ax.plot([x, x], [23.36, 23.86], color="white", linewidth=0.5)
        ax.plot([x, x], [29.44, 29.94], color="white", linewidth=0.5)

    # Field outline
    ax.plot([10, 10], [0, 53.3], color="white", linewidth=2)
    ax.plot([110, 110], [0, 53.3], color="white", linewidth=2)
    ax.plot([0, 120], [0, 0], color="white", linewidth=2)
    ax.plot([0, 120], [53.3, 53.3], color="white", linewidth=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def draw_play_state(play, prev_yardline=None, is_drive_end=False):
    """Draw the field with the current play state overlaid."""
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    draw_field(ax)

    yardline = play["yardline"]
    yards_to_go = play["yardsToGo"]
    ball_x = yardline + 10  # field coords: 0-10 left endzone, 10-110 field, 110-120 right endzone

    # Line of scrimmage (yellow dashed)
    ax.axvline(ball_x, color="#FFD700", linewidth=3, alpha=0.85, linestyle="--", zorder=5)
    ax.text(ball_x, 53.3 + 1.2, "LOS", fontsize=8, color="#FFD700",
            ha="center", va="bottom", fontweight="bold", clip_on=False)

    # First-down marker (blue line)
    first_down_x = ball_x + yards_to_go
    if first_down_x <= 110:
        ax.axvline(first_down_x, color="#2196F3", linewidth=3, alpha=0.85, zorder=5)
        ax.text(first_down_x, 53.3 + 1.2, "1st", fontsize=8, color="#2196F3",
                ha="center", va="bottom", fontweight="bold", clip_on=False)

    # ---- Player positions and motion trails ----
    off_pos = get_offense_positions(play.get("off_formation", "SHOTGUN"), ball_x)
    def_pos = get_defense_positions(play.get("def_formation", "4-3"), ball_x)
    trails = get_motion_trails(play, off_pos, def_pos)

    # Draw motion trails
    for start, end, role, side, is_key in trails:
        color = "#64B5F6" if side == "offense" else "#EF5350"
        lw = 2.5 if is_key else 1.5
        alpha = 0.7 if is_key else 0.35
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw),
                    alpha=alpha, zorder=3)

    # Draw ball flight (dashed line for pass plays)
    flight = get_ball_flight(play, trails)
    if flight:
        (qx, qy), (wx, wy) = flight
        ax.plot([qx, wx], [qy, wy], '--', color='white', lw=1.5, alpha=0.6, zorder=4)

    # Draw player dots (offense = blue, defense = red)
    off_xs = [p[0] for p in off_pos]
    off_ys = [p[1] for p in off_pos]
    def_xs = [p[0] for p in def_pos]
    def_ys = [p[1] for p in def_pos]
    ax.scatter(off_xs, off_ys, c="#1565C0", s=150, zorder=6, edgecolors="white", linewidths=1)
    ax.scatter(def_xs, def_ys, c="#B71C1C", s=150, zorder=6, edgecolors="white", linewidths=1)

    # Draw role labels
    for x, y, role in off_pos:
        ax.text(x, y, role[:2], fontsize=4.5, color="white", ha="center", va="center",
                fontweight="bold", zorder=7)
    for x, y, role in def_pos:
        ax.text(x, y, role[:2], fontsize=4.5, color="white", ha="center", va="center",
                fontweight="bold", zorder=7)

    # Offense formation label (blue, left of ball)
    off_form = play.get("off_formation", "")
    ax.text(ball_x - 6, 50, off_form, fontsize=11, color="#64B5F6",
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#64B5F6", alpha=0.85),
            zorder=8)

    # Defense formation label (red, right of ball)
    def_form = play.get("def_formation", "")
    ax.text(ball_x + 6, 50, def_form, fontsize=11, color="#EF5350",
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a2e", edgecolor="#EF5350", alpha=0.85),
            zorder=8)

    # Show yards gained arrow if play has result
    yards = play.get("yards", 0)
    result = play.get("result", "")
    drive_result = play.get("drive_result")

    if result and prev_yardline is not None:
        prev_x = prev_yardline + 10
        new_x = min(max(ball_x + yards, 10), 110)

        if drive_result == "touchdown":
            # Gold arrow to endzone
            ax.annotate("", xy=(110, 26.65), xytext=(prev_x, 26.65),
                        arrowprops=dict(arrowstyle="-|>", color="#FFD700", lw=4), zorder=7)
            ax.scatter(112, 26.65, c="#FFD700", s=500, marker="*", zorder=10)
            ax.text(115, 15, "TOUCHDOWN!", fontsize=16, color="#FFD700",
                    ha="center", va="center", fontweight="bold", rotation=90,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#1b5e20", edgecolor="#FFD700", alpha=0.9),
                    zorder=10)
        elif drive_result in ("interception", "fumble_lost"):
            color = "#FF1744"
            label = "INTERCEPTION!" if drive_result == "interception" else "FUMBLE!"
            ax.scatter(ball_x, 26.65, c=color, s=400, marker="X", zorder=11, linewidths=2)
            ax.text(ball_x, 15, label, fontsize=14, color=color,
                    ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor=color, alpha=0.9),
                    zorder=10)
        elif drive_result in ("field_goal_made", "field_goal_missed"):
            if drive_result == "field_goal_made":
                ax.text(112, 26.65, "FG\nGOOD", fontsize=14, color="#FFD700",
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1b5e20", edgecolor="#FFD700", alpha=0.9),
                        zorder=10)
            else:
                ax.text(112, 26.65, "FG\nMISS", fontsize=14, color="#FF1744",
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", edgecolor="#FF1744", alpha=0.9),
                        zorder=10)
        elif yards != 0:
            arrow_color = "#4CAF50" if yards > 0 else "#FF5252"
            ax.annotate("", xy=(new_x, 26.65), xytext=(prev_x, 26.65),
                        arrowprops=dict(arrowstyle="-|>", color=arrow_color, lw=3), zorder=7)
            sign = "+" if yards > 0 else ""
            ax.text((prev_x + new_x) / 2, 20, f"{sign}{yards:.0f} yds",
                    fontsize=12, color=arrow_color, ha="center", va="center",
                    fontweight="bold", zorder=8)

    # Title bar
    down_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(play["down"], f"{play['down']}th")
    title = f"{down_str} & {play['yardsToGo']}  |  Q{play['quarter']}  {play['clock'] // 60}:{play['clock'] % 60:02d}  |  Ball on {yardline}"
    ax.set_title(title, fontsize=14, color="white", fontweight="bold", pad=12)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Format info panels
# ---------------------------------------------------------------------------
def format_game_state(play):
    down_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(play["down"], f"{play['down']}th")
    minutes = play["clock"] // 60
    seconds = play["clock"] % 60
    yardline = play["yardline"]

    # Field position description
    if yardline <= 50:
        field_pos = f"Own {yardline}"
    else:
        field_pos = f"Opp {100 - yardline}"

    red_zone = " \U0001f7e5 RED ZONE" if yardline >= 80 else ""

    return f"""### Game State
| | |
|---|---|
| **Down & Distance** | {down_str} & {play['yardsToGo']} |
| **Quarter** | Q{play['quarter']} |
| **Game Clock** | {minutes}:{seconds:02d} |
| **Field Position** | {field_pos}{red_zone} |
| **Score Diff** | {play['score_diff']:+d} |
"""


def format_play_call(play):
    off_play = play["off_playType"]
    off_detail = ""
    if off_play in ("pass", "play_action"):
        off_detail = f" {play['off_designedPass']}"
    elif off_play == "run":
        off_detail = f" {play['off_runConcept']}"

    result = play.get("result", "")
    yards = play.get("yards", 0)
    drive_result = play.get("drive_result")

    result_text = ""
    if result:
        sign = "+" if yards > 0 else ""
        result_text = f"**Result:** {result} {sign}{yards:.0f} yds"
        if drive_result:
            result_text += f"\n\n**Drive Result: {drive_result.upper()}**"

    off_reward = play.get("offense_reward", 0)
    def_reward = play.get("defense_reward", 0)

    violations_text = ""
    if play.get("off_violations"):
        violations_text += f"\n\n\u26a0\ufe0f Off violations: {', '.join(play['off_violations'])}"
    if play.get("def_violations"):
        violations_text += f"\n\u26a0\ufe0f Def violations: {', '.join(play['def_violations'])}"

    return f"""### Play Call
**OFFENSE:** {play['off_formation']} {off_play}{off_detail}
{f"Receiver alignment: {play['off_receiverAlignment']}" if play.get('off_receiverAlignment', 'none') != 'none' else ''}

**DEFENSE:** {play['def_formation']} {play['def_manZone']} {play['def_coverage']} ({play['def_rushers']} rushers)

---
{result_text}

| Offense Reward | Defense Reward |
|---|---|
| {off_reward:+.1f} | {def_reward:+.1f} |
{violations_text}"""


def format_drive_log(drive_plays, current_play_idx):
    """Build the scrollable drive log up to current play."""
    lines = ["### Drive Log\n"]
    for i, p in enumerate(drive_plays[:current_play_idx + 1]):
        down_str = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(p["down"], f"{p['down']}th")
        yardline = p["yardline"]
        if yardline <= 50:
            field_pos = f"own {yardline}"
        else:
            field_pos = f"opp {100 - yardline}"

        play_type = p["off_playType"]
        yards = p.get("yards", 0)
        result = p.get("result", "")
        sign = "+" if yards > 0 else ""

        marker = "\u25b6 " if i == current_play_idx else "  "
        line = f"{marker}**Play {p['play']}:** {down_str}&{p['yardsToGo']} {field_pos} | {p['off_formation']} {play_type} \u2192 {result} {sign}{yards:.0f} yds"

        drive_result = p.get("drive_result")
        if drive_result and i == current_play_idx:
            line += f"\n\n**Drive Result: {drive_result.upper()}**"

        lines.append(line)

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------
def get_checkpoint_choices():
    return [(CHECKPOINT_LABELS.get(cp, cp), cp) for cp in CHECKPOINTS]


def get_drive_choices(checkpoint):
    drives = get_drives_for_checkpoint(checkpoint)
    choices = []
    for d_num in sorted(drives.keys()):
        label = drive_label(drives[d_num])
        choices.append((label, d_num))
    return choices


def on_checkpoint_change(checkpoint):
    choices = get_drive_choices(checkpoint)
    first_drive = choices[0][1] if choices else None
    return (
        gr.update(choices=choices, value=first_drive),
        *on_drive_change(checkpoint, first_drive)
    )


def on_drive_change(checkpoint, drive_num):
    if drive_num is None:
        empty_fig, ax = plt.subplots(figsize=(14, 6))
        draw_field(ax)
        plt.tight_layout()
        return empty_fig, "", "", "", {"checkpoint": checkpoint, "drive": drive_num, "play_idx": 0}

    drives = get_drives_for_checkpoint(checkpoint)
    drive_plays = drives.get(drive_num, [])
    if not drive_plays:
        empty_fig, ax = plt.subplots(figsize=(14, 6))
        draw_field(ax)
        plt.tight_layout()
        return empty_fig, "", "", "", {"checkpoint": checkpoint, "drive": drive_num, "play_idx": 0}

    play = drive_plays[0]
    fig = draw_play_state(play, prev_yardline=None)
    state_md = format_game_state(play)
    call_md = format_play_call(play)
    log_md = format_drive_log(drive_plays, 0)

    return fig, state_md, call_md, log_md, {"checkpoint": checkpoint, "drive": drive_num, "play_idx": 0}


def step_play(state, direction):
    """Advance or go back one play. direction: +1 or -1."""
    cp = state["checkpoint"]
    d_num = state["drive"]
    idx = state["play_idx"]

    drives = get_drives_for_checkpoint(cp)
    drive_plays = drives.get(d_num, [])
    if not drive_plays:
        return plt.figure(), "", "", "", state

    new_idx = max(0, min(idx + direction, len(drive_plays) - 1))
    state["play_idx"] = new_idx

    play = drive_plays[new_idx]
    prev_yardline = drive_plays[new_idx - 1]["yardline"] if new_idx > 0 else None
    fig = draw_play_state(play, prev_yardline=prev_yardline)
    state_md = format_game_state(play)
    call_md = format_play_call(play)
    log_md = format_drive_log(drive_plays, new_idx)

    return fig, state_md, call_md, log_md, state


def on_next(state):
    return step_play(state, +1)


def on_prev(state):
    return step_play(state, -1)


def on_auto_play(state):
    """Generator that yields each play with a delay for auto-play."""
    import time
    cp = state["checkpoint"]
    d_num = state["drive"]
    idx = state["play_idx"]

    drives = get_drives_for_checkpoint(cp)
    drive_plays = drives.get(d_num, [])
    if not drive_plays:
        return

    for i in range(idx, len(drive_plays)):
        state["play_idx"] = i
        play = drive_plays[i]
        prev_yardline = drive_plays[i - 1]["yardline"] if i > 0 else None
        fig = draw_play_state(play, prev_yardline=prev_yardline)
        state_md = format_game_state(play)
        call_md = format_play_call(play)
        log_md = format_drive_log(drive_plays, i)
        yield fig, state_md, call_md, log_md, state
        plt.close(fig)
        if i < len(drive_plays) - 1:
            time.sleep(1.5)


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------
def build_app():
    # Compute overall stats for header
    total_plays = len(ALL_PLAYS)
    total_drives = len(ALL_DRIVES)
    n_checkpoints = len(CHECKPOINTS)

    with gr.Blocks(title="NFL Play-Calling AI") as app:
        gr.Markdown("# NFL Play-Calling AI \u2014 Adversarial Football", elem_classes=["main-title"])
        gr.Markdown(
            f"Trained LLM offense vs defense models \u2014 {total_plays} plays across {total_drives} drives, {n_checkpoints} training checkpoints",
            elem_classes=["subtitle"]
        )

        state = gr.State({"checkpoint": CHECKPOINTS[0], "drive": 1, "play_idx": 0})

        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                choices=get_checkpoint_choices(),
                value=CHECKPOINTS[0],
                label="Training Phase",
                scale=2,
            )
            drive_dd = gr.Dropdown(
                choices=get_drive_choices(CHECKPOINTS[0]),
                value=1,
                label="Drive",
                scale=3,
            )

        with gr.Row():
            prev_btn = gr.Button("\u25c0 Prev Play", scale=1)
            next_btn = gr.Button("Next Play \u25b6", scale=1, variant="primary")
            auto_btn = gr.Button("Auto-Play \u25b6\u25b6", scale=1, variant="secondary")

        field_plot = gr.Plot(label="", elem_classes=["field-plot"])

        with gr.Row(equal_height=True):
            game_state_md = gr.Markdown()
            play_call_md = gr.Markdown()

        drive_log_md = gr.Markdown()

        # Wire up events
        outputs = [field_plot, game_state_md, play_call_md, drive_log_md, state]

        checkpoint_dd.change(
            on_checkpoint_change, inputs=[checkpoint_dd],
            outputs=[drive_dd] + outputs,
        )

        drive_dd.change(
            on_drive_change, inputs=[checkpoint_dd, drive_dd],
            outputs=outputs,
        )

        next_btn.click(on_next, inputs=[state], outputs=outputs)
        prev_btn.click(on_prev, inputs=[state], outputs=outputs)
        auto_btn.click(on_auto_play, inputs=[state], outputs=outputs)

        # Load initial state
        app.load(
            on_drive_change, inputs=[checkpoint_dd, drive_dd],
            outputs=outputs,
        )

    return app


app = build_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.green,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #888; margin-top: 0; font-size: 0.95em; }
        .field-plot { border: 2px solid #333; border-radius: 8px; }
        footer { display: none !important; }
        """,
    )
