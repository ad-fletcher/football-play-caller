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
    ax = fig.axes[0]
    assert len(ax.collections) >= 1
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


import json

def test_all_actual_plays_render():
    """Smoke test: every play in the dataset renders without error."""
    with open("checkpoints/adversarial/phase_play_by_play.json") as f:
        plays = json.load(f)
    for i, play in enumerate(plays):
        prev_yl = plays[i-1]["yardline"] if i > 0 and plays[i-1].get("drive") == play.get("drive") else None
        fig = draw_play_state(play, prev_yardline=prev_yl)
        assert fig is not None, f"Play {i} failed to render"
        plt.close(fig)
