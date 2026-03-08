"""
Evaluate trained LoRA models — generate rich play-by-play data for analysis & demo video.

Usage:
  python eval_drives.py                          # 100 drives with final models
  python eval_drives.py --n_drives 200           # more drives
  python eval_drives.py --checkpoint offense_lora_r40 defense_lora_r40  # specific checkpoint

Output: checkpoints/adversarial/eval_play_by_play.json
"""

import json
import os
import argparse
import time
from collections import Counter

import torch
from unsloth import FastLanguageModel

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
MAX_COMPLETION_LENGTH = 80
TEMPERATURE = 0.7
OUTPUT_DIR = "checkpoints/adversarial"


def _get_stop_token_ids(tokenizer):
    stop_ids = []
    for token in ["}", "}\n", "} ", "}\n\n"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[-1])
    return list(set(stop_ids)) if stop_ids else None


def _truncate_at_json_end(text: str) -> str:
    brace_end = text.find("}")
    if brace_end != -1:
        return text[:brace_end + 1]
    return text


def run_eval(n_drives, off_checkpoint, def_checkpoint):
    from football_env.server.environment import FootballDriveEnvironment
    from football_env.models import GameAction
    from football_env.prompts import (
        format_offense_obs, format_defense_obs,
        parse_offense_response, parse_defense_response,
    )
    from football_env.validation import validate_offense, validate_defense

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load offense model
    print(f"Loading offense model from {off_checkpoint}...")
    off_model, off_tokenizer = FastLanguageModel.from_pretrained(
        model_name=off_checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(off_model)

    # Load defense model
    print(f"Loading defense model from {def_checkpoint}...")
    def_model, def_tokenizer = FastLanguageModel.from_pretrained(
        model_name=def_checkpoint,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(def_model)

    off_stop_ids = _get_stop_token_ids(off_tokenizer)
    def_stop_ids = _get_stop_token_ids(def_tokenizer)

    env = FootballDriveEnvironment()
    all_plays = []
    drive_summaries = []

    print(f"\nRunning {n_drives} drives...\n")
    start_time = time.time()

    for drive_num in range(1, n_drives + 1):
        obs = env.reset()
        drive_plays = []
        play_num = 0

        # Drive start state
        drive_start = {
            "yardline": obs.absoluteYardlineNumber,
            "quarter": obs.quarter,
            "clock": obs.gameClock_seconds,
            "score_diff": obs.score_diff,
        }

        while not obs.done:
            # Pre-play state
            pre_state = {
                "down": obs.down,
                "yardsToGo": obs.yardsToGo,
                "yardline": obs.absoluteYardlineNumber,
                "quarter": obs.quarter,
                "clock": obs.gameClock_seconds,
                "score_diff": obs.score_diff,
            }

            # Generate offense
            off_messages = format_offense_obs(obs)
            off_text = off_tokenizer.apply_chat_template(off_messages, tokenize=False, add_generation_prompt=True)
            off_ids = off_tokenizer(off_text, return_tensors="pt").to(device)
            gen_kwargs = dict(max_new_tokens=MAX_COMPLETION_LENGTH, temperature=TEMPERATURE, do_sample=True)
            if off_stop_ids:
                gen_kwargs["eos_token_id"] = off_stop_ids
            with torch.no_grad():
                off_out = off_model.generate(**off_ids, **gen_kwargs)
            off_response_ids = off_out[0][off_ids["input_ids"].shape[1]:]
            off_response_text = _truncate_at_json_end(off_tokenizer.decode(off_response_ids, skip_special_tokens=True))

            # Generate defense
            def_messages = format_defense_obs(obs)
            def_text = def_tokenizer.apply_chat_template(def_messages, tokenize=False, add_generation_prompt=True)
            def_ids = def_tokenizer(def_text, return_tensors="pt").to(device)
            gen_kwargs = dict(max_new_tokens=MAX_COMPLETION_LENGTH, temperature=TEMPERATURE, do_sample=True)
            if def_stop_ids:
                gen_kwargs["eos_token_id"] = def_stop_ids
            with torch.no_grad():
                def_out = def_model.generate(**def_ids, **gen_kwargs)
            def_response_ids = def_out[0][def_ids["input_ids"].shape[1]:]
            def_response_text = _truncate_at_json_end(def_tokenizer.decode(def_response_ids, skip_special_tokens=True))

            # Parse actions
            off_action = parse_offense_response(off_response_text)
            def_action = parse_defense_response(def_response_text)

            # Validate (for logging contradiction info)
            off_penalty, off_violations = validate_offense(off_action, pre_state["down"])
            def_penalty, def_violations = validate_defense(def_action)

            # Step
            composite = GameAction(offense=off_action, defense=def_action)
            obs = env.step(composite)

            play_num += 1
            play_record = {
                "drive": drive_num,
                "play": play_num,
                # Pre-play game state
                **pre_state,
                # Offense action (all fields)
                "off_formation": off_action.offenseFormation,
                "off_playType": off_action.playType,
                "off_designedPass": off_action.designedPass,
                "off_receiverAlignment": off_action.receiverAlignment,
                "off_runConcept": off_action.pff_runConceptPrimary,
                "off_raw_response": off_response_text,
                "off_violations": off_violations,
                "off_penalty": off_penalty,
                # Defense action (all fields)
                "def_formation": def_action.defFormation,
                "def_manZone": def_action.pff_manZone,
                "def_coverage": def_action.pff_passCoverage,
                "def_rushers": def_action.passRushers,
                "def_raw_response": def_response_text,
                "def_violations": def_violations,
                "def_penalty": def_penalty,
                # Outcome
                "result": obs.last_play_result,
                "yards": obs.last_play_yards,
                "offense_reward": obs.offense_reward,
                "defense_reward": obs.defense_reward,
                # Post-play state
                "post_yardline": obs.absoluteYardlineNumber,
                "post_down": obs.down,
                "post_yardsToGo": obs.yardsToGo,
                "drive_result": obs.drive_result if obs.done else None,
            }
            drive_plays.append(play_record)
            all_plays.append(play_record)

        # Drive summary
        total_yards = sum(p["yards"] for p in drive_plays)
        total_off_reward = sum(p["offense_reward"] for p in drive_plays)
        total_def_reward = sum(p["defense_reward"] for p in drive_plays)
        play_types = Counter(p["off_playType"] for p in drive_plays)
        formations_off = Counter(p["off_formation"] for p in drive_plays)
        formations_def = Counter(p["def_formation"] for p in drive_plays)
        coverages = Counter(p["def_coverage"] for p in drive_plays)

        summary = {
            "drive": drive_num,
            "num_plays": len(drive_plays),
            "start_yardline": drive_start["yardline"],
            "quarter": drive_start["quarter"],
            "score_diff": drive_start["score_diff"],
            "drive_result": obs.drive_result,
            "total_yards": total_yards,
            "total_offense_reward": total_off_reward,
            "total_defense_reward": total_def_reward,
            "play_types": dict(play_types),
            "off_formations": dict(formations_off),
            "def_formations": dict(formations_def),
            "coverages": dict(coverages),
            "off_violations_count": sum(1 for p in drive_plays if p["off_violations"]),
            "def_violations_count": sum(1 for p in drive_plays if p["def_violations"]),
        }
        drive_summaries.append(summary)

        elapsed = time.time() - start_time
        per_drive = elapsed / drive_num
        eta = per_drive * (n_drives - drive_num)
        print(
            f"  Drive {drive_num}/{n_drives}: {len(drive_plays)} plays, "
            f"{obs.drive_result}, {total_yards} yds, "
            f"off_r={total_off_reward:.1f} | "
            f"ETA: {eta:.0f}s",
            flush=True,
        )

    elapsed = time.time() - start_time
    print(f"\nDone! {n_drives} drives in {elapsed:.1f}s ({elapsed/n_drives:.2f}s/drive)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pbp_path = os.path.join(OUTPUT_DIR, "eval_play_by_play.json")
    with open(pbp_path, "w") as f:
        json.dump(all_plays, f, indent=1)
    print(f"Saved {len(all_plays)} plays to {pbp_path}")

    summary_path = os.path.join(OUTPUT_DIR, "eval_drive_summaries.json")
    with open(summary_path, "w") as f:
        json.dump(drive_summaries, f, indent=2)
    print(f"Saved {len(drive_summaries)} drive summaries to {summary_path}")

    # Print aggregate stats
    print("\n" + "=" * 60)
    print("AGGREGATE STATS")
    print("=" * 60)

    results = Counter(s["drive_result"] for s in drive_summaries)
    print(f"\nDrive outcomes:")
    for result, count in results.most_common():
        print(f"  {result}: {count} ({100*count/n_drives:.1f}%)")

    print(f"\nAvg plays/drive: {sum(s['num_plays'] for s in drive_summaries)/n_drives:.1f}")
    print(f"Avg yards/drive: {sum(s['total_yards'] for s in drive_summaries)/n_drives:.1f}")
    print(f"Avg offense reward/drive: {sum(s['total_offense_reward'] for s in drive_summaries)/n_drives:.2f}")
    print(f"Avg defense reward/drive: {sum(s['total_defense_reward'] for s in drive_summaries)/n_drives:.2f}")

    all_play_types = Counter()
    all_off_forms = Counter()
    all_def_forms = Counter()
    all_covs = Counter()
    for s in drive_summaries:
        all_play_types.update(s["play_types"])
        all_off_forms.update(s["off_formations"])
        all_def_forms.update(s["def_formations"])
        all_covs.update(s["coverages"])

    total_plays = sum(all_play_types.values())
    print(f"\nOffense play types:")
    for pt, count in all_play_types.most_common():
        print(f"  {pt}: {count} ({100*count/total_plays:.1f}%)")

    print(f"\nOffense formations:")
    for f, count in all_off_forms.most_common():
        print(f"  {f}: {count} ({100*count/total_plays:.1f}%)")

    print(f"\nDefense formations:")
    for f, count in all_def_forms.most_common():
        print(f"  {f}: {count} ({100*count/total_plays:.1f}%)")

    print(f"\nDefense coverages:")
    for c, count in all_covs.most_common():
        print(f"  {c}: {count} ({100*count/total_plays:.1f}%)")

    violations = sum(s["off_violations_count"] + s["def_violations_count"] for s in drive_summaries)
    print(f"\nTotal plays with violations: {violations}/{total_plays} ({100*violations/total_plays:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained adversarial football models")
    parser.add_argument("--n_drives", type=int, default=100, help="Number of drives to simulate")
    parser.add_argument("--checkpoint", nargs=2, default=None,
                        metavar=("OFFENSE", "DEFENSE"),
                        help="Checkpoint folder names (e.g. offense_lora_r40 defense_lora_r40)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir

    if args.checkpoint:
        off_ckpt = os.path.join(OUTPUT_DIR, args.checkpoint[0])
        def_ckpt = os.path.join(OUTPUT_DIR, args.checkpoint[1])
    else:
        off_ckpt = os.path.join(OUTPUT_DIR, "offense_lora_final")
        def_ckpt = os.path.join(OUTPUT_DIR, "defense_lora_final")

    run_eval(args.n_drives, off_ckpt, def_ckpt)
