"""
Evaluate models at different training checkpoints to show learning progression.

Runs eval at:
  - Round 10  (Phase 0: both training — early/random)
  - Round 30  (Phase 1: offense training, defense frozen)
  - Round 50  (Phase 2: defense training, offense frozen)
  - Round 70  (Phase 3: both co-adapting)
  - Final     (end of training)

Usage:
  python eval_phases.py                  # 20 drives per checkpoint
  python eval_phases.py --n_drives 50   # more drives per checkpoint
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
MAX_SEQ_LENGTH = 2048
MAX_COMPLETION_LENGTH = 80
TEMPERATURE = 0.7
CHECKPOINT_DIR = "checkpoints/adversarial"

CHECKPOINTS = [
    ("round_10", "offense_lora_r10", "defense_lora_r10", "Phase 0: Both learning basics"),
    ("round_30", "offense_lora_r30", "defense_lora_r30", "Phase 1: Offense trains, defense frozen"),
    ("round_50", "offense_lora_r50", "defense_lora_r50", "Phase 2: Defense trains, offense frozen"),
    ("round_70", "offense_lora_r70", "defense_lora_r70", "Phase 3: Both co-adapting"),
    ("final",    "offense_lora_final", "defense_lora_final", "Final models"),
]


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


def run_drives(env, off_model, off_tokenizer, def_model, def_tokenizer,
               off_stop_ids, def_stop_ids, n_drives, device, checkpoint_name):
    from football_env.models import GameAction
    from football_env.prompts import (
        format_offense_obs, format_defense_obs,
        parse_offense_response, parse_defense_response,
    )
    from football_env.validation import validate_offense, validate_defense

    all_plays = []
    drive_summaries = []

    for drive_num in range(1, n_drives + 1):
        obs = env.reset()
        drive_plays = []
        play_num = 0

        while not obs.done:
            pre_state = {
                "down": obs.down, "yardsToGo": obs.yardsToGo,
                "yardline": obs.absoluteYardlineNumber,
                "quarter": obs.quarter, "clock": obs.gameClock_seconds,
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
                def_out = def_model.generate(**off_ids, **gen_kwargs)
            def_response_ids = def_out[0][def_ids["input_ids"].shape[1]:]
            def_response_text = _truncate_at_json_end(def_tokenizer.decode(def_response_ids, skip_special_tokens=True))

            off_action = parse_offense_response(off_response_text)
            def_action = parse_defense_response(def_response_text)

            off_penalty, off_violations = validate_offense(off_action, pre_state["down"])
            def_penalty, def_violations = validate_defense(def_action)

            composite = GameAction(offense=off_action, defense=def_action)
            obs = env.step(composite)

            play_num += 1
            play_record = {
                "checkpoint": checkpoint_name,
                "drive": drive_num, "play": play_num,
                **pre_state,
                "off_formation": off_action.offenseFormation,
                "off_playType": off_action.playType,
                "off_designedPass": off_action.designedPass,
                "off_receiverAlignment": off_action.receiverAlignment,
                "off_runConcept": off_action.pff_runConceptPrimary,
                "off_raw_response": off_response_text,
                "off_violations": off_violations,
                "def_formation": def_action.defFormation,
                "def_manZone": def_action.pff_manZone,
                "def_coverage": def_action.pff_passCoverage,
                "def_rushers": def_action.passRushers,
                "def_raw_response": def_response_text,
                "def_violations": def_violations,
                "result": obs.last_play_result,
                "yards": obs.last_play_yards,
                "offense_reward": obs.offense_reward,
                "defense_reward": obs.defense_reward,
                "drive_result": obs.drive_result if obs.done else None,
            }
            drive_plays.append(play_record)
            all_plays.append(play_record)

        total_yards = sum(p["yards"] for p in drive_plays)
        total_off_r = sum(p["offense_reward"] for p in drive_plays)
        drive_summaries.append({
            "checkpoint": checkpoint_name,
            "drive": drive_num,
            "num_plays": len(drive_plays),
            "drive_result": obs.drive_result,
            "total_yards": total_yards,
            "total_offense_reward": total_off_r,
            "play_types": dict(Counter(p["off_playType"] for p in drive_plays)),
            "off_formations": dict(Counter(p["off_formation"] for p in drive_plays)),
            "def_formations": dict(Counter(p["def_formation"] for p in drive_plays)),
            "coverages": dict(Counter(p["def_coverage"] for p in drive_plays)),
        })

        print(f"    Drive {drive_num}/{n_drives}: {len(drive_plays)} plays, {obs.drive_result}, {total_yards:.0f} yds", flush=True)

    return all_plays, drive_summaries


def print_phase_summary(name, description, summaries):
    n = len(summaries)
    results = Counter(s["drive_result"] for s in summaries)
    avg_plays = sum(s["num_plays"] for s in summaries) / n
    avg_yards = sum(s["total_yards"] for s in summaries) / n
    avg_off_r = sum(s["total_offense_reward"] for s in summaries) / n

    all_pt = Counter()
    all_of = Counter()
    all_df = Counter()
    all_cov = Counter()
    for s in summaries:
        all_pt.update(s["play_types"])
        all_of.update(s["off_formations"])
        all_df.update(s["def_formations"])
        all_cov.update(s["coverages"])
    total_plays = sum(all_pt.values())

    print(f"\n{'='*60}")
    print(f"  {name}: {description}")
    print(f"{'='*60}")
    print(f"  Outcomes: {', '.join(f'{k}:{v}' for k,v in results.most_common())}")
    print(f"  TD rate: {results.get('touchdown',0)/n*100:.0f}% | Turnover: {(results.get('interception',0)+results.get('fumble_lost',0))/n*100:.0f}% | Punt: {results.get('punt',0)/n*100:.0f}%")
    print(f"  Avg plays/drive: {avg_plays:.1f} | Avg yards: {avg_yards:.1f} | Avg off reward: {avg_off_r:.1f}")
    print(f"  Play types: {', '.join(f'{k}:{v}' for k,v in all_pt.most_common(5))}")
    print(f"  Off formations: {', '.join(f'{k}:{v}' for k,v in all_of.most_common(5))}")
    print(f"  Def formations: {', '.join(f'{k}:{v}' for k,v in all_df.most_common(5))}")
    print(f"  Coverages: {', '.join(f'{k}:{v}' for k,v in all_cov.most_common(5))}")

    violations = 0
    for s in summaries:
        # Count from plays directly - we'll compute from the all_plays data
        pass
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_drives", type=int, default=15)
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    args = parser.parse_args()

    from football_env.server.environment import FootballDriveEnvironment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = FootballDriveEnvironment()

    all_plays_combined = []
    all_summaries_combined = []

    # We need to reload models for each checkpoint
    # Unsloth can load LoRA on top of base — load base once, swap LoRA
    for i, (name, off_name, def_name, description) in enumerate(CHECKPOINTS):
        off_path = os.path.join(args.checkpoint_dir, off_name)
        def_path = os.path.join(args.checkpoint_dir, def_name)

        if not os.path.exists(off_path) or not os.path.exists(def_path):
            print(f"\nSkipping {name}: checkpoint not found ({off_path} or {def_path})")
            continue

        print(f"\n{'#'*60}")
        print(f"  Loading {name}: {description}")
        print(f"{'#'*60}")

        # Load models fresh for each checkpoint
        off_model, off_tokenizer = FastLanguageModel.from_pretrained(
            model_name=off_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(off_model)

        def_model, def_tokenizer = FastLanguageModel.from_pretrained(
            model_name=def_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(def_model)

        off_stop_ids = _get_stop_token_ids(off_tokenizer)
        def_stop_ids = _get_stop_token_ids(def_tokenizer)

        start = time.time()
        plays, summaries = run_drives(
            env, off_model, off_tokenizer, def_model, def_tokenizer,
            off_stop_ids, def_stop_ids, args.n_drives, device, name,
        )
        elapsed = time.time() - start
        print(f"  ({elapsed:.0f}s for {args.n_drives} drives)")

        all_plays_combined.extend(plays)
        all_summaries_combined.extend(summaries)

        print_phase_summary(name, description, summaries)

        # Save incrementally
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(os.path.join(args.checkpoint_dir, "phase_play_by_play.json"), "w") as f:
            json.dump(all_plays_combined, f, indent=1)
        with open(os.path.join(args.checkpoint_dir, "phase_summaries.json"), "w") as f:
            json.dump(all_summaries_combined, f, indent=2)

        # Free memory before loading next checkpoint
        del off_model, def_model, off_tokenizer, def_tokenizer
        torch.cuda.empty_cache()

    # Final comparison table
    print("\n" + "=" * 80)
    print("  TRAINING PROGRESSION COMPARISON")
    print("=" * 80)
    print(f"  {'Phase':<12} {'TD%':>6} {'INT%':>6} {'Punt%':>6} {'Avg Yds':>8} {'Avg Plays':>10} {'Off Reward':>11}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*11}")

    for name, _, _, desc in CHECKPOINTS:
        sums = [s for s in all_summaries_combined if s["checkpoint"] == name]
        if not sums:
            continue
        n = len(sums)
        results = Counter(s["drive_result"] for s in sums)
        td_pct = results.get("touchdown", 0) / n * 100
        int_pct = (results.get("interception", 0) + results.get("fumble_lost", 0)) / n * 100
        punt_pct = results.get("punt", 0) / n * 100
        avg_yds = sum(s["total_yards"] for s in sums) / n
        avg_plays = sum(s["num_plays"] for s in sums) / n
        avg_off_r = sum(s["total_offense_reward"] for s in sums) / n
        print(f"  {name:<12} {td_pct:>5.0f}% {int_pct:>5.0f}% {punt_pct:>5.0f}% {avg_yds:>7.1f} {avg_plays:>9.1f} {avg_off_r:>10.1f}")

    print(f"\nAll data saved to:")
    print(f"  {os.path.join(args.checkpoint_dir, 'phase_play_by_play.json')}")
    print(f"  {os.path.join(args.checkpoint_dir, 'phase_summaries.json')}")


if __name__ == "__main__":
    main()
