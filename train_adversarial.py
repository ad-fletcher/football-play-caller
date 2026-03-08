"""
Adversarial REINFORCE Training: Offense vs Defense
===================================================
Two Qwen2.5-1.5B-Instruct models (LoRA) trained adversarially.
Offense learns to score, defense learns to stop them.

Custom REINFORCE with batch baseline (not GRPOTrainer).

Install:
  pip install unsloth torch transformers peft datasets matplotlib
"""

import json
import os
import pathlib
import random
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 2048
LORA_RANK = 32
OUTPUT_DIR = "checkpoints/adversarial"

# Training
EPISODES_PER_ROUND = 32        # drives per collection round
NUM_ROUNDS = 80                # total training rounds
LEARNING_RATE = 1e-5
MAX_COMPLETION_LENGTH = 80
TEMPERATURE = 0.7

# Phased training: (start_round, end_round, train_offense, train_defense)
TRAINING_PHASES = [
    (1,  15, True,  True),   # Phase 0: both learn basics (valid JSON, formations)
    (16, 40, True,  False),  # Phase 1: offense learns to exploit static defense
    (41, 65, False, True),   # Phase 2: defense adapts to trained offense
    (66, 80, True,  True),   # Phase 3: co-adaptation
]

# Unsloth
LOAD_4BIT = True
FAST_INFERENCE = False  # vLLM can only init once per process; use HF generate for two models
GPU_MEMORY_UTIL = 0.5

SAVE_EVERY = 10  # save every N rounds


# ──────────────────────────────────────────────
# Episode Collection
# ──────────────────────────────────────────────
def _get_stop_token_ids(tokenizer):
    """Get token IDs that should stop generation (closing brace = end of JSON)."""
    stop_ids = []
    for token in ["}", "}\n", "} ", "}\n\n"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if ids:
            stop_ids.append(ids[-1])
    return list(set(stop_ids)) if stop_ids else None


def _truncate_at_json_end(text: str) -> str:
    """Truncate response after first complete JSON object."""
    brace_end = text.find("}")
    if brace_end != -1:
        return text[:brace_end + 1]
    return text


def collect_episodes(env, off_model, off_tokenizer, def_model, def_tokenizer, n_episodes, device, round_num=0):
    """Roll out full drives, collect (prompt_ids, response_ids, reward) per agent + play-by-play log."""
    from football_env.models import GameAction
    from football_env.prompts import (
        format_offense_obs, format_defense_obs,
        parse_offense_response, parse_defense_response,
    )

    # Pre-compute stop tokens
    off_stop_ids = _get_stop_token_ids(off_tokenizer)
    def_stop_ids = _get_stop_token_ids(def_tokenizer)

    offense_episodes = []
    defense_episodes = []
    all_drive_results = []

    all_plays = []  # play-by-play log

    for ep in range(n_episodes):
        obs = env.reset()
        drive_off = []
        drive_def = []
        play_num = 0

        while not obs.done:
            # Snapshot pre-play state
            pre_state = {
                "down": obs.down, "yardsToGo": obs.yardsToGo,
                "yardline": obs.absoluteYardlineNumber,
                "quarter": obs.quarter, "clock": obs.gameClock_seconds,
                "score_diff": obs.score_diff,
            }

            # Format prompts
            off_messages = format_offense_obs(obs)
            def_messages = format_defense_obs(obs)

            # Generate offense action
            off_text = off_tokenizer.apply_chat_template(off_messages, tokenize=False, add_generation_prompt=True)
            off_ids = off_tokenizer(off_text, return_tensors="pt").to(device)
            gen_kwargs = dict(max_new_tokens=MAX_COMPLETION_LENGTH, temperature=TEMPERATURE, do_sample=True)
            if off_stop_ids:
                gen_kwargs["eos_token_id"] = off_stop_ids
            with torch.no_grad():
                off_out = off_model.generate(**off_ids, **gen_kwargs)
            off_response_ids = off_out[0][off_ids["input_ids"].shape[1]:]
            off_response_text = _truncate_at_json_end(off_tokenizer.decode(off_response_ids, skip_special_tokens=True))

            # Generate defense action
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

            # Step environment
            composite = GameAction(offense=off_action, defense=def_action)
            obs = env.step(composite)

            # Store (prompt_text, response_text, reward)
            drive_off.append((off_text, off_response_text, obs.offense_reward))
            drive_def.append((def_text, def_response_text, obs.defense_reward))

            # Log play-by-play
            play_num += 1
            all_plays.append({
                "round": round_num, "drive": ep + 1, "play": play_num,
                **pre_state,
                "offense": off_action.model_dump(),
                "defense": def_action.model_dump(),
                "raw_offense_response": off_response_text,
                "raw_defense_response": def_response_text,
                "result": obs.last_play_result,
                "yards": obs.last_play_yards,
                "offense_reward": obs.offense_reward,
                "defense_reward": obs.defense_reward,
                "drive_result": obs.drive_result if obs.done else None,
            })

        offense_episodes.extend(drive_off)
        defense_episodes.extend(drive_def)
        all_drive_results.append(obs.drive_result)

        total_off_r = sum(r for _, _, r in drive_off)
        print(f"  Drive {ep+1}/{n_episodes}: {len(drive_off)} plays, {obs.drive_result}, off_reward={total_off_r:.1f}", flush=True)

    return offense_episodes, defense_episodes, all_drive_results, all_plays


# ──────────────────────────────────────────────
# REINFORCE Update
# ──────────────────────────────────────────────
def reinforce_update(model, tokenizer, optimizer, episodes, device):
    """REINFORCE with batch mean baseline."""
    if not episodes:
        return 0.0

    prompts, responses, rewards = zip(*episodes)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    advantages = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)

    model.train()
    total_loss = 0.0

    for prompt, response, advantage in zip(prompts, responses, advantages):
        # Re-enable gradients for log prob computation
        full_text = prompt + response
        full_ids = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = prompt_ids["input_ids"].shape[1]

        outputs = model(**full_ids)
        logits = outputs.logits[0, prompt_len - 1:-1, :]
        response_token_ids = full_ids["input_ids"][0, prompt_len:]

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_token_ids.unsqueeze(1)).squeeze(1)
        log_prob = token_log_probs.sum()

        loss = -(advantage * log_prob)
        loss.backward()
        total_loss += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / len(episodes)


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
@dataclass
class TrainingLog:
    rounds: list = field(default_factory=list)
    offense_rewards: list = field(default_factory=list)
    defense_rewards: list = field(default_factory=list)
    offense_losses: list = field(default_factory=list)
    defense_losses: list = field(default_factory=list)
    drive_results: list = field(default_factory=list)
    td_rates: list = field(default_factory=list)
    turnover_rates: list = field(default_factory=list)
    punt_rates: list = field(default_factory=list)

    def log_round(self, round_num, off_episodes, def_episodes, off_loss, def_loss, drive_results):
        self.rounds.append(round_num)
        off_rewards = [r for _, _, r in off_episodes]
        def_rewards = [r for _, _, r in def_episodes]
        self.offense_rewards.append(float(np.mean(off_rewards)))
        self.defense_rewards.append(float(np.mean(def_rewards)))
        self.offense_losses.append(off_loss)
        self.defense_losses.append(def_loss)
        self.drive_results.append(drive_results)

        # Compute outcome rates
        n = max(len(drive_results), 1)
        from collections import Counter
        counts = Counter(drive_results)
        self.td_rates.append(counts.get("touchdown", 0) / n)
        self.turnover_rates.append((counts.get("interception", 0) + counts.get("fumble_lost", 0)) / n)
        self.punt_rates.append(counts.get("punt", 0) / n)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "training_log.json"), "w") as f:
            json.dump({
                "rounds": self.rounds,
                "offense_rewards": self.offense_rewards,
                "defense_rewards": self.defense_rewards,
                "offense_losses": self.offense_losses,
                "defense_losses": self.defense_losses,
                "drive_results": self.drive_results,
                "td_rates": self.td_rates,
                "turnover_rates": self.turnover_rates,
                "punt_rates": self.punt_rates,
            }, f, indent=2)

        self._plot_rewards(output_dir)
        self._plot_outcomes(output_dir)
        self._plot_losses(output_dir)
        self._plot_combined(output_dir)

    def _add_phase_bands(self, ax):
        """Add shaded background bands for each training phase."""
        colors = ["#e8e8e8", "#cce5ff", "#ffe0cc", "#d4edda"]
        labels = ["Both train", "Off trains", "Def trains", "Both train"]
        for i, (start, end, _, _) in enumerate(TRAINING_PHASES):
            ax.axvspan(start - 0.5, end + 0.5, alpha=0.3, color=colors[i % len(colors)], label=labels[i] if i < len(labels) else "")

    def _plot_rewards(self, output_dir):
        fig, ax = plt.subplots(figsize=(12, 5))
        self._add_phase_bands(ax)
        ax.plot(self.rounds, self.offense_rewards, "b-", alpha=0.4, linewidth=1)
        ax.plot(self.rounds, self.defense_rewards, "r-", alpha=0.4, linewidth=1)
        w = min(5, len(self.rounds))
        if w > 1:
            off_ma = np.convolve(self.offense_rewards, np.ones(w)/w, mode="valid")
            def_ma = np.convolve(self.defense_rewards, np.ones(w)/w, mode="valid")
            ax.plot(self.rounds[w-1:], off_ma, "b-", lw=2.5, label="Offense (MA)")
            ax.plot(self.rounds[w-1:], def_ma, "r-", lw=2.5, label="Defense (MA)")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Mean Reward per Play", fontsize=12)
        ax.set_title("Adversarial Training: Reward Curves", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "reward_curves.png"), dpi=150)
        plt.close(fig)

    def _plot_outcomes(self, output_dir):
        fig, ax = plt.subplots(figsize=(12, 5))
        self._add_phase_bands(ax)
        w = min(5, len(self.rounds))
        if w > 1 and len(self.td_rates) >= w:
            td_ma = np.convolve(self.td_rates, np.ones(w)/w, mode="valid")
            to_ma = np.convolve(self.turnover_rates, np.ones(w)/w, mode="valid")
            punt_ma = np.convolve(self.punt_rates, np.ones(w)/w, mode="valid")
            ax.plot(self.rounds[w-1:], td_ma, "g-", lw=2.5, label="TD rate")
            ax.plot(self.rounds[w-1:], to_ma, "r-", lw=2.5, label="Turnover rate")
            ax.plot(self.rounds[w-1:], punt_ma, "gray", lw=2.5, label="Punt rate")
        else:
            ax.plot(self.rounds, self.td_rates, "g-", lw=2, label="TD rate")
            ax.plot(self.rounds, self.turnover_rates, "r-", lw=2, label="Turnover rate")
            ax.plot(self.rounds, self.punt_rates, "gray", lw=2, label="Punt rate")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Rate per Drive", fontsize=12)
        ax.set_title("Drive Outcomes Over Training", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "outcome_curves.png"), dpi=150)
        plt.close(fig)

    def _plot_losses(self, output_dir):
        fig, ax = plt.subplots(figsize=(12, 5))
        self._add_phase_bands(ax)
        ax.plot(self.rounds, self.offense_losses, "b-", alpha=0.7, lw=1.5, label="Offense Loss")
        ax.plot(self.rounds, self.defense_losses, "r-", alpha=0.7, lw=1.5, label="Defense Loss")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Policy Loss", fontsize=12)
        ax.set_title("REINFORCE Policy Losses", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=150)
        plt.close(fig)

    def _plot_combined(self, output_dir):
        """Single figure with all three panels — good for the demo."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Rewards
        ax = axes[0]
        self._add_phase_bands(ax)
        ax.plot(self.rounds, self.offense_rewards, "b-", alpha=0.4, lw=1)
        ax.plot(self.rounds, self.defense_rewards, "r-", alpha=0.4, lw=1)
        w = min(5, len(self.rounds))
        if w > 1:
            off_ma = np.convolve(self.offense_rewards, np.ones(w)/w, mode="valid")
            def_ma = np.convolve(self.defense_rewards, np.ones(w)/w, mode="valid")
            ax.plot(self.rounds[w-1:], off_ma, "b-", lw=2.5, label="Offense")
            ax.plot(self.rounds[w-1:], def_ma, "r-", lw=2.5, label="Defense")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Rewards")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Outcomes
        ax = axes[1]
        self._add_phase_bands(ax)
        if w > 1 and len(self.td_rates) >= w:
            ax.plot(self.rounds[w-1:], np.convolve(self.td_rates, np.ones(w)/w, mode="valid"), "g-", lw=2.5, label="TD")
            ax.plot(self.rounds[w-1:], np.convolve(self.turnover_rates, np.ones(w)/w, mode="valid"), "r-", lw=2.5, label="Turnover")
            ax.plot(self.rounds[w-1:], np.convolve(self.punt_rates, np.ones(w)/w, mode="valid"), "gray", lw=2.5, label="Punt")
        ax.set_xlabel("Round")
        ax.set_ylabel("Rate")
        ax.set_title("Drive Outcomes")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Losses
        ax = axes[2]
        self._add_phase_bands(ax)
        ax.plot(self.rounds, self.offense_losses, "b-", lw=1.5, alpha=0.7, label="Offense")
        ax.plot(self.rounds, self.defense_losses, "r-", lw=1.5, alpha=0.7, label="Defense")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Adversarial Football: Offense vs Defense Training", fontsize=16, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_dashboard.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load models ──
    from unsloth import FastLanguageModel

    print(f"Loading offense model: {MODEL_NAME}...")
    off_model, off_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
    )
    off_model = FastLanguageModel.get_peft_model(
        off_model, r=LORA_RANK, lora_alpha=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    print(f"Loading defense model: {MODEL_NAME}...")
    def_model, def_tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
    )
    def_model = FastLanguageModel.get_peft_model(
        def_model, r=LORA_RANK, lora_alpha=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=43,
    )

    # ── Optimizers ──
    off_optimizer = AdamW(off_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    def_optimizer = AdamW(def_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # ── Environment ──
    from football_env.server.environment import FootballDriveEnvironment
    env = FootballDriveEnvironment()

    # ── Training loop ──
    log = TrainingLog()
    all_plays = []  # play-by-play across all rounds
    print(f"Starting adversarial training: {NUM_ROUNDS} rounds, {EPISODES_PER_ROUND} episodes/round")
    print(f"Training phases:")
    for start, end, train_off, train_def in TRAINING_PHASES:
        off_str = "TRAIN" if train_off else "frozen"
        def_str = "TRAIN" if train_def else "frozen"
        print(f"  Rounds {start}-{end}: offense={off_str}, defense={def_str}")

    for round_num in range(1, NUM_ROUNDS + 1):
        # Determine current phase
        train_offense, train_defense = True, True
        for start, end, t_off, t_def in TRAINING_PHASES:
            if start <= round_num <= end:
                train_offense, train_defense = t_off, t_def
                break

        # Collect episodes (inference mode — disables dropout, enables 2x faster generation)
        FastLanguageModel.for_inference(off_model)
        FastLanguageModel.for_inference(def_model)
        off_episodes, def_episodes, drive_results, plays = collect_episodes(
            env, off_model, off_tokenizer, def_model, def_tokenizer,
            EPISODES_PER_ROUND, device, round_num=round_num,
        )
        all_plays.extend(plays)

        # Update offense
        off_loss = 0.0
        if train_offense:
            FastLanguageModel.for_training(off_model)
            off_loss = reinforce_update(off_model, off_tokenizer, off_optimizer, off_episodes, device)

        # Update defense
        def_loss = 0.0
        if train_defense:
            FastLanguageModel.for_training(def_model)
            def_loss = reinforce_update(def_model, def_tokenizer, def_optimizer, def_episodes, device)

        # Log
        off_mean = np.mean([r for _, _, r in off_episodes])
        def_mean = np.mean([r for _, _, r in def_episodes])
        log.log_round(round_num, off_episodes, def_episodes, off_loss, def_loss, drive_results)
        phase_str = f"[off={'T' if train_offense else 'F'} def={'T' if train_defense else 'F'}]"
        from collections import Counter
        result_summary = ", ".join(f"{k}:{v}" for k, v in Counter(drive_results).most_common())
        print(f"Round {round_num}/{NUM_ROUNDS} {phase_str} | Off reward: {off_mean:.3f} | Def reward: {def_mean:.3f} | Off loss: {off_loss:.4f} | Def loss: {def_loss:.4f} | {result_summary}", flush=True)

        # Save checkpoints
        if round_num % SAVE_EVERY == 0:
            print(f"  Saving checkpoint at round {round_num}...")
            off_model.save_pretrained(os.path.join(OUTPUT_DIR, f"offense_lora_r{round_num}"))
            def_model.save_pretrained(os.path.join(OUTPUT_DIR, f"defense_lora_r{round_num}"))
            log.save(OUTPUT_DIR)
            with open(os.path.join(OUTPUT_DIR, "play_by_play.json"), "w") as f:
                json.dump(all_plays, f)

    # Final save
    print("Saving final models...")
    off_model.save_pretrained(os.path.join(OUTPUT_DIR, "offense_lora_final"))
    def_model.save_pretrained(os.path.join(OUTPUT_DIR, "defense_lora_final"))
    with open(os.path.join(OUTPUT_DIR, "play_by_play.json"), "w") as f:
        json.dump(all_plays, f)
    log.save(OUTPUT_DIR)
    print(f"Training complete! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
