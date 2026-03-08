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
EPISODES_PER_ROUND = 64       # drives per collection round
NUM_ROUNDS = 100               # total training rounds
LEARNING_RATE = 1e-5
MAX_COMPLETION_LENGTH = 150
TEMPERATURE = 0.7

# Unsloth
LOAD_4BIT = True
FAST_INFERENCE = False  # vLLM can only init once per process; use HF generate for two models
GPU_MEMORY_UTIL = 0.5

SAVE_EVERY = 10  # save every N rounds


# ──────────────────────────────────────────────
# Episode Collection
# ──────────────────────────────────────────────
def collect_episodes(env, off_model, off_tokenizer, def_model, def_tokenizer, n_episodes, device):
    """Roll out full drives, collect (prompt_ids, response_ids, reward) per agent."""
    from football_env.models import GameAction
    from football_env.prompts import (
        format_offense_obs, format_defense_obs,
        parse_offense_response, parse_defense_response,
    )

    offense_episodes = []
    defense_episodes = []

    for ep in range(n_episodes):
        obs = env.reset()
        drive_off = []
        drive_def = []

        while not obs.done:
            # Format prompts
            off_messages = format_offense_obs(obs)
            def_messages = format_defense_obs(obs)

            # Generate offense action
            off_text = off_tokenizer.apply_chat_template(off_messages, tokenize=False, add_generation_prompt=True)
            off_ids = off_tokenizer(off_text, return_tensors="pt").to(device)
            with torch.no_grad():
                off_out = off_model.generate(
                    **off_ids, max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=TEMPERATURE, do_sample=True,
                )
            off_response_ids = off_out[0][off_ids["input_ids"].shape[1]:]
            off_response_text = off_tokenizer.decode(off_response_ids, skip_special_tokens=True)

            # Generate defense action
            def_text = def_tokenizer.apply_chat_template(def_messages, tokenize=False, add_generation_prompt=True)
            def_ids = def_tokenizer(def_text, return_tensors="pt").to(device)
            with torch.no_grad():
                def_out = def_model.generate(
                    **def_ids, max_new_tokens=MAX_COMPLETION_LENGTH,
                    temperature=TEMPERATURE, do_sample=True,
                )
            def_response_ids = def_out[0][def_ids["input_ids"].shape[1]:]
            def_response_text = def_tokenizer.decode(def_response_ids, skip_special_tokens=True)

            # Parse actions
            off_action = parse_offense_response(off_response_text)
            def_action = parse_defense_response(def_response_text)

            # Step environment
            composite = GameAction(offense=off_action, defense=def_action)
            obs = env.step(composite)

            # Store (prompt_text, response_text, reward)
            drive_off.append((off_text, off_response_text, obs.offense_reward))
            drive_def.append((def_text, def_response_text, obs.defense_reward))

        offense_episodes.extend(drive_off)
        defense_episodes.extend(drive_def)

    return offense_episodes, defense_episodes


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

    def log_round(self, round_num, off_episodes, def_episodes, off_loss, def_loss, drive_results):
        self.rounds.append(round_num)
        off_rewards = [r for _, _, r in off_episodes]
        def_rewards = [r for _, _, r in def_episodes]
        self.offense_rewards.append(float(np.mean(off_rewards)))
        self.defense_rewards.append(float(np.mean(def_rewards)))
        self.offense_losses.append(off_loss)
        self.defense_losses.append(def_loss)
        self.drive_results.append(drive_results)

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
            }, f, indent=2)

        # Plot reward curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(self.rounds, self.offense_rewards, "b-", alpha=0.5, label="Offense")
        ax.plot(self.rounds, self.defense_rewards, "r-", alpha=0.5, label="Defense")
        w = min(10, len(self.rounds))
        if w > 1:
            off_ma = np.convolve(self.offense_rewards, np.ones(w)/w, mode="valid")
            def_ma = np.convolve(self.defense_rewards, np.ones(w)/w, mode="valid")
            ax.plot(self.rounds[w-1:], off_ma, "b-", lw=2, label=f"Off MA-{w}")
            ax.plot(self.rounds[w-1:], def_ma, "r-", lw=2, label=f"Def MA-{w}")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Adversarial Training: Rewards")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(self.rounds, self.offense_losses, "b-", alpha=0.5, label="Offense Loss")
        ax.plot(self.rounds, self.defense_losses, "r-", alpha=0.5, label="Defense Loss")
        ax.set_xlabel("Round")
        ax.set_ylabel("Loss")
        ax.set_title("Policy Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
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
    print(f"Starting adversarial training: {NUM_ROUNDS} rounds, {EPISODES_PER_ROUND} episodes/round")

    for round_num in range(1, NUM_ROUNDS + 1):
        # Collect episodes (inference mode — disables dropout, enables 2x faster generation)
        FastLanguageModel.for_inference(off_model)
        FastLanguageModel.for_inference(def_model)
        off_episodes, def_episodes = collect_episodes(
            env, off_model, off_tokenizer, def_model, def_tokenizer,
            EPISODES_PER_ROUND, device,
        )

        # Update offense (training mode — re-enables LoRA gradients)
        FastLanguageModel.for_training(off_model)
        off_loss = reinforce_update(off_model, off_tokenizer, off_optimizer, off_episodes, device)

        # Update defense
        FastLanguageModel.for_training(def_model)
        def_loss = reinforce_update(def_model, def_tokenizer, def_optimizer, def_episodes, device)

        # Log
        off_mean = np.mean([r for _, _, r in off_episodes])
        def_mean = np.mean([r for _, _, r in def_episodes])
        log.log_round(round_num, off_episodes, def_episodes, off_loss, def_loss, {})
        print(f"Round {round_num}/{NUM_ROUNDS} | Off reward: {off_mean:.3f} | Def reward: {def_mean:.3f} | Off loss: {off_loss:.4f} | Def loss: {def_loss:.4f} | Episodes: {len(off_episodes)} plays")

        # Save checkpoints
        if round_num % SAVE_EVERY == 0:
            print(f"  Saving checkpoint at round {round_num}...")
            off_model.save_pretrained(os.path.join(OUTPUT_DIR, f"offense_lora_r{round_num}"))
            def_model.save_pretrained(os.path.join(OUTPUT_DIR, f"defense_lora_r{round_num}"))
            log.save(OUTPUT_DIR)

    # Final save
    print("Saving final models...")
    off_model.save_pretrained(os.path.join(OUTPUT_DIR, "offense_lora_final"))
    def_model.save_pretrained(os.path.join(OUTPUT_DIR, "defense_lora_final"))
    log.save(OUTPUT_DIR)
    print(f"Training complete! Results in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
