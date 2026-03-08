"""
GRPO Training: NFL Play-Calling with Structured JSON Output
============================================================
Unsloth + TRL GRPOTrainer | Qwen2.5-1.5B-Instruct | LoRA
Target: Northflank H100

The LLM learns to be an offensive coordinator:
  - Sees defensive formation
  - Outputs JSON: {"formation": "...", "play_type": "..."}
  - Gets rewarded with real NFL yard outcomes from lookup table

Install:
  pip install unsloth trl datasets matplotlib
  # or: pip install "unsloth[cu124-torch260]" for CUDA 12.4
"""

import json
import os
import pathlib
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_NAME = "unsloth/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 512
LORA_RANK = 32
OUTPUT_DIR = "checkpoints/football_grpo"
LOOKUP_TABLE_PATH = "data/lookup_table.json"

# Training hyperparams
NUM_TRAIN_PROMPTS = 4096
NUM_GENERATIONS = 8       # completions per prompt (GRPO group size)
MAX_STEPS = 500
SAVE_STEPS = 100
BATCH_SIZE = 4            # prompts per device per step
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 5e-6
MAX_COMPLETION_LENGTH = 128

# Unsloth / vLLM
FAST_INFERENCE = True     # vLLM-accelerated generation (set False if issues)
LOAD_4BIT = True
GPU_MEMORY_UTIL = 0.5     # fraction for vLLM; rest for training

# ──────────────────────────────────────────────
# Lookup table (NFL play outcomes)
# ──────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
with open(SCRIPT_DIR / LOOKUP_TABLE_PATH) as f:
    _data = json.load(f)

LOOKUP = _data["lookup"]
DEFENSE_NAMES = list(_data["defense_distribution"].keys())
DEFENSE_WEIGHTS = list(_data["defense_distribution"].values())

VALID_FORMATIONS = ["SHOTGUN", "SINGLEBACK", "EMPTY", "I_FORM", "PISTOL"]
VALID_PLAY_TYPES = ["run", "pass", "play_action"]

# ──────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an NFL offensive coordinator. Given the game situation and defensive formation, call a play.

Respond with ONLY a JSON object in this exact format:
{"formation": "<FORMATION>", "play_type": "<PLAY_TYPE>"}

Valid formations: SHOTGUN, SINGLEBACK, EMPTY, I_FORM, PISTOL
Valid play types: run, pass, play_action"""


def make_prompt(defense: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"1st & 10 from your own 25. Defense: {defense}. Call your play."},
    ]


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
def build_dataset(n: int = NUM_TRAIN_PROMPTS) -> Dataset:
    """Each row = one episode with a randomly sampled defense."""
    defenses = random.choices(DEFENSE_NAMES, weights=DEFENSE_WEIGHTS, k=n)
    return Dataset.from_dict({
        "prompt": [make_prompt(d) for d in defenses],
        "defense_formation": defenses,
    })


# ──────────────────────────────────────────────
# Reward functions
# ──────────────────────────────────────────────
def _extract_text(completion) -> str:
    """Get raw text from a completion (handles both chat and string format)."""
    if isinstance(completion, list):
        return completion[0]["content"]
    return str(completion)


def _parse_action(text: str) -> tuple[str, str] | None:
    """Parse JSON action from model output. Returns (formation, play_type) or None."""
    text = text.strip()
    # Try to find JSON in the text (model might add extra text around it)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(text[start:end])
        formation = parsed.get("formation", "")
        play_type = parsed.get("play_type", "")
        if formation in VALID_FORMATIONS and play_type in VALID_PLAY_TYPES:
            return (formation, play_type)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return None


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for producing valid JSON with valid action fields.

    +1.0 for valid JSON with correct formation & play_type
    -5.0 for anything else
    """
    rewards = []
    for c in completions:
        action = _parse_action(_extract_text(c))
        rewards.append(1.0 if action else -5.0)
    return rewards


def play_reward(completions, defense_formation, **kwargs) -> list[float]:
    """Reward = sampled yards gained from real NFL data.

    Uses the lookup table: (formation, play_type, defense) → sample outcome.
    Invalid/unparseable actions get -5.0.
    """
    rewards = []
    for c, defense in zip(completions, defense_formation):
        action = _parse_action(_extract_text(c))
        if action is None:
            rewards.append(-5.0)
            continue

        formation, play_type = action
        key = f"{formation}|{play_type}|{defense}"

        if key in LOOKUP:
            bucket = LOOKUP[key]
            idx = random.randrange(len(bucket["rewards"]))
            rewards.append(float(bucket["rewards"][idx]))
        else:
            # Fallback: pool all defenses for same (formation, play_type)
            prefix = f"{formation}|{play_type}|"
            fallback = [k for k in LOOKUP if k.startswith(prefix)]
            if fallback:
                all_r = []
                for fk in fallback:
                    all_r.extend(LOOKUP[fk]["rewards"])
                rewards.append(float(random.choice(all_r)))
            else:
                rewards.append(0.0)

    return rewards


# ──────────────────────────────────────────────
# Logging callback
# ──────────────────────────────────────────────
class RewardLogger(TrainerCallback):
    """Captures per-step reward metrics for plotting."""

    def __init__(self):
        self.steps = []
        self.rewards = []
        self.format_rewards = []
        self.play_rewards = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        step = state.global_step
        if "reward" in logs:
            self.steps.append(step)
            self.rewards.append(logs["reward"])
        if "rewards/format_reward" in logs:
            self.format_rewards.append(logs["rewards/format_reward"])
        if "rewards/play_reward" in logs:
            self.play_rewards.append(logs["rewards/play_reward"])

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        if not self.steps:
            print("No reward data to save.")
            return

        # Save raw data
        log_path = os.path.join(output_dir, "reward_log.json")
        with open(log_path, "w") as f:
            json.dump({
                "steps": self.steps,
                "rewards": self.rewards,
                "format_rewards": self.format_rewards,
                "play_rewards": self.play_rewards,
            }, f, indent=2)
        print(f"Reward log saved to {log_path}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Total reward
        ax = axes[0]
        ax.plot(self.steps, self.rewards, "b-", alpha=0.3, label="Raw")
        w = min(20, len(self.rewards))
        if w > 1:
            ma = np.convolve(self.rewards, np.ones(w) / w, mode="valid")
            ax.plot(self.steps[w - 1:], ma, "r-", lw=2, label=f"MA-{w}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Reward")
        ax.set_title("GRPO Training: Total Reward")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Component rewards
        ax = axes[1]
        if self.format_rewards:
            ax.plot(self.steps[:len(self.format_rewards)], self.format_rewards,
                    "g-", alpha=0.5, label="Format")
        if self.play_rewards:
            ax.plot(self.steps[:len(self.play_rewards)], self.play_rewards,
                    "b-", alpha=0.5, label="Play (yards)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Components")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, "reward_curve.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Reward curve saved to {plot_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(42)

    # ── Load model ──
    from unsloth import FastLanguageModel

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_4BIT,
        fast_inference=FAST_INFERENCE,
        max_lora_rank=LORA_RANK,
        gpu_memory_utilization=GPU_MEMORY_UTIL,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Dataset ──
    print(f"Building dataset ({NUM_TRAIN_PROMPTS} episodes)...")
    dataset = build_dataset()

    # ── Training config ──
    reward_logger = RewardLogger()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # Optimizer
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.1,
        bf16=True,
        # Batching
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        # GRPO
        num_generations=NUM_GENERATIONS,
        num_iterations=1,
        beta=0.0,
        # Generation
        max_prompt_length=256,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=0.7,
        # Schedule
        max_steps=MAX_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=1,
        report_to="none",
    )

    # ── Train ──
    print("Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, play_reward],
        args=training_args,
        train_dataset=dataset,
        callbacks=[reward_logger],
    )

    trainer.train()

    # ── Save ──
    print("Saving LoRA adapter...")
    lora_path = os.path.join(OUTPUT_DIR, "lora_adapter")
    model.save_lora(lora_path)
    print(f"LoRA adapter saved to {lora_path}")

    # Save merged model (optional, for deployment)
    merged_path = os.path.join(OUTPUT_DIR, "merged_model")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {merged_path}")

    # Save reward curves
    reward_logger.save(OUTPUT_DIR)

    print("\nTraining complete!")
    print(f"  Checkpoints: {OUTPUT_DIR}/checkpoint-*/")
    print(f"  LoRA adapter: {lora_path}/")
    print(f"  Merged model: {merged_path}/")
    print(f"  Reward curve: {OUTPUT_DIR}/reward_curve.png")


if __name__ == "__main__":
    main()
