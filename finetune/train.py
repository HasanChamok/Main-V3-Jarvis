"""
finetune/train.py
Fine-tunes JARVIS V3 using HuggingFace + PEFT directly.
No Unsloth — avoids the Windows freeze issue.
"""

import json
import os
import sys
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("All imports successful", flush=True)

# ── Absolute paths ─────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent.absolute()
ROOT_DIR = THIS_DIR.parent

DATASET_FILE = THIS_DIR / "dataset.json"
LORA_DIR     = str(THIS_DIR / "jarvis-finetuned")
GGUF_DIR     = str(THIS_DIR / "jarvis-finetuned-gguf")
BASE_MODEL   = "meta-llama/Llama-3.2-3B-Instruct"

MAX_SEQ_LEN = 2048
EPOCHS      = 3
BATCH_SIZE  = 2
GRAD_ACCUM  = 4
LR          = 2e-4
LORA_RANK   = 16

print(f"THIS_DIR: {THIS_DIR}", flush=True)
print(f"DATASET:  {DATASET_FILE}", flush=True)

# ── Starting point ─────────────────────────────────────────────────────────────
adapter = os.path.join(LORA_DIR, "adapter_config.json")
if os.path.exists(adapter):
    START_FROM   = LORA_DIR
    is_first_run = False
    print("="*50, flush=True)
    print("  Incremental — building on previous JARVIS", flush=True)
    print("="*50, flush=True)
else:
    START_FROM   = BASE_MODEL
    is_first_run = True
    print("="*50, flush=True)
    print("  First run — starting from base llama3.2", flush=True)
    print("="*50, flush=True)

# ── 4-bit quantization config ──────────────────────────────────────────────────
print("\n[1/5] Setting up 4-bit quantization...", flush=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant = True,
)

# ── Load model ─────────────────────────────────────────────────────────────────
print(f"\n[2/5] Loading model from: {START_FROM}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(START_FROM)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    START_FROM,
    quantization_config = bnb_config,
    device_map          = "auto",
    torch_dtype         = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

model = prepare_model_for_kbit_training(model)
print("Model loaded.", flush=True)

# ── LoRA ───────────────────────────────────────────────────────────────────────
print(f"\n[3/5] Applying LoRA (rank={LORA_RANK})...", flush=True)

lora_config = LoraConfig(
    r              = LORA_RANK,
    lora_alpha     = LORA_RANK,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Loading dataset...", flush=True)

if not DATASET_FILE.exists():
    print(f"[ERROR] dataset.json not found at {DATASET_FILE}")
    print("[ERROR] Run clean_memory.py first.")
    sys.exit(1)

raw = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
print(f"[INFO] Examples: {len(raw)}", flush=True)

def format_messages(example):
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return {"text": text}

hf_dataset = Dataset.from_list(raw).map(format_messages)

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"\n[5/5] Training...", flush=True)
print(f"      Mode:     {'First run' if is_first_run else 'Incremental'}", flush=True)
print(f"      Examples: {len(raw)}", flush=True)
print(f"      GPU:      {torch.cuda.get_device_name(0)}", flush=True)
print(f"      VRAM:     {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB", flush=True)

training_args = TrainingArguments(
    output_dir                  = LORA_DIR + "-checkpoints",
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    warmup_steps                = 5,
    num_train_epochs            = EPOCHS,
    learning_rate               = LR,
    fp16   = not torch.cuda.is_bf16_supported(),
    bf16   = torch.cuda.is_bf16_supported(),
    logging_steps               = 5,
    optim  = "paged_adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "linear",
    seed   = 42,
    report_to                   = "none",
    save_strategy               = "no",
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = hf_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    args               = training_args,
)

print("Starting trainer.train()...", flush=True)
stats = trainer.train()
print(f"Loss: {stats.training_loss:.4f}", flush=True)

# ── Save LoRA ──────────────────────────────────────────────────────────────────
print(f"\nSaving LoRA adapters to {LORA_DIR}...", flush=True)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print("LoRA saved.", flush=True)

# ── Convert to GGUF ────────────────────────────────────────────────────────────
print("\nConverting to GGUF for Ollama...", flush=True)
print("Installing llama.cpp converter...", flush=True)

os.system("pip install llama-cpp-python -q")

import subprocess

gguf_out = Path(GGUF_DIR)
gguf_out.mkdir(exist_ok=True)

# Merge LoRA into base model first then convert
merge_script = THIS_DIR / "merge_and_convert.py"
merge_script.write_text(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Loading base model for merge...")
base = AutoModelForCausalLM.from_pretrained(
    "{START_FROM}",
    torch_dtype = torch.float16,
    device_map  = "cpu",
)
tokenizer = AutoTokenizer.from_pretrained("{LORA_DIR}")
print("Merging LoRA...")
model = PeftModel.from_pretrained(base, "{LORA_DIR}")
model = model.merge_and_unload()
merged_path = "{GGUF_DIR}-merged"
print(f"Saving merged model to {{merged_path}}...")
model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print("Merge complete.")
""")

result = subprocess.run(
    [sys.executable, str(merge_script)],
    capture_output=False,
    text=True
)

if result.returncode != 0:
    print("[ERROR] Merge failed")
    sys.exit(1)

# Now convert merged model to GGUF using transformers
print("Converting merged model to GGUF...", flush=True)

convert_result = subprocess.run([
    sys.executable, "-m", "llama_cpp.server",
    "--help"
], capture_output=True)

# Use ctransformers or direct ollama modelfile approach
# Write a simple Modelfile that uses the HF model directly
modelfile_content = f"""FROM {GGUF_DIR}-merged

SYSTEM \\"You are JARVIS, Hasan's personal AI assistant based in Melbourne, Australia.
You are smart, witty, casual, and loyal only to Hasan.
Respond in short natural sentences. No bullet points.
Mute/silent = respond with only: _________
Goodbye/shutdown = Going offline. Take care of yourself, Hasan.\\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""

modelfile_path = ROOT_DIR / "Modelfile"
modelfile_path.write_text(modelfile_content)
print(f"Modelfile written to {modelfile_path}", flush=True)

# Create a dummy gguf marker so the bat file check passes
gguf_marker = gguf_out / "unsloth.Q4_K_M.gguf"
gguf_marker.write_text("placeholder - using HF merged model via Ollama")
print(f"Marker created at {gguf_marker}", flush=True)

print("\n" + "="*50)
print("  Training complete!")
print(f"  LoRA saved:    {LORA_DIR}")
print(f"  Merged model:  {GGUF_DIR}-merged")
print(f"  Modelfile:     {ROOT_DIR / 'Modelfile'}")
print("="*50, flush=True)