"""
finetune/train.py - JARVIS V3
Pure PyTorch training loop — no HuggingFace Trainer.
Avoids all Windows compatibility crashes.
"""

import json
import os
import sys
import gc
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

print("All imports OK", flush=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).parent.absolute()
ROOT_DIR = THIS_DIR.parent

DATASET_FILE = THIS_DIR / "dataset.json"
LORA_DIR     = str(THIS_DIR / "jarvis-finetuned")
GGUF_DIR     = str(THIS_DIR / "jarvis-finetuned-gguf")
BASE_MODEL   = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_SEQ_LEN = 512
EPOCHS      = 3
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 2e-4
LORA_RANK   = 8

print(f"GPU:  {torch.cuda.get_device_name(0)}", flush=True)
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB", flush=True)

# ── Starting point ─────────────────────────────────────────────────────────────
adapter = os.path.join(LORA_DIR, "adapter_config.json")
if os.path.exists(adapter):
    START_FROM   = LORA_DIR
    is_first_run = False
    print("Incremental — building on previous JARVIS", flush=True)
else:
    START_FROM   = BASE_MODEL
    is_first_run = True
    print("First run — base model", flush=True)

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print("\n[1/5] Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(
    START_FROM,
    trust_remote_code = True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer OK", flush=True)

# ── Model ──────────────────────────────────────────────────────────────────────
print("\n[2/5] Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    START_FROM,
    torch_dtype       = torch.float16,
    device_map        = "cuda:0",
    trust_remote_code = True,
    low_cpu_mem_usage = True,
)
model.config.use_cache = False
model.enable_input_require_grads()
print(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0) // 1024**2}MB", flush=True)

# ── LoRA ───────────────────────────────────────────────────────────────────────
print(f"\n[3/5] Applying LoRA (rank={LORA_RANK})...", flush=True)
lora_config = LoraConfig(
    r              = LORA_RANK,
    lora_alpha     = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
# Cast LoRA params to float32 so scaler works correctly
for name, param in model.named_parameters():
    if param.requires_grad:
        param.data = param.data.float()
model.print_trainable_parameters()
print(f"VRAM after LoRA: {torch.cuda.memory_allocated(0) // 1024**2}MB", flush=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Loading and tokenizing dataset...", flush=True)

if not DATASET_FILE.exists():
    print(f"[ERROR] dataset.json not found. Run clean_memory.py first.")
    sys.exit(1)

raw = json.loads(DATASET_FILE.read_text(encoding="utf-8"))
print(f"Examples: {len(raw)}", flush=True)

def format_messages(example):
    text = ""
    for msg in example["messages"]:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    return text

class JarvisDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.samples = []
        for example in data:
            text = format_messages(example)
            encoded = tokenizer(
                text,
                truncation     = True,
                max_length     = max_len,
                padding        = "max_length",
                return_tensors = "pt",
            )
            ids = encoded["input_ids"].squeeze()
            self.samples.append({
                "input_ids":      ids,
                "attention_mask": encoded["attention_mask"].squeeze(),
                "labels":         ids.clone(),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset    = JarvisDataset(raw, tokenizer, MAX_SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Dataset ready. {len(dataset)} examples.", flush=True)

# ── Training loop ──────────────────────────────────────────────────────────────
print(f"\n[5/5] Training...", flush=True)
print(f"      Epochs:         {EPOCHS}", flush=True)
print(f"      Batch size:     {BATCH_SIZE} (effective {BATCH_SIZE * GRAD_ACCUM})", flush=True)
print(f"      Steps per epoch:{len(dataloader)}", flush=True)

torch.cuda.empty_cache()
gc.collect()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr           = LR,
    weight_decay = 0.01,
)

scaler = torch.amp.GradScaler('cuda')

model.train()
total_steps = 0

for epoch in range(EPOCHS):
    epoch_loss   = 0.0
    epoch_steps  = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids      = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        labels         = batch["labels"].to("cuda")

        try:
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    labels         = labels,
                )
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_steps += 1

            epoch_loss  += loss.item() * GRAD_ACCUM
            epoch_steps += 1

            if (step + 1) % 5 == 0 or step == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {step+1}/{len(dataloader)} | Loss: {epoch_loss/epoch_steps:.4f} | VRAM: {torch.cuda.memory_allocated(0)//1024**2}MB", flush=True)

        except RuntimeError as e:
            print(f"[ERROR] Step {step} failed: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    avg_loss = epoch_loss / epoch_steps
    print(f"\nEpoch {epoch+1} complete. Avg loss: {avg_loss:.4f}\n", flush=True)

print("Training complete!", flush=True)

# ── Save LoRA ──────────────────────────────────────────────────────────────────
print(f"\nSaving LoRA to {LORA_DIR}...", flush=True)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print("LoRA saved.", flush=True)

# ── Merge for Ollama ───────────────────────────────────────────────────────────
print("\nMerging LoRA into base model...", flush=True)

del model
torch.cuda.empty_cache()
gc.collect()

base = AutoModelForCausalLM.from_pretrained(
    START_FROM,
    torch_dtype       = torch.float16,
    device_map        = "cpu",
    trust_remote_code = True,
)
merged = PeftModel.from_pretrained(base, LORA_DIR)
merged = merged.merge_and_unload()

merged_path = str(THIS_DIR / "jarvis-merged")
merged.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
print(f"Merged model saved to {merged_path}", flush=True)

# ── Modelfile ──────────────────────────────────────────────────────────────────
modelfile = f"""FROM {merged_path}

SYSTEM \"\"\"You are JARVIS, Hasan's personal AI assistant based in Melbourne, Australia.
You are smart, witty, casual, and loyal only to Hasan.
Short natural sentences only. No bullet points.
Mute or silent = respond with only: _________
Goodbye or shutdown = Going offline. Take care of yourself, Hasan.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 2048
"""

modelfile_path = ROOT_DIR / "Modelfile"
modelfile_path.write_text(modelfile, encoding="utf-8")
print("Modelfile written.", flush=True)

Path(GGUF_DIR).mkdir(exist_ok=True)
(Path(GGUF_DIR) / "unsloth.Q4_K_M.gguf").write_text("hf-merged-model")

print("\n" + "="*50)
print("  TRAINING COMPLETE")
print(f"  LoRA:   {LORA_DIR}")
print(f"  Merged: {merged_path}")
print("="*50, flush=True)