# SFT best practices for Qwen3-8B reasoning models

**LLaMA-Factory emerges as the optimal framework** for fine-tuning Qwen3-8B-Instruct on Operations Research debugging tasks with structured `<think>` reasoning outputs. With native Qwen3 template support, DeepSpeed ZeRO integration, and 64.8k GitHub stars backing its academic credibility (ACL 2024 paper with 1000+ citations), it provides the most robust path to reproducible results for NeurIPS 2026. Your 2×A800 80GB setup with **160GB total VRAM** offers substantial headroom—expect training to complete in **under 1 hour** rather than the 24-hour budget, leaving room for extensive hyperparameter exploration.

The critical technical insight for your use case: Qwen3's `<think>` tags are **regular text tokens, not special tokens**, meaning the tokenizer handles them naturally without modification. Use the `qwen3` template for reasoning mode, pair LoRA rank 32-64 with ZeRO Stage 2 (not Stage 3, which breaks LoRA gradient flow), and maintain a conservative **learning rate of 2e-4** to prevent catastrophic forgetting of the base model's reasoning capabilities.

---

## Framework selection favors LLaMA-Factory's Qwen3 integration

The framework landscape for Qwen fine-tuning has consolidated around five major options, each with distinct strengths. LLaMA-Factory stands out for reasoning tasks due to its explicit template system that handles thinking mode correctly.

| Framework | Qwen3 Support | Multi-GPU | `<think>` Handling | Best For |
|-----------|--------------|-----------|-------------------|----------|
| **LLaMA-Factory** | Day 0 (Apr 2025) | DeepSpeed ZeRO-2/3 | `qwen3` / `qwen3_nothink` templates | Your use case |
| ms-swift | Official Alibaba | Megatron, DeepSpeed | Native support | MoE models |
| Axolotl | Oct 2025 | ND Parallelism | Via tokenizer | Long-context |
| TRL | Native | Via Accelerate | Manual config | Research flexibility |
| Unsloth | Day 0 | **Not in OSS** | `/think` prompts | Single GPU only |

**LLaMA-Factory wins** for three reasons. First, its `template: qwen3` configuration generates `<think>...</think>` blocks automatically while `qwen3_nothink` suppresses them—matching your output format requirements precisely. Second, the framework includes pre-built DeepSpeed configs optimized for your A800 setup. Third, its documentation explicitly covers Qwen3 reasoning mode training, reducing configuration guesswork. The critical limitation of Unsloth—no multi-GPU support in the open-source version—eliminates it despite its 2× speed advantage.

---

## Optimal LoRA configuration balances capacity and overfitting risk

For an 8B model with **5,216 training samples**, LoRA rank selection becomes a key decision point. Research indicates rank 32 provides the optimal balance—sufficient capacity for domain-specific learning without the overfitting risk that comes with higher ranks (128+) on small datasets.

The complete LoRA configuration targets all linear layers in both attention and MLP blocks. Qwen3's architecture uses **7 target modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` for attention, plus `gate_proj`, `up_proj`, `down_proj` for the MLP. Targeting all layers significantly outperforms attention-only approaches according to the QLoRA paper.

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                              # Rank: 32 for 8B models
    lora_alpha=64,                     # Alpha: 2× rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"       # MLP
    ],
    lora_dropout=0.05,                 # Light regularization
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Why ZeRO Stage 2, not Stage 3**: DeepSpeed ZeRO-3 partitions model parameters across GPUs, which breaks gradient flow when LoRA adapters are added. ZeRO-2 partitions only optimizer states and gradients—sufficient for your setup since LoRA training parameters are tiny (~100MB) and the 8B model base (~16GB in bf16) fits comfortably on each GPU.

---

## Training hyperparameters for sub-hour convergence

Your hardware configuration supports aggressive batching. With **160GB total VRAM** and short sequences (input 200-500 + output 50-150 = ~650 tokens max), you can run **per-device batch size 4** with **gradient accumulation 4**, yielding an effective batch size of **32** across both GPUs.

| Parameter | Recommended Value | Rationale |
|-----------|------------------|-----------|
| Per-device batch size | 4 | Conservative for stability |
| Gradient accumulation | 4 | Effective batch = 32 |
| Learning rate | **2e-4** | Standard for LoRA |
| Warmup ratio | 0.05 | ~25 steps for 469 total |
| Scheduler | Cosine | Smooth decay |
| Epochs | **2-3** | Overfitting risk above 4 |
| Max seq length | 650 | Input + output buffer |

The training math: 5,216 samples ÷ 32 effective batch = **163 steps per epoch**. Three epochs total **489 steps**. At roughly 3-5 seconds per step on A800s, total training time is approximately **25-40 minutes**—well under budget, enabling multiple ablation runs.

```yaml
# ds_config_zero2.json
{
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "none"},
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0
}
```

---

## Data format conversion preserves reasoning structure

Converting Alpaca-style data to Qwen chat format requires wrapping instruction/input pairs into the ChatML message structure while ensuring `<think>` tags remain intact in assistant responses. The key insight: **disable Qwen3's native thinking mode** (`enable_thinking=False`) when applying the chat template, since you provide custom `<think>` tags in your training data.

```python
import json

def convert_to_qwen_chat(alpaca_data: list, system_prompt: str) -> list:
    """Convert Alpaca format to Qwen chat format with reasoning tags."""
    converted = []
    
    for item in alpaca_data:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\n\n{item['input']}"
        
        # Ensure output maintains <think> structure
        output = item["output"]
        if "<think>" not in output:
            # Add tags if missing (adjust based on your data)
            output = f"<think>\n{output}\n</think>\n\nACTION_REQUIRED"
        
        converted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
        })
    
    return converted

# System prompt for OR debugging
SYSTEM_PROMPT = """You are an Operations Research debugging assistant.
Analyze problems step-by-step in <think> tags, then output the action.

Format:
<think>
Your detailed reasoning...
</think>

ACTION_TYPE(arguments)"""

# Convert and save
with open("alpaca_data.json") as f:
    data = json.load(f)

qwen_data = convert_to_qwen_chat(data, SYSTEM_PROMPT)

with open("train_data.jsonl", "w") as f:
    for item in qwen_data:
        f.write(json.dumps(item) + "\n")
```

**Special token handling**: The `<think>` and `</think>` tags use XML notation and are tokenized as regular text—they're learned patterns, not pre-defined special tokens like `<|im_start|>`. No tokenizer modification is needed. However, always set `eos_token="<|im_end|>"` explicitly to ensure proper generation termination.

---

## Complete training script for reproducible results

The following script combines all recommendations into a production-ready training pipeline using LLaMA-Factory's configuration format.

```yaml
# qwen3_or_debug_lora.yaml
model_name_or_path: Qwen/Qwen3-8B-Instruct
template: qwen3                    # Enables thinking mode
finetuning_type: lora

# LoRA Configuration
lora_rank: 32
lora_alpha: 64
lora_target: all                   # All 7 linear layers
lora_dropout: 0.05

# Dataset
dataset: or_debug_train            # Define in dataset_info.json
cutoff_len: 650

# Training
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.01

# Precision and Memory
bf16: true
flash_attn: fa2
gradient_checkpointing: true
deepspeed: ds_config_zero2.json

# Logging and Checkpointing
logging_steps: 10
save_steps: 100
save_total_limit: 3
eval_steps: 100
evaluation_strategy: steps

# Reproducibility
seed: 42
```

**Launch command for 2×A800**:
```bash
CUDA_VISIBLE_DEVICES=0,1 llamafactory-cli train qwen3_or_debug_lora.yaml
```

For TRL-based training with more control:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model with flash attention
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Apply LoRA
model = get_peft_model(model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

# Format function
def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False  # We provide our own <think> tags
    )

dataset = load_dataset("json", data_files="train_data.jsonl", split="train")
dataset = dataset.map(lambda x: {"text": format_chat(x)})

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=SFTConfig(
        output_dir="./qwen3-or-debug",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        deepspeed="ds_config_zero2.json",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        seed=42,
    ),
    max_seq_length=650,
)

trainer.train()
trainer.save_model()
```

---

## Evaluation requires format compliance beyond loss metrics

Training loss alone is insufficient for structured reasoning outputs. Your evaluation framework should measure three dimensions: **format compliance** (does output match `<think>...</think>` followed by `ACTION_TYPE(args)`), **action accuracy** (does the predicted action match ground truth), and **reasoning quality** (are intermediate steps logical and faithful).

```python
import re
from typing import Dict

def evaluate_structured_output(prediction: str, gold_action: str) -> Dict:
    """Comprehensive evaluation for OR debugging outputs."""
    
    # Check format compliance
    think_pattern = r'<think>(.*?)</think>'
    action_pattern = r'([A-Z_]+)\((.*?)\)'
    
    think_match = re.search(think_pattern, prediction, re.DOTALL)
    remaining = re.sub(think_pattern, '', prediction, re.DOTALL).strip()
    action_match = re.search(action_pattern, remaining)
    
    format_ok = think_match is not None and action_match is not None
    
    # Check action accuracy
    action_correct = False
    if action_match:
        pred_action = f"{action_match.group(1)}({action_match.group(2)})"
        action_correct = pred_action.strip() == gold_action.strip()
    
    return {
        "format_compliance": format_ok,
        "action_exact_match": action_correct,
        "has_think_tags": think_match is not None,
        "reasoning_length": len(think_match.group(1)) if think_match else 0
    }
```

**Validation split recommendation**: Use 80/10/10 train/val/test with stratification by action type and problem difficulty. For academic rigor, run 5-fold cross-validation during development and report mean ± std across seeds (42, 123, 456) in the paper.

**Chain-of-thought faithfulness testing**: Truncate reasoning at 25%, 50%, 75% and check if the final action changes. If predictions remain identical despite truncation, the reasoning may be unfaithful—a red flag for reviewers.

---

## Critical pitfalls to avoid during training

Several failure modes frequently derail reasoning model fine-tuning. The most dangerous is **multi-epoch overfitting**—research by Sebastian Raschka shows that instruction fine-tuning often degrades after 2 epochs on small datasets. Monitor validation loss closely and stop training if it increases while training loss continues to decrease.

**Tag balancing failures** occur when models learn to generate long reasoning but truncate at max_length before producing the closing `</think>` tag. Your short output sequences (50-150 tokens) minimize this risk, but still verify tag balance in validation outputs.

**Prompt format mismatch** destroys instruction-following capability. Always use `tokenizer.apply_chat_template()` rather than manual string formatting—the Qwen3 chat template includes subtle whitespace and delimiter patterns that affect generation quality.

**Mathematical reasoning degradation** is a documented side effect of domain-specific fine-tuning. LoRA mitigates this by freezing base weights, but consider including 10-20% of general mathematical reasoning examples in your training mix if OR debugging requires numerical computation.

For inference, avoid greedy decoding—Qwen3 produces repetitive outputs with temperature=0. Use **temperature=0.6** and **top_p=0.95** for thinking mode generation.

---

## Evaluation checklist before deployment

**Pre-training verification:**
- [ ] Data converted to Qwen chat format with consistent `<think>` structure
- [ ] Validation split stratified by action type
- [ ] Chat template applied with `enable_thinking=False`
- [ ] All random seeds documented (42, 123, 456)

**Training monitoring:**
- [ ] Validation loss tracked alongside format compliance rate
- [ ] Checkpoints saved every 100 steps (LoRA adapters only, ~50MB each)
- [ ] Early stopping armed with patience=5

**Post-training evaluation:**
- [ ] Format compliance rate >95%
- [ ] Action accuracy compared to ground truth
- [ ] Reasoning faithfulness via truncation tests
- [ ] Human evaluation on 50-100 representative samples
- [ ] Cross-validation results with confidence intervals

**Reproducibility for NeurIPS:**
- [ ] All hyperparameters in methods table
- [ ] Hardware config: 2×A800 80GB, training time
- [ ] Code released on GitHub with requirements.txt
- [ ] LoRA adapters uploaded to Hugging Face Hub

---

## Conclusion

LLaMA-Factory with LoRA rank 32 and DeepSpeed ZeRO Stage 2 provides the most robust configuration for fine-tuning Qwen3-8B-Instruct on your Operations Research debugging task. The combination of explicit Qwen3 template support, native multi-GPU integration, and academic credibility makes it the clear choice over alternatives.

Key technical decisions that differ from common defaults: use ZeRO-2 (not ZeRO-3) with LoRA to preserve gradient flow; set `enable_thinking=False` in chat template application since you provide custom tags; limit training to 2-3 epochs maximum to prevent overfitting on 5,216 samples. Your training will complete in under an hour, leaving substantial budget for hyperparameter sweeps across LoRA ranks (16, 32, 64) and learning rates (1e-4, 2e-4, 5e-4).

The format compliance evaluation framework—measuring tag balance, action accuracy, and reasoning faithfulness—provides metrics beyond loss that reviewers expect for reasoning model papers. Combined with proper seed documentation and cross-validation, this approach meets NeurIPS reproducibility standards.