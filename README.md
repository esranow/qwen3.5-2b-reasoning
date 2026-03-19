# 🧠 Qwen3.5-2B-Reasoning

**Qwen3.5-2B-Reasoning** is a high-density Reasoning model distilled from **Claude-4.6 Opus** trajectories. It brings structured, multi-step Chain-of-Thought (CoT) capabilities to a 2.2B parameter footprint, optimized specifically for **NVIDIA Tesla T4** hardware. By enforcing a strict `<think>` scaffold, it minimizes the logical drift common in small-scale models.

### 🚀 Key Features
- **Structured CoT**: Native [Understanding] → [Plan] → [Execution] → [Verification] flow.
- **Developer Native**: Built-in support for the `developer` role (no Jinja crashes).
- **Efficiency**: NF4 quantized for sub-4GB VRAM inference.
- **Distilled Logic**: Trained on 3,000+ high-quality reasoning trajectories.

---

### 💻 Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "Srikri7/qwen3.5-2b-reasoning"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Elite Reasoning System Prompt
messages = [
    {"role": "system", "content": "You are an elite reasoning system. Think step-by-step inside <think> tags."},
    {"role": "user", "content": "If I have 3 oranges today and I ate 1 yesterday, how many oranges do I have now?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs, 
    max_new_tokens=1024, 
    temperature=0.1,
    do_sample=True
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
