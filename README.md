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
```

📊 Performance Benchmark: Fine-Tune vs. Base
When tested on sequential logic puzzles (like the classic 5L and 3L water jug problem), the reasoning fine-tune effectively utilizes the <think> block to simulate the state machine and reaches the correct conclusion. In contrast, the base model hits context constraints and fails to properly format or conclude its output.

Benchmarking Srikri7 FT...
Benchmarking Qwen Base...

================================================================================
🧠 MODEL: Srikri7 Reasoning FT
================================================================================
THINKING PROCESS:
----------------------------------------
The user wants me to solve a puzzle: "You have a 5-liter jug, a 3-liter jug, and an unlimited supply of water. You need to measure exactly 4 liters of water. How do you do it?"

I need to find a sequence of steps that allows me to measure exactly 4 liters of water using the given jugs.

Let's think about this problem. I have two jugs with capacities 5L and 3L. I can fill them from an infinite source of water. I can also empty them into a sink or discard the water.

Goal: Measure exactly 4 liters.

Possible approach:
- Fill the 5L jug
- Pour from 5L jug into 3L jug until 3L jug is full
- The 5L jug now has 2L left
- Empty the 3L jug
- Pour the 2L from 5L jug into 3L jug
- Fill the 5L jug again
- Pour from 5L jug into 3L jug until 3L jug is full (it will take 1L more)
- The 5L jug now has 4L left
- This 4L is what we want!

Let me verify this:
1. Fill 5L jug (has 5L)
2. Pour 5L jug into 3L jug (3L jug is full, 5L jug has 2L left)
3. Empty 3L jug
4. Pour remaining 2L from 5L jug into 3L jug (3L jug is full, 5L jug has 0L left)
5. Fill 5L jug (has 5L)
6. Pour from 5L jug into 3L jug until 3L jug is full (3L jug was full, so pour 1L from 5L jug into 3L jug, leaving 4L in 5L jug)
7. We have 4L in the 5L jug!

This works!

Let me write down the solution clearly:
1. Fill the 5-liter jug
2. Pour the water from the 5-liter jug into the 3-liter jug
3. Empty the 3-liter jug
4. Pour the remaining water from the 5-liter jug into the 3-liter jug
5. Fill the 5-liter jug again
6. Pour the water from the 5-liter jug into the 3-liter jug until the 3-liter jug is full
7. The 5-liter jug now contains exactly 4 liters of water

This is the correct solution.
----------------------------------------
FINAL ANSWER:
Fill the 5-liter jug, pour into the 3-liter jug, empty the 3-liter jug, pour the remaining 2 liters from the 5-liter jug into the 3-liter jug, fill the 5-liter jug again, pour into the 3-liter jug until full, leaving 4 liters in the 5-liter jug.
----------------------------------------
⏱️ METRICS: 635 tokens generated in 59.56s (10.66 tok/s)

================================================================================
🧠 MODEL: Qwen Base Model
================================================================================
THINKING PROCESS:
----------------------------------------
We need to solve the classic water jug problem: we have two jugs of capacities 5L and 3L, and unlimited water source. We want to measure exactly 4 liters in one of the jugs (or perhaps in either jug). The typical solution: fill the 5L jug, pour into 3L jug until full, leaving 2L in 5L jug. Empty the 3L jug, pour the 2L from 5L into 3L jug, then fill the 5L jug again, pour into 3L jug which already has 2L, so it can hold 1L more, leaving 4L in the 5L jug. So answer: steps.

But the user says "You have a 5-liter jug, a 3-liter jug, and an unlimited supply of water. You need to measure exactly 4 liters of water. How do you do it?" So we need to provide a step-by-step procedure.

We should put reasoning inside <think> and
----------------------------------------
FINAL ANSWER:
and
----------------------------------------
⏱️ METRICS: 215 tokens generated in 53.56s (4.01 tok/s)


