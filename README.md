# Sequential LLM Distillation

> [!IMPORTANT]
> **Archived: superseded by [ELI500Model-Sequential-Distillation](https://github.com/juyal-devashish/ELI500Model-Sequential-Distillation),** which holds the final code, the report and the results. This repository is the earlier KPOD-first version from the EECS 498 proposal stage, kept for history.


This repository explores distillation and compression of Large Language Models (LLMs) — techniques that aim to transfer reasoning and knowledge from large, high-capacity models (teachers) into smaller, efficient models (students).  
The goal is to retain emergent capabilities (reasoning, in-context learning, chain-of-thought) while significantly reducing inference cost.

---

## Summary

This project studies the trade-offs between model size, reasoning depth, and performance in LLMs through systematic distillation experiments.  
We implement and evaluate teacher–student frameworks, combining classical and modern methods such as:

-  **Knowledge Distillation (KD)** — soft label transfer from teacher to student  
-  **Sequence-level Distillation** — transferring behavior across generation steps  
-  **Representation & Feature Matching** — aligning internal states (hidden layers, attention maps)  
-  **Loss Design** — exploring KL, cosine, and contrastive objectives for improved generalization  

Our experiments focus on whether small distilled models can preserve reasoning behavior seen in larger base models (e.g., GPT-4, Llama-3).

---

## Research Objectives

- Quantify reasoning degradation during compression
- Compare different distillation strategies across benchmarks
- Analyze attention and hidden state alignment
- Evaluate efficiency vs. capability trade-offs

