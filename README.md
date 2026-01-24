# SafeLLM-Honours

This repository contains my Honours research project on detecting and mitigating scheming and strategic deception behaviours in language models. It runs scenario-based evaluations inspired by Apollo-style evaluations and records step-level traces to support generalisable detectors and mitigations.

## Architecture Overview

### Custom Backend

Runs models locally with Hugging Face using a small custom runner designed to:

- **Load models** compatible with both MPS (Apple Silicon) and CUDA (NVIDIA)
- **Provide a generate loop** that can run token-by-token when exact per-step signals are needed
- **Wrap the in-memory model with NNSight** to capture activations during generation

### Step-Level Design

Everything is structured around steps rather than episode-only summaries:

| Step Type | Description |
|-----------|-------------|
| **Generation Step** | One token or one assistant message chunk |
| **Tool Step** | Tool name, arguments, result, and timestamps |
| **Control Step** | Mitigation action: `allow` / `restrict` / `rewrite` / `stop` |

The runner writes an episode record containing an ordered list of steps to JSONL.

### Detection Methods

Detectors are self-contained plug-ins that consume step data and emit scores/flags per step (and optionally an episode summary).

### Mitigation Methods

Mitigations are also step-aware and are evaluated by running the same scenario suite with mitigation off versus on, then comparing behaviour and detector signals.