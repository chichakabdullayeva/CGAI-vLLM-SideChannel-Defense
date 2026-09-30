# CGAI: Counter-Guided Adaptive Isolation for Multi-Tenant LLM Serving

A discrete-event Proof-of-Concept (PoC) simulation evaluating defense mechanisms against side-channel attack vectors in multi-tenant LLM serving infrastructures (e.g., vLLM).

## Key Empirical Findings

- **Tier A (HMAC Key Salting):** Completely reduces single-shot probing attack success rate (ASR) from 100% to chance level (~0.0%).
- **Boundary Salting Efficiency:** Preserves cross-tenant preamble reuse, adding **0.00% overhead** under standard cache capacities.
- **Cache Pressure & False Positives:** Under standard cache limits (12,000 blocks), the refined Tier B watcher yields **0.00% overhead** on template-stable classifier workloads. Under heavy cache pressure (4,000 blocks), cache eviction increases false-positive jitter, resulting in a **2.76% mean TTFT overhead**.
- **Defense-in-Depth:** Demonstrates that jitter alone (Tier C) leaves a 30–47% ASR, confirming that cryptographic isolation (Tier A) is essential.

## Referenced Preprints
- SLAC (arXiv:2608.09225)
- KVGov (arXiv:2609.06853)
- KV-Cache Contention Study (arXiv:2608.09075)

## How to Run

```bash
python cgai_sim.py --seed 7 --requests 8000 --jitter-ms 120 --cache-blocks 12000
