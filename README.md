# NEURON

**Neural Encoding with Unified Recurrent Optimized Network**

> Hybrid memory architecture combining exact recall with infinite-capacity fuzzy understanding for LLMs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/ertugrulakben/neuron/actions/workflows/ci.yml/badge.svg)](https://github.com/ertugrulakben/neuron/actions)

---

## The Problem

Current LLM memory systems force a trade-off:

| Approach | Pros | Cons |
|----------|------|------|
| **Full Context** | 100% accuracy | O(n²) cost, context limits |
| **RAG** | Exact retrieval | Fragmented, no semantic understanding |
| **Summarization** | Infinite capacity | Loses critical details |

**Example failure:**
- You tell the AI: "The meeting is at 3:45 PM on March 15th"
- After 100 more messages, you ask: "When is the meeting?"
- RAG might miss it, summarization might say "afternoon in March"

---

## Our Solution: Dual-Track Memory

NEURON routes information to the right memory type:

```
INPUT → [Importance Router] → Critical? → CRYSTAL (exact KV store)
                            → General?  → MORPH (neural state matrix)
                            → Surprising? → BOTH (dual write)
```

### Crystal Memory (Symbolic)
- **What:** Key-value store with semantic indexing
- **Stores:** Names, dates, numbers, code, specific facts
- **Retrieval:** High-fidelity semantic matching (>95% cosine threshold)
- **Capacity:** ~10K items (with importance-weighted LRU eviction)

### Morph Layer (Neural)
- **What:** Continuous state matrix updated by HyperNetwork
- **Stores:** Context, relationships, tone, general understanding
- **Retrieval:** Approximate but always available
- **Capacity:** Bounded O(1) memory (rank² parameters, ~4K for rank=64)

---

## Key Innovations

### 1. Temporal Belief Graph (TBG) — NOVEL CONTRIBUTION

**No existing memory system handles contradictions explicitly.** TBG tracks:

```python
import torch
from neuron import NEURON

memory = NEURON(d_model=512)

# Store a fact
key = torch.randn(512)
memory.absorb(key, text="Meeting at 3pm")  # Stored with confidence 0.7

# Later, contradictory info arrives
key2 = key + torch.randn(512) * 0.05  # Same topic
stats = memory.absorb(key2, text="Meeting at 4pm")  # Contradiction detected!

print(stats["contradictions"])
# [{"type": "value", "existing_value": "Meeting at 3pm", ...}]
print(stats["belief_confidence"])
# 0.4  — reduced due to conflict

# Query unresolved contradictions
memory.get_contradictions()  # Returns unresolved conflicts
```

**What TBG does:**
- **Contradiction Detection:** Identifies value conflicts, negations, temporal conflicts
- **Belief Evolution:** Tracks confidence over time (corroboration ↑, contradiction ↓)
- **Evidence Aggregation:** Multiple sources saying same thing = higher confidence

### 2. Surprise-Modulated Type Routing (SMTR)

Unlike Titans (Google, 2025) which uses surprise for write/skip decisions, NEURON uses surprise to decide **where** to route:

```python
routing_decision = f(importance, surprise, embedding)

if critical and surprising:
    write_crystal(priority=HIGH)
elif critical:
    write_crystal(priority=NORMAL)
elif surprising:
    write_both()  # Dual write - hedge our bets
else:
    write_morph()
```

### 3. Horizontal Cross-Memory Consolidation (HCMC)

Unlike TiMem (2026) which consolidates vertically (raw→abstract), NEURON consolidates horizontally between memory types:

```
Crystal patterns → Guide Morph organization
Morph context → Improve Crystal retrieval ranking
```

### 4. Dual Representation Paradigm

Unlike BudgetMem (2025) where both memories are text-based, NEURON uses fundamentally different representations:
- **Crystal:** Discrete symbolic (key-value pairs)
- **Morph:** Continuous neural (weight matrices)

---

## Quick Start

### Installation

```bash
# From PyPI
pip install neuron-memory

# From source
git clone https://github.com/ertugrulakben/neuron.git
cd neuron
pip install -e .
```

### Basic Usage

```python
import torch
from neuron import NEURON, NeuronConfig

# Initialize with config (recommended)
config = NeuronConfig(d_model=512, crystal_size=5000, morph_rank=64)
memory = NEURON(config=config)

# Or with keyword arguments
memory = NEURON(d_model=512, crystal_size=5000)

# Create embeddings (from your encoder of choice)
embedding = torch.randn(512)

# Absorb information — router decides where to store
stats = memory.absorb(embedding, text="The project budget is $75,000")
print(stats["decision"])  # "crystal" — detected money pattern

stats = memory.absorb(torch.randn(512), text="We discussed the approach")
print(stats["decision"])  # "morph" — general information

# Query the memory
query = torch.randn(512)
results = memory.query(query, top_k=5)

print(results["crystal_results"])  # Exact facts from Crystal
print(results["morph_context"])    # Fuzzy context from Morph
print(results["fused_output"])     # Adaptively fused output
```

See [examples/](examples/) for more detailed usage.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           NEURON                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   INPUT ──► ENCODER ──► IMPORTANCE ROUTER (SMTR)                 │
│                              │                                    │
│                    ┌─────────┴─────────┐                         │
│                    │                   │                          │
│                    ▼                   ▼                          │
│            ┌──────────────┐   ┌──────────────┐                   │
│            │   CRYSTAL    │   │    MORPH     │                   │
│            │   MEMORY     │◄─►│    LAYER     │                   │
│            │  (Symbolic)  │   │   (Neural)   │                   │
│            └──────┬───────┘   └──────┬───────┘                   │
│                   │                  │                            │
│            ┌──────┴───────┐         │                            │
│            │  TEMPORAL    │         │                            │
│            │ BELIEF GRAPH │         │                            │
│            │   (TBG)      │         │                            │
│            └──────┬───────┘         │                            │
│                   │                  │                            │
│                   └────────┬─────────┘                           │
│                            │                                      │
│                            ▼                                      │
│                    ┌──────────────┐                               │
│                    │    FUSION    │                               │
│                    └──────┬───────┘                               │
│                           │                                       │
│                           ▼                                       │
│                        OUTPUT                                     │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Benchmarks

### Internal Test Results (Large Sample)

| Benchmark | n | Score | Details |
|-----------|---|-------|---------|
| **Crystal Recall Accuracy** | 500 | 100.00% | 500/500 exact retrievals |
| **Routing Accuracy (SMTR)** | 1000 | 99.90% | 999/1000 correct routing decisions |
| **Memory Efficiency** | 2000 | 100.00% | Crystal bounded at max, Morph O(1) |
| **Context Retention** | 100 | 95.66% | Centroid alignment with input distribution |
| **Overall Accuracy** | - | **98.52%** | Averaged across accuracy metrics |

### Speed Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Absorb | ~4 ms/item | Store + route decision |
| Query | ~1.4 ms/item | Crystal + Morph retrieval + fusion |

### Architectural Comparison (Not Empirically Validated)

| Property | Full Context | RAG | **NEURON** |
|----------|-------------|-----|------------|
| Memory Growth | O(n) tokens | O(k) chunks | **O(1)** fixed |
| Routing | None | Similarity | **SMTR** (importance x surprise) |
| Exact Recall | Yes | Partial | **Hybrid** (Crystal) |
| Context Retention | Yes | Limited | **Yes** (Morph) |

> **Note:** Above is architectural comparison, not empirical benchmark. Validated comparisons require testing on standard benchmarks (LoCoMo, RAGBench, etc.).

*Internal benchmarks (n=500-2000) on synthetic data. External benchmark validation pending.*

---

## Comparison with Related Work

| System | Memory Type | Routing | Consolidation | Contradiction Handling |
|--------|-------------|---------|---------------|------------------------|
| **Titans** (Google, 2025) | Single neural | Surprise→Write | None | None |
| **BudgetMem** (2025) | Dual text | Salience scoring | None | None |
| **TiMem** (2026) | Hierarchical | Complexity-based | Vertical | None |
| **Mem0** (2025) | Graph-based | Dense retrieval | DB updates | Overwrites silently |
| **NEURON** (Ours) | **Symbolic + Neural** | **SMTR** | **Horizontal** | **TBG** (explicit detection) |

---

## Citation

If you use NEURON in your research, please cite:

```bibtex
@misc{neuron2026,
  title={NEURON: Surprise-Modulated Routing Between Symbolic and Neural Memory for Infinite Context LLMs},
  author={Akben, Ertugrul},
  year={2026},
  url={https://github.com/ertugrulakben/neuron}
}
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Titans team at Google for surprise-gated memory insights
- TiMem authors for hierarchical consolidation concepts
- BudgetMem team for dual-memory inspiration

---

**Built with curiosity by [Ertugrul Akben](https://ertugrulakben.com)**
