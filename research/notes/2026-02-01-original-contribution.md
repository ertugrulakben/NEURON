# NEURON - Gerçek Özgün Katkı Analizi

**Tarih:** 2026-02-01
**Durum:** Literatür taraması sonrası özgünlük belirleme

---

## Mevcut Çalışmalar (Artık Özgün DEĞİL)

| Konsept | Kim Yaptı? | Ne Zaman? |
|---------|-----------|-----------|
| Surprise-gated memory write | Titans (Google) | Ocak 2025 |
| Temporal Semantic Memory | TSM (arXiv:2601.07468) | Ocak 2026 |
| Sleep-like consolidation | NeuroDream, SRC | 2024-2025 |
| HyperNetwork for LLM | Profile-to-PEFT, MotherNet | 2024-2025 |
| Adaptive forgetting | Titans, FOREVER | 2024-2025 |

---

## NEURON'un GERÇEK Özgün Katkıları

### 1. Dual-Track Memory Architecture (Kısmen Yeni)

**Mevcut:** Single memory (Titans, Memoria, RAG)
**NEURON:** İki farklı memory TYPE

```
┌─────────────────────────────────────────────────────────────┐
│                    DUAL-TRACK MEMORY                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   CRYSTAL (Symbolic)         MORPH (Neural)                 │
│   ──────────────────         ─────────────────              │
│   • Key-Value Store          • State Matrix                 │
│   • Exact retrieval          • Approximate matching         │
│   • Discrete items           • Continuous encoding          │
│   • O(n) space               • O(1) space                   │
│   • "Ne" soruları            • "Nasıl/Neden" soruları       │
│                                                             │
│   FARK: İki farklı representation paradigm, tek sistemde    │
└─────────────────────────────────────────────────────────────┘
```

**Özgünlük Skoru:** 7/10 (Hibrit var ama symbolic+neural bu şekilde yok)

---

### 2. Surprise-Modulated Type Routing (YENİ!)

**Titans:** Surprise → Write/Don't write (binary)
**NEURON:** Surprise × Importance → Route to which memory (multi-path)

```python
# Titans Yaklaşımı (Mevcut)
if surprise > threshold:
    write_to_memory(info)
else:
    skip

# NEURON Yaklaşımı (YENİ!)
routing_score = importance × (1 + α × surprise)
surprise_boost = surprise > θ_surprise

if importance > θ_critical:
    if surprise_boost:
        write_crystal(info, priority=HIGH)  # Beklenmedik kritik
    else:
        write_crystal(info, priority=NORMAL)  # Beklenen kritik
elif surprise_boost:
    write_both(info)  # Beklenmedik genel → Dual write
else:
    write_morph(info)  # Beklenen genel
```

**Formül:**
```
R(x) = σ(W_r · [I(x), S(x), E(x)])

I(x) = importance_score(x)    # NER + neural classifier
S(x) = surprise_score(x)      # 1 - P(x|context)
E(x) = embedding(x)           # Semantic representation

Routing Decision:
- R(x) > 0.7 → Crystal only
- R(x) > 0.5 AND S(x) > 0.6 → Crystal + Morph (dual)
- R(x) > 0.5 → Morph with Crystal backup
- else → Morph only
```

**Özgünlük Skoru:** 9/10 (Bu kombinasyon literatürde YOK!)

---

### 3. Cross-Memory Consolidation (YENİ!)

**Mevcut sistemler:** Tek yönlü (Input → Memory)
**NEURON:** Çift yönlü (Crystal ↔ Morph interaction)

```
┌─────────────────────────────────────────────────────────────┐
│              CROSS-MEMORY CONSOLIDATION                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐          ┌─────────────┐                 │
│   │   CRYSTAL   │ ◄──────► │    MORPH    │                 │
│   │   Memory    │          │    Layer    │                 │
│   └──────┬──────┘          └──────┬──────┘                 │
│          │                        │                         │
│          │    Pattern Flow        │                         │
│          │   ────────────►        │                         │
│          │   Crystal'daki         │   Morph state'i         │
│          │   sık erişilen         │   organize etmeye       │
│          │   pattern'ler          │   yardım eder           │
│          │                        │                         │
│          │   Context Flow         │                         │
│          │   ◄────────────        │                         │
│          │   Morph'un genel       │   Crystal retrieval'ı   │
│          │   context anlayışı     │   improve eder          │
│          │                        │                         │
│   ┌──────▼────────────────────────▼──────┐                 │
│   │         CONSOLIDATION CYCLE           │                 │
│   │   (Periodic, background process)      │                 │
│   │                                       │                 │
│   │   1. Crystal clusters → Morph priors  │                 │
│   │   2. Morph context → Crystal rerank   │                 │
│   │   3. Redundancy detection             │                 │
│   │   4. Importance recalibration         │                 │
│   └───────────────────────────────────────┘                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Özgünlük Skoru:** 10/10 (Bu tamamen YENİ!)

---

### 4. Importance-Adaptive Decay (Kısmen Yeni)

**EWC:** Task importance based
**Titans:** Surprise based
**NEURON:** Multi-factor (Time × Semantic × Importance × Access)

```python
def compute_adaptive_decay(item, current_context):
    """
    Dört faktörlü decay hesaplama
    """
    # Zaman faktörü
    time_factor = exp(-α * (now - item.last_access))

    # Semantik benzerlik
    semantic_factor = cosine_sim(item.embedding, current_context)

    # Önem skoru (statik)
    importance_factor = item.importance_score

    # Erişim frekansı
    access_factor = log(1 + item.access_count) / log(1 + max_access)

    # Kombine decay
    decay = (
        w_time * time_factor +
        w_semantic * semantic_factor +
        w_importance * importance_factor +
        w_access * access_factor
    )

    return decay
```

**Özgünlük Skoru:** 7/10 (Faktörler ayrı var, bu kombinasyon kısmen yeni)

---

## Akademik Positioning

### Paper Odak Noktası

**Ana Katkı (Highlight):**
1. **Surprise-Modulated Type Routing (SMTR)** - %40
2. **Cross-Memory Consolidation (CMC)** - %40
3. **Dual-Track Architecture** - %20

### Önerilen Paper Başlığı

**Seçenek 1:**
> "NEURON: Surprise-Modulated Routing Between Symbolic and Neural Memory for Infinite Context LLMs"

**Seçenek 2:**
> "Beyond Single-Track Memory: Cross-Consolidation of Exact and Fuzzy Representations in Language Models"

**Seçenek 3:**
> "Where to Store Matters: Type-Aware Memory Routing with Bidirectional Consolidation"

### Abstract Taslağı (Güncel)

> Large language models struggle with the trade-off between exact recall and infinite capacity: retrieval-augmented systems preserve facts but scale linearly, while compression-based approaches offer constant memory but lose precision. We introduce **NEURON** (Neural Encoding with Unified Recurrent Optimized Network), featuring two novel contributions: (1) **Surprise-Modulated Type Routing (SMTR)**, which uses prediction-error signals to dynamically route information between a symbolic Crystal Memory (for exact facts) and a neural Morph Layer (for semantic understanding), and (2) **Cross-Memory Consolidation (CMC)**, a bidirectional process where Crystal patterns guide Morph organization while Morph context improves Crystal retrieval. Unlike prior work that treats memory as monolithic, NEURON recognizes that different information types require different storage paradigms. Experiments show [TBD: benchmark results].

---

## Sonraki Adımlar

1. [ ] SMTR algoritmasını formalize et
2. [ ] CMC mekanizmasını detaylandır
3. [ ] Prototype implementasyonu
4. [ ] Toy dataset ile proof-of-concept
5. [ ] Paper outline

---

## Referanslar

- Titans: arXiv:2501.00663
- TSM: arXiv:2601.07468
- Memory-Augmented Transformers Survey: arXiv:2508.10824
- Memoria: arXiv:2512.12686
