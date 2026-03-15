"""
NEURON Contradiction Detection Example (TBG)

Demonstrates the Temporal Belief Graph — NEURON's key novel contribution:
1. Store a fact via TBG
2. Store a contradictory fact with controlled similarity
3. Observe contradiction detection
4. Query belief confidence and history
"""

import torch
import torch.nn.functional as F

from neuron.core.belief import TemporalBeliefGraph
from neuron.core.crystal import CrystalMemory


def make_similar_embedding(base: torch.Tensor, target_similarity: float = 0.90) -> torch.Tensor:
    """Create an embedding with a controlled cosine similarity to base."""
    base_norm = F.normalize(base.unsqueeze(0), dim=-1).squeeze()
    # Create orthogonal component
    random = torch.randn_like(base)
    orthogonal = random - (random @ base_norm) * base_norm
    orthogonal = F.normalize(orthogonal.unsqueeze(0), dim=-1).squeeze()
    # Mix: cos_sim = alpha / sqrt(alpha^2 + beta^2)
    # For target sim s: beta = alpha * sqrt(1/s^2 - 1)
    alpha = 1.0
    beta = alpha * (1.0 / target_similarity**2 - 1) ** 0.5
    result = alpha * base_norm + beta * orthogonal
    return result


def main():
    d_model = 256

    crystal = CrystalMemory(max_items=100, embedding_dim=d_model)
    tbg = TemporalBeliefGraph(crystal, contradiction_threshold=0.85)

    # Base topic embedding
    topic = torch.randn(d_model)

    # --- Step 1: Store initial fact ---
    print("=== Step 1: Store Initial Fact ===")
    result1 = tbg.store(topic, "Meeting at 3pm", importance=0.8)
    print(f"  Stored: {result1['stored']}")
    print(f"  Confidence: {result1['belief_state']['confidence']}")

    # --- Step 2: Store contradictory fact ---
    # Same topic (sim ~0.90) but different value → contradiction detected
    print("\n=== Step 2: Store Contradictory Fact ===")
    emb2 = make_similar_embedding(topic, target_similarity=0.90)
    result2 = tbg.store(emb2, "Meeting at 4pm", importance=0.8)
    print(f"  Stored: {result2['stored']}")
    print(f"  Contradictions: {len(result2['contradictions'])}")
    for c in result2["contradictions"]:
        print(f"    Type: {c['type']}")
        print(f"    Existing value: \"{c['existing_value']}\"")
        print(f"    Similarity: {c['similarity']:.3f}")
    if result2["belief_state"]:
        print(f"  Confidence: {result2['belief_state']['confidence']}")
        print("  (Lower because contradiction was detected)")

    # --- Step 3: Corroborate the new fact ---
    print("\n=== Step 3: Corroborate the New Fact ===")
    emb3 = make_similar_embedding(emb2, target_similarity=0.92)
    result3 = tbg.store(emb3, "Meeting at 4pm per calendar", importance=0.9)
    print(f"  Stored: {result3['stored']}")
    print(f"  Corroborations: {len(result3['corroborations'])}")

    # --- Check contradictions ---
    print("\n=== Unresolved Contradictions ===")
    contradictions = tbg.get_contradictions(unresolved_only=True)
    print(f"  Total: {len(contradictions)}")
    for c in contradictions:
        print(f"    {c.contradiction_type.value}: \"{c.conflicting_value}\"")

    # --- Belief history ---
    print("\n=== Belief History ===")
    for entry in tbg.get_belief_history():
        print(f"  Item #{entry['item_idx']}: confidence={entry['confidence']:.3f}")

    # --- Stats ---
    print("\n=== TBG Stats ===")
    stats = tbg.stats()
    for k, v in stats.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
