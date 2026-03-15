"""
NEURON Basic Usage Example

Demonstrates the core workflow:
1. Create a NEURON instance
2. Absorb information (critical vs. general)
3. Query the memory system
4. Inspect routing decisions and memory stats
"""

import torch

from neuron import NEURON, NeuronConfig


def main():
    # --- Setup ---
    config = NeuronConfig(
        d_model=512,
        crystal_size=1000,
        morph_rank=64,
    )
    memory = NEURON(config=config)
    print(f"NEURON initialized: d_model={config.d_model}, crystal_size={config.crystal_size}")
    print()

    # --- Absorb critical information ---
    # These contain patterns (dates, names, money) that the router
    # will detect and route to Crystal Memory for exact recall.
    critical_items = [
        "The project budget is $75,000.",
        "Deadline is 15/03/2026 at 3:45 PM.",
        "Contact John Smith at john@company.com.",
    ]

    print("=== Absorbing Critical Information ===")
    for text in critical_items:
        embedding = torch.randn(config.d_model)
        stats = memory.absorb(embedding, text=text)
        print(f"  [{stats['decision']:>12}] {text}")

    # --- Absorb general information ---
    # No critical patterns → routes to Morph Layer for fuzzy context.
    general_items = [
        "We discussed the project requirements in detail.",
        "The team agreed on the general approach.",
        "Everyone seemed positive about the outcome.",
    ]

    print("\n=== Absorbing General Information ===")
    for text in general_items:
        embedding = torch.randn(config.d_model)
        stats = memory.absorb(embedding, text=text)
        print(f"  [{stats['decision']:>12}] {text}")

    # --- Query ---
    print("\n=== Querying Memory ===")
    query_embedding = torch.randn(config.d_model)
    results = memory.query(query_embedding, top_k=3)

    print(f"  Crystal results: {len(results['crystal_results'])} items")
    for r in results["crystal_results"]:
        print(f"    - {r['value']} (sim={r['similarity']:.3f})")

    morph_mag = results["morph_context"].abs().mean().item()
    print(f"  Morph context magnitude: {morph_mag:.4f}")
    print(f"  Fused output shape: {results['fused_output'].shape}")

    # --- Stats ---
    print("\n=== Memory Stats ===")
    stats = memory.stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # --- Session reset ---
    print("\n=== Session Reset ===")
    memory.reset_session()
    print(f"  Crystal items preserved: {len(memory.crystal)}")
    print(f"  Morph tokens after reset: {memory.morph.n_tokens.item()}")


if __name__ == "__main__":
    main()
