"""
NEURON Benchmark Suite

Objective tests to measure:
1. Crystal recall accuracy
2. Morph context quality
3. Routing accuracy
4. Memory efficiency
5. Speed
"""

import time
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch

from neuron.core.neuron import NEURON
from neuron.core.router import RouteDecision


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    score: float
    details: dict


def create_test_data(n_items: int = 100) -> List[Tuple[torch.Tensor, str, str]]:
    """
    Create test dataset with critical and general items.

    Returns: List of (embedding, text, expected_type)
    """
    data = []

    # Critical items (dates, numbers, names)
    critical_templates = [
        "The meeting is scheduled for {}/03/2026 at {}:00.",
        "John Smith's budget is ${},000.",
        "Contact email: user{}@company.com",
        "Password: X7y{}Z!",
        "Project deadline: {}/04/2026",
    ]

    # General items
    general_templates = [
        "We discussed the project requirements in detail.",
        "The team agreed on the general approach.",
        "There were some concerns about the timeline.",
        "Everyone seemed positive about the outcome.",
        "The meeting went well overall.",
    ]

    for i in range(n_items):
        embedding = torch.randn(512)

        if i % 2 == 0:  # Critical
            template = random.choice(critical_templates)
            text = template.format(
                random.randint(1, 28),
                random.randint(1, 12),
            )
            expected = "critical"
        else:  # General
            text = random.choice(general_templates)
            expected = "general"

        data.append((embedding, text, expected))

    return data


def benchmark_crystal_recall(neuron: NEURON, n_items: int = 50) -> BenchmarkResult:
    """
    Benchmark: Crystal exact recall accuracy.

    Store items with known keys, then retrieve and measure accuracy.
    """
    correct = 0
    total = 0

    # Store items
    stored_items = []
    for i in range(n_items):
        key = torch.randn(512)
        value = f"Critical fact #{i}: Meeting at {i}:00"
        neuron.absorb(key, text=value)
        stored_items.append((key, value))

    # Retrieve and check
    for key, expected_value in stored_items:
        results = neuron.query(key, top_k=1)

        if results["crystal_results"]:
            retrieved = results["crystal_results"][0]["value"]
            if expected_value in str(retrieved):
                correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0

    return BenchmarkResult(
        name="Crystal Recall Accuracy",
        score=accuracy,
        details={
            "correct": correct,
            "total": total,
            "items_in_crystal": len(neuron.crystal),
        }
    )


def benchmark_routing_accuracy(neuron: NEURON, n_items: int = 100) -> BenchmarkResult:
    """
    Benchmark: Routing decision accuracy.

    Check if critical items go to Crystal and general items go to Morph.
    """
    correct_routing = 0
    total = 0

    test_data = create_test_data(n_items)

    for embedding, text, expected in test_data:
        stats = neuron.absorb(embedding, text=text)

        decision = stats["decision"]

        if expected == "critical":
            # Should go to crystal or both
            if decision in ["crystal", "crystal_high", "both"]:
                correct_routing += 1
        else:  # general
            # Should go to morph or both
            if decision in ["morph", "both"]:
                correct_routing += 1

        total += 1

    accuracy = correct_routing / total if total > 0 else 0

    return BenchmarkResult(
        name="Routing Accuracy",
        score=accuracy,
        details={
            "correct": correct_routing,
            "total": total,
        }
    )


def benchmark_memory_efficiency(neuron: NEURON, n_items: int = 1000) -> BenchmarkResult:
    """
    Benchmark: Memory efficiency.

    Measure memory usage as items increase.
    """
    import sys

    # Initial memory
    initial_crystal = len(neuron.crystal)
    initial_morph_tokens = neuron.morph.n_tokens.item()

    # Absorb many items
    for i in range(n_items):
        embedding = torch.randn(512)
        text = f"Item {i}" if i % 2 == 0 else f"Meeting at {i}:00"
        neuron.absorb(embedding, text=text)

    # Final memory
    final_crystal = len(neuron.crystal)
    final_morph_tokens = neuron.morph.n_tokens.item()

    # Morph should be O(1) - fixed size
    morph_size = neuron.morph.M.numel() * 4  # 4 bytes per float32

    # Crystal grows but is bounded
    crystal_bounded = final_crystal <= neuron.crystal.max_items

    return BenchmarkResult(
        name="Memory Efficiency",
        score=1.0 if crystal_bounded else 0.5,
        details={
            "crystal_items": final_crystal,
            "crystal_max": neuron.crystal.max_items,
            "morph_tokens_processed": final_morph_tokens,
            "morph_matrix_bytes": morph_size,
            "crystal_bounded": crystal_bounded,
        }
    )


def benchmark_speed(neuron: NEURON, n_items: int = 100) -> BenchmarkResult:
    """
    Benchmark: Absorb and query speed.
    """
    # Absorb speed
    embeddings = [torch.randn(512) for _ in range(n_items)]
    texts = [f"Test item {i}" for i in range(n_items)]

    start = time.time()
    for emb, text in zip(embeddings, texts):
        neuron.absorb(emb, text=text)
    absorb_time = time.time() - start

    # Query speed
    queries = [torch.randn(512) for _ in range(n_items)]

    start = time.time()
    for q in queries:
        neuron.query(q, top_k=5)
    query_time = time.time() - start

    return BenchmarkResult(
        name="Speed",
        score=1.0,  # Informational
        details={
            "absorb_total_ms": absorb_time * 1000,
            "absorb_per_item_ms": (absorb_time * 1000) / n_items,
            "query_total_ms": query_time * 1000,
            "query_per_item_ms": (query_time * 1000) / n_items,
        }
    )


def benchmark_context_retention(neuron: NEURON) -> BenchmarkResult:
    """
    Benchmark: Long-term context retention in Morph.

    Test if Morph retains general context over many inputs.
    """
    # Establish a consistent context direction
    context_direction = torch.randn(512)
    context_direction = context_direction / context_direction.norm()

    # Feed many inputs in that direction
    for _ in range(50):
        embedding = context_direction + torch.randn(512) * 0.1
        neuron.absorb(embedding, text="Consistent context input")

    # Check if centroid aligns with direction
    centroid = neuron.morph.centroid
    centroid_norm = centroid / (centroid.norm() + 1e-8)

    alignment = torch.dot(centroid_norm, context_direction).item()

    return BenchmarkResult(
        name="Context Retention",
        score=max(0, alignment),
        details={
            "alignment_score": alignment,
            "centroid_magnitude": centroid.norm().item(),
        }
    )


def run_all_benchmarks() -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    print("=" * 60)
    print("NEURON BENCHMARK SUITE")
    print("=" * 60)
    print()

    results = []

    # Create fresh NEURON instance for each benchmark
    # Increased sample sizes for better statistical validity

    print("Running Crystal Recall benchmark (n=500)...")
    neuron = NEURON(d_model=512, crystal_size=1000, morph_rank=64)
    result = benchmark_crystal_recall(neuron, n_items=500)
    results.append(result)
    print(f"  Score: {result.score:.2%}")

    print("Running Routing Accuracy benchmark (n=1000)...")
    neuron = NEURON(d_model=512, crystal_size=2000, morph_rank=64)
    result = benchmark_routing_accuracy(neuron, n_items=1000)
    results.append(result)
    print(f"  Score: {result.score:.2%}")

    print("Running Memory Efficiency benchmark (n=2000)...")
    neuron = NEURON(d_model=512, crystal_size=100, morph_rank=64)  # Small crystal to test eviction
    result = benchmark_memory_efficiency(neuron, n_items=2000)
    results.append(result)
    print(f"  Score: {result.score:.2%}")

    print("Running Speed benchmark (n=500)...")
    neuron = NEURON(d_model=512, crystal_size=1000, morph_rank=64)
    result = benchmark_speed(neuron, n_items=500)
    results.append(result)
    print(f"  Absorb: {result.details['absorb_per_item_ms']:.2f} ms/item")
    print(f"  Query: {result.details['query_per_item_ms']:.2f} ms/item")

    print("Running Context Retention benchmark (n=100)...")
    neuron = NEURON(d_model=512, crystal_size=1000, morph_rank=64)
    result = benchmark_context_retention(neuron)
    results.append(result)
    print(f"  Score: {result.score:.2%}")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary."""
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print()

    for result in results:
        print(f"[*] {result.name}")
        print(f"   Score: {result.score:.2%}")
        for key, value in result.details.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        print()

    # Overall score (average of accuracy benchmarks)
    accuracy_scores = [r.score for r in results if "Accuracy" in r.name or "Recall" in r.name or "Retention" in r.name]
    if accuracy_scores:
        overall = sum(accuracy_scores) / len(accuracy_scores)
        print(f">>> Overall Accuracy Score: {overall:.2%}")
    print()


if __name__ == "__main__":
    results = run_all_benchmarks()
    print_summary(results)
