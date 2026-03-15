"""Tests for Crystal Memory"""

import pytest
import torch

from neuron.core.crystal import CrystalMemory, MemoryItem


class TestCrystalMemory:
    """Test suite for Crystal Memory."""

    def test_init(self):
        """Test initialization."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)
        assert crystal.max_items == 100
        assert crystal.embedding_dim == 64
        assert len(crystal) == 0

    def test_store_single(self):
        """Test storing a single item."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        key = torch.randn(64)
        value = "Test value"

        stored = crystal.store(key, value, importance=0.8)

        assert stored == True
        assert len(crystal) == 1
        assert crystal.items[0].value == "Test value"
        assert crystal.items[0].importance == 0.8

    def test_store_duplicate_updates(self):
        """Test that storing duplicate updates existing item."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        key = torch.randn(64)

        crystal.store(key, "Original", importance=0.5)
        stored = crystal.store(key, "Updated", importance=0.9)

        assert stored == False  # Duplicate detected
        assert len(crystal) == 1
        assert crystal.items[0].value == "Updated"
        assert crystal.items[0].importance == 0.9  # Max of old and new

    def test_retrieve_exact(self):
        """Test exact retrieval."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        # Store items
        keys = [torch.randn(64) for _ in range(5)]
        for i, key in enumerate(keys):
            crystal.store(key, f"Item {i}", importance=0.5 + i * 0.1)

        # Query with exact key
        results = crystal.retrieve(keys[2], top_k=1)

        assert len(results) >= 1
        assert results[0]["value"] == "Item 2"
        assert results[0]["similarity"] > 0.99

    def test_retrieve_similar(self):
        """Test retrieval with similar but not exact key."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        key = torch.randn(64)
        crystal.store(key, "Target", importance=0.8)

        # Query with slightly perturbed key
        noisy_key = key + torch.randn(64) * 0.1
        results = crystal.retrieve(noisy_key, top_k=1)

        assert len(results) >= 1
        assert results[0]["value"] == "Target"
        assert results[0]["similarity"] > 0.8

    def test_eviction_at_capacity(self):
        """Test LRU eviction when at capacity."""
        crystal = CrystalMemory(max_items=5, embedding_dim=64)

        # Fill to capacity
        for i in range(5):
            crystal.store(torch.randn(64), f"Item {i}", importance=0.5)

        assert len(crystal) == 5

        # Add one more - should trigger eviction
        crystal.store(torch.randn(64), "New item", importance=0.5)

        assert len(crystal) == 5  # Still at capacity

    def test_importance_affects_eviction(self):
        """Test that high importance items are less likely to be evicted."""
        crystal = CrystalMemory(max_items=3, embedding_dim=64)

        # Store low importance items
        for i in range(2):
            crystal.store(torch.randn(64), f"Low {i}", importance=0.1)

        # Store high importance item
        high_key = torch.randn(64)
        crystal.store(high_key, "High importance", importance=0.99)

        # Add more items to trigger eviction
        for i in range(2):
            crystal.store(torch.randn(64), f"New {i}", importance=0.5)

        # High importance item should still be there
        results = crystal.retrieve(high_key, top_k=1)
        assert any(r["value"] == "High importance" for r in results)

    def test_clear(self):
        """Test clearing memory."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        for i in range(10):
            crystal.store(torch.randn(64), f"Item {i}")

        assert len(crystal) == 10

        crystal.clear()

        assert len(crystal) == 0

    def test_access_count_increases(self):
        """Test that access count increases on retrieval."""
        crystal = CrystalMemory(max_items=100, embedding_dim=64)

        key = torch.randn(64)
        crystal.store(key, "Test")

        assert crystal.items[0].access_count == 0

        # Retrieve multiple times
        for _ in range(3):
            crystal.retrieve(key, top_k=1)

        assert crystal.items[0].access_count == 3


class TestCrystalMemoryEdgeCases:
    """Edge case tests for Crystal Memory."""

    def test_empty_retrieval(self):
        """Test retrieval from empty memory."""
        crystal = CrystalMemory()
        results = crystal.retrieve(torch.randn(768), top_k=5)
        assert results == []

    def test_retrieval_below_threshold(self):
        """Test that low similarity results are filtered."""
        crystal = CrystalMemory(similarity_threshold=0.9)

        crystal.store(torch.randn(768), "Item")

        # Query with unrelated key
        results = crystal.retrieve(torch.randn(768), top_k=5)

        # Should be empty or filtered due to low similarity
        assert all(r["similarity"] >= 0.9 for r in results)

    def test_batch_embedding(self):
        """Test with 2D embedding input."""
        crystal = CrystalMemory(embedding_dim=64)

        # 2D input (batch of 1)
        key = torch.randn(1, 64)
        crystal.store(key, "Batch test")

        assert len(crystal) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
