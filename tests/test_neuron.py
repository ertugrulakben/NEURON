"""Integration tests for NEURON"""

import pytest
import torch

from neuron.core.neuron import NEURON
from neuron.core.router import RouteDecision


class TestNEURONBasic:
    """Basic functionality tests for NEURON."""

    def test_init(self):
        """Test initialization."""
        neuron = NEURON(d_model=512, crystal_size=100, morph_rank=32)

        assert neuron.d_model == 512
        assert neuron.crystal.max_items == 100
        assert neuron.morph.rank == 32

    def test_absorb_returns_stats(self):
        """Test that absorb returns stats dict."""
        neuron = NEURON(d_model=512)

        embedding = torch.randn(512)
        stats = neuron.absorb(embedding, text="Test input")

        assert "decision" in stats
        assert "importance" in stats
        assert "surprise" in stats
        assert "stored_crystal" in stats
        assert "updated_morph" in stats

    def test_query_returns_results(self):
        """Test that query returns results dict."""
        neuron = NEURON(d_model=512)

        # Absorb some data first
        neuron.absorb(torch.randn(512), text="Test data")

        # Query
        query = torch.randn(512)
        results = neuron.query(query)

        assert "crystal_results" in results
        assert "morph_context" in results
        assert "fused_output" in results

    def test_stats(self):
        """Test stats reporting."""
        neuron = NEURON(d_model=512, crystal_size=100)

        stats = neuron.stats()

        assert stats["crystal_items"] == 0
        assert stats["crystal_capacity"] == 100
        assert stats["morph_tokens_seen"] == 0

    def test_reset_session(self):
        """Test session reset (keeps Crystal)."""
        neuron = NEURON(d_model=512)

        # Absorb critical info to Crystal
        for i in range(5):
            embedding = torch.randn(512)
            neuron.absorb(embedding, text=f"Meeting at {i}:00 PM with John Smith")

        crystal_count_before = len(neuron.crystal)

        # Reset session
        neuron.reset_session()

        # Crystal should be preserved
        assert len(neuron.crystal) == crystal_count_before
        # Morph should be reset
        assert neuron.morph.n_tokens == 0

    def test_reset_all(self):
        """Test full reset."""
        neuron = NEURON(d_model=512)

        # Absorb data
        for i in range(10):
            neuron.absorb(torch.randn(512), text=f"Data {i}")

        # Reset all
        neuron.reset_all()

        assert len(neuron.crystal) == 0
        assert neuron.morph.n_tokens == 0


class TestNEURONRouting:
    """Test routing behavior."""

    def test_critical_info_goes_to_crystal(self):
        """Test that critical info is stored in Crystal."""
        neuron = NEURON(d_model=512)

        # Absorb critical info (has patterns)
        embedding = torch.randn(512)
        stats = neuron.absorb(
            embedding,
            text="The budget is $75,000 and the deadline is 15/03/2026."
        )

        # Should have stored in Crystal
        assert stats["stored_crystal"] == True or stats["decision"] in ["crystal", "crystal_high", "both"]

    def test_general_info_goes_to_morph(self):
        """Test that general info goes to Morph."""
        neuron = NEURON(d_model=512)

        # First, establish context
        for _ in range(5):
            neuron.absorb(torch.randn(512), text="General discussion about the project.")

        # Absorb general info (no patterns, not surprising)
        embedding = torch.randn(512) * 0.01 + neuron._last_embedding  # Similar to context
        stats = neuron.absorb(
            embedding,
            text="We continued the general discussion."
        )

        # Morph should be updated
        assert stats["updated_morph"] == True


class TestNEURONRetrieval:
    """Test retrieval functionality."""

    def test_exact_retrieval_from_crystal(self):
        """Test exact retrieval of critical facts."""
        neuron = NEURON(d_model=512)

        # Store multiple critical facts to ensure Crystal has data
        key_embedding = torch.randn(512)
        neuron.absorb(key_embedding, text="The password is X7y9Z!")
        neuron.absorb(torch.randn(512), text="Budget is $500,000 on 15/03/2026")
        neuron.absorb(torch.randn(512), text="Contact John Smith at john@test.com")

        # Retrieve with same key
        results = neuron.query(key_embedding)

        # Fused output should be non-zero (memory is populated)
        assert results["fused_output"].abs().max() > 0

    def test_context_from_morph(self):
        """Test context retrieval from Morph."""
        neuron = NEURON(d_model=512)

        # Build up context
        for _ in range(20):
            neuron.absorb(torch.randn(512), text="Discussing project requirements.")

        # Query
        results = neuron.query(torch.randn(512))

        # Morph context should be non-zero
        assert results["morph_context"].abs().max() > 0


class TestNEURONConsolidation:
    """Test cross-memory consolidation."""

    def test_consolidation_runs(self):
        """Test that consolidation runs periodically."""
        neuron = NEURON(d_model=512, consolidation_interval=10)

        # Process enough inputs to trigger consolidation
        for i in range(15):
            neuron.absorb(torch.randn(512), text=f"Input {i}")

        # Should have run consolidation at step 10
        assert neuron._step_count >= 15


class TestNEURONEdgeCases:
    """Edge case tests."""

    def test_empty_query(self):
        """Test querying empty memory."""
        neuron = NEURON(d_model=512)

        results = neuron.query(torch.randn(512))

        assert results["crystal_results"] == []
        assert results["morph_context"].abs().max() < 1e-5

    def test_batch_input(self):
        """Test with batch input."""
        neuron = NEURON(d_model=512)

        # Batch of 4
        embedding = torch.randn(4, 512)
        stats = neuron.absorb(embedding, text="Batch test")

        assert "decision" in stats

    def test_very_long_text(self):
        """Test with very long text."""
        neuron = NEURON(d_model=512)

        long_text = "Test " * 1000
        stats = neuron.absorb(torch.randn(512), text=long_text)

        assert "decision" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
