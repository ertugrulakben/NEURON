"""Tests for Morph Layer"""

import pytest
import torch

from neuron.core.morph import MorphLayer, HyperNetwork


class TestHyperNetwork:
    """Test suite for HyperNetwork."""

    def test_init(self):
        """Test initialization."""
        hyper = HyperNetwork(d_model=512, rank=64)
        assert hyper.d_model == 512
        assert hyper.rank == 64

    def test_forward_shape(self):
        """Test output shape."""
        hyper = HyperNetwork(d_model=512, rank=64)

        x = torch.randn(4, 512)  # batch of 4
        delta_W = hyper(x)

        assert delta_W.shape == (64, 64)

    def test_forward_deterministic(self):
        """Test that same input gives same output."""
        hyper = HyperNetwork(d_model=512, rank=64)
        hyper.eval()

        x = torch.randn(2, 512)

        with torch.no_grad():
            out1 = hyper(x)
            out2 = hyper(x)

        assert torch.allclose(out1, out2)


class TestMorphLayer:
    """Test suite for Morph Layer."""

    def test_init(self):
        """Test initialization."""
        morph = MorphLayer(d_model=512, rank=64)

        assert morph.d_model == 512
        assert morph.rank == 64
        assert morph.M.shape == (64, 64)
        assert morph.M.abs().max() == 0  # Zero initialized

    def test_forward_updates_state(self):
        """Test that forward pass updates state matrix."""
        morph = MorphLayer(d_model=512, rank=64)

        initial_state = morph.M.clone()

        x = torch.randn(2, 512)
        morph(x)

        # State should have changed
        assert not torch.allclose(morph.M, initial_state)

    def test_forward_output_shape(self):
        """Test output shape."""
        morph = MorphLayer(d_model=512, rank=64)

        x = torch.randn(2, 512)
        context = morph(x)

        assert context.shape == (64,)

    def test_state_accumulates(self):
        """Test that state accumulates over multiple inputs."""
        morph = MorphLayer(d_model=512, rank=64, base_decay=0.99)

        # Process multiple inputs
        for _ in range(10):
            x = torch.randn(2, 512)
            morph(x)

        # State should have accumulated (non-zero)
        assert morph.M.abs().max() > 0

    def test_decay_affects_state(self):
        """Test that decay prevents state from growing unboundedly."""
        morph = MorphLayer(d_model=512, rank=64, base_decay=0.5)

        # Process many inputs
        for _ in range(50):
            morph(torch.randn(2, 512))

        state_magnitude = morph.M.abs().mean().item()

        # State should be bounded (not exploding) due to decay
        # With decay=0.5, state magnitude should stabilize
        assert state_magnitude < 100, f"State magnitude too large: {state_magnitude}"

    def test_reset(self):
        """Test reset functionality."""
        morph = MorphLayer(d_model=512, rank=64)

        # Build up state
        for _ in range(5):
            morph(torch.randn(2, 512))

        assert morph.M.abs().max() > 0
        assert morph.n_tokens > 0

        # Reset
        morph.reset()

        assert morph.M.abs().max() == 0
        assert morph.n_tokens == 0
        assert morph.centroid.abs().max() == 0

    def test_centroid_tracking(self):
        """Test that centroid tracks input distribution."""
        morph = MorphLayer(d_model=512, rank=64)

        # Feed consistent inputs in one direction
        for _ in range(20):
            x = torch.ones(2, 512) + torch.randn(2, 512) * 0.1
            morph(x)

        # Centroid should be close to all-ones
        assert morph.centroid.mean() > 0.5

    def test_get_context_vector(self):
        """Test context vector extraction."""
        morph = MorphLayer(d_model=512, rank=64)

        # Empty state should give zero context
        ctx_empty = morph.get_context_vector()
        assert ctx_empty.abs().max() < 1e-5

        # After updates, should give non-zero context
        morph(torch.randn(2, 512))
        ctx_updated = morph.get_context_vector()
        assert ctx_updated.abs().max() > 0


class TestMorphLayerAdaptiveDecay:
    """Test adaptive decay behavior."""

    def test_similar_inputs_slow_decay(self):
        """Similar inputs should result in slower decay (higher λ)."""
        morph = MorphLayer(d_model=64, rank=16, base_decay=0.95)

        # Initialize centroid with some inputs
        base_input = torch.randn(1, 64)
        for _ in range(5):
            morph(base_input + torch.randn(1, 64) * 0.01)

        # Compute decay for similar input
        similar = base_input + torch.randn(1, 64) * 0.01
        decay_similar = morph._compute_decay(similar)

        # Compute decay for different input
        different = torch.randn(1, 64) * 5
        decay_different = morph._compute_decay(different)

        # Similar should have higher decay (slower forgetting)
        assert decay_similar > decay_different

    def test_first_input_uses_base_decay(self):
        """First input should use base decay."""
        morph = MorphLayer(d_model=64, rank=16, base_decay=0.95)

        decay = morph._compute_decay(torch.randn(1, 64))

        assert decay == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
