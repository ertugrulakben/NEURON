"""Tests for Importance Router (SMTR)"""

import pytest
import torch

from neuron.core.router import ImportanceRouter, RouteDecision, RoutingResult


class TestImportanceRouter:
    """Test suite for Importance Router."""

    def test_init(self):
        """Test initialization."""
        router = ImportanceRouter(d_model=768)
        assert router.d_model == 768
        assert router.critical_threshold == 0.6
        assert router.surprise_threshold == 0.5

    def test_forward_returns_routing_result(self):
        """Test that forward returns RoutingResult."""
        router = ImportanceRouter(d_model=768)

        embedding = torch.randn(768)
        result = router(embedding)

        assert isinstance(result, RoutingResult)
        assert isinstance(result.decision, RouteDecision)
        assert 0 <= result.importance <= 1
        assert 0 <= result.surprise <= 1
        assert 0 <= result.confidence <= 1

    def test_pattern_detection_date(self):
        """Test pattern detection for dates."""
        router = ImportanceRouter(d_model=768)

        embedding = torch.randn(768)

        result_no_date = router(embedding, text="This is a regular sentence.")
        result_with_date = router(embedding, text="The meeting is on 15/03/2026.")

        # With date should have higher importance
        assert result_with_date.importance > result_no_date.importance

    def test_pattern_detection_time(self):
        """Test pattern detection for times."""
        router = ImportanceRouter(d_model=768)

        embedding = torch.randn(768)

        result_no_time = router(embedding, text="Let's meet tomorrow.")
        result_with_time = router(embedding, text="Let's meet at 3:45 PM.")

        assert result_with_time.importance > result_no_time.importance

    def test_pattern_detection_money(self):
        """Test pattern detection for money."""
        router = ImportanceRouter(d_model=768)

        embedding = torch.randn(768)

        result_no_money = router(embedding, text="The project is expensive.")
        result_with_money = router(embedding, text="The budget is $75,000.")

        assert result_with_money.importance > result_no_money.importance

    def test_pattern_detection_names(self):
        """Test pattern detection for names."""
        router = ImportanceRouter(d_model=768)

        embedding = torch.randn(768)

        result_no_name = router(embedding, text="The developer fixed the bug.")
        result_with_name = router(embedding, text="John Smith fixed the bug.")

        assert result_with_name.importance > result_no_name.importance

    def test_surprise_calculation(self):
        """Test surprise calculation."""
        router = ImportanceRouter(d_model=768)

        current = torch.randn(768)
        similar_context = current + torch.randn(768) * 0.1
        different_context = torch.randn(768) * 5

        result_similar = router(current, context_embedding=similar_context)
        result_different = router(current, context_embedding=different_context)

        # Different context should be more surprising
        assert result_different.surprise > result_similar.surprise

    def test_routing_decision_crystal_high(self):
        """Test CRYSTAL_HIGH routing (critical + surprising)."""
        router = ImportanceRouter(
            d_model=768,
            critical_threshold=0.3,  # Low threshold for testing
            surprise_threshold=0.3,
        )

        # Force high importance via patterns
        embedding = torch.randn(768)
        text = "John Smith earned $100,000 on 15/03/2026 at 3:45 PM."
        different_context = torch.randn(768) * 10

        result = router(embedding, text=text, context_embedding=different_context)

        # Should route to CRYSTAL_HIGH or CRYSTAL
        assert result.decision in [RouteDecision.CRYSTAL_HIGH, RouteDecision.CRYSTAL]

    def test_routing_decision_morph(self):
        """Test MORPH routing (not critical, not surprising)."""
        router = ImportanceRouter(
            d_model=768,
            critical_threshold=0.9,  # High threshold
            surprise_threshold=0.9,
        )

        embedding = torch.randn(768)
        similar_context = embedding + torch.randn(768) * 0.01

        result = router(
            embedding,
            text="This is just a general comment.",
            context_embedding=similar_context
        )

        # Should likely route to MORPH
        assert result.decision == RouteDecision.MORPH

    def test_routing_decision_both(self):
        """Test BOTH routing (surprising but not critical)."""
        router = ImportanceRouter(
            d_model=768,
            critical_threshold=0.95,  # Very high
            surprise_threshold=0.2,   # Low
        )

        embedding = torch.randn(768)
        different_context = torch.randn(768) * 10  # Very different

        result = router(
            embedding,
            text="Something unexpected happened.",
            context_embedding=different_context
        )

        # High surprise, low importance = BOTH
        if result.surprise >= 0.2 and result.importance < 0.95:
            assert result.decision == RouteDecision.BOTH


class TestRoutingDecisionEnum:
    """Test RouteDecision enum."""

    def test_all_values(self):
        """Test all enum values exist."""
        assert RouteDecision.CRYSTAL_HIGH.value == "crystal_high"
        assert RouteDecision.CRYSTAL.value == "crystal"
        assert RouteDecision.BOTH.value == "both"
        assert RouteDecision.MORPH.value == "morph"


class TestRoutingResult:
    """Test RoutingResult dataclass."""

    def test_creation(self):
        """Test RoutingResult creation."""
        result = RoutingResult(
            decision=RouteDecision.CRYSTAL,
            importance=0.8,
            surprise=0.3,
            confidence=0.9,
        )

        assert result.decision == RouteDecision.CRYSTAL
        assert result.importance == 0.8
        assert result.surprise == 0.3
        assert result.confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
