"""
Importance Router: Surprise-Modulated Type Routing (SMTR)

Key Innovation: Uses both importance AND surprise to determine
where information should be routed:
- Critical + Surprising → Crystal (high priority)
- Critical → Crystal (normal priority)
- Surprising (not critical) → Both Crystal and Morph
- General → Morph only

This is different from Titans (Google, 2025) which uses surprise
only for write/skip decisions, not for routing between memory types.
"""

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteDecision(Enum):
    """Routing decision types."""
    CRYSTAL_HIGH = "crystal_high"      # Critical + Surprising
    CRYSTAL = "crystal"                 # Critical only
    BOTH = "both"                       # Surprising but not critical
    MORPH = "morph"                     # General information


@dataclass
class RoutingResult:
    """Result of routing decision."""
    decision: RouteDecision
    importance: float
    surprise: float
    confidence: float


class ImportanceRouter(nn.Module):
    """
    Surprise-Modulated Type Routing (SMTR)

    Combines neural classification with pattern matching to determine
    where information should be stored:

    1. Neural: Learned classifier for importance scoring
    2. Pattern: Regex patterns for critical info (dates, names, numbers)
    3. Surprise: Measures unexpectedness via prediction error

    Args:
        d_model: Model dimension (default: 768)
        critical_threshold: Threshold for critical routing (default: 0.6)
        surprise_threshold: Threshold for surprise boost (default: 0.5)
    """

    # Common patterns for critical information
    CRITICAL_PATTERNS = [
        r'\b\d{1,2}[:/]\d{2}\b',                    # Time (3:45, 15:30)
        r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b',      # Date (15/03/2026)
        r'\b\$[\d,]+(?:\.\d{2})?\b',               # Money ($75,000)
        r'\b\d+%\b',                                # Percentage (15%)
        r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b',     # Names (John Smith)
        r'\b[A-Z]{2,}\b',                          # Acronyms (API, LLM)
        r'```[\s\S]*?```',                         # Code blocks
        r'\b\d{3,}\b',                             # Large numbers (1000+)
        r'\bhttps?://\S+\b',                       # URLs
        r'\b[\w.-]+@[\w.-]+\.\w+\b',              # Emails
    ]

    def __init__(
        self,
        d_model: int = 768,
        critical_threshold: float = 0.6,
        surprise_threshold: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.critical_threshold = critical_threshold
        self.surprise_threshold = surprise_threshold

        # Neural importance classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Compile patterns
        self.patterns = [re.compile(p) for p in self.CRITICAL_PATTERNS]

    def forward(
        self,
        embedding: torch.Tensor,
        text: Optional[str] = None,
        context_embedding: Optional[torch.Tensor] = None,
    ) -> RoutingResult:
        """
        Determine routing for input.

        Args:
            embedding: Input embedding [batch, d_model] or [d_model]
            text: Original text (for pattern matching)
            context_embedding: Previous context (for surprise calculation)

        Returns:
            RoutingResult with decision and scores
        """
        # Ensure 2D
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Neural importance score
        importance = self.classifier(embedding).mean().item()

        # Pattern boost
        if text:
            pattern_boost = self._check_patterns(text)
            importance = min(1.0, importance + pattern_boost * 0.3)

        # Surprise score
        surprise = 0.0
        if context_embedding is not None:
            surprise = self._compute_surprise(embedding, context_embedding)

        # Make routing decision
        decision = self._make_decision(importance, surprise)
        confidence = self._compute_confidence(importance, surprise, decision)

        return RoutingResult(
            decision=decision,
            importance=importance,
            surprise=surprise,
            confidence=confidence,
        )

    def _check_patterns(self, text: str) -> float:
        """Check for critical patterns in text."""
        matches = sum(1 for p in self.patterns if p.search(text))
        return min(1.0, matches / 3)  # Normalize

    def _compute_surprise(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
    ) -> float:
        """
        Compute surprise as inverse similarity to context.

        Surprise = 1 - similarity(current, context)
        """
        if context.dim() == 1:
            context = context.unsqueeze(0)

        similarity = F.cosine_similarity(
            current.mean(dim=0, keepdim=True),
            context.mean(dim=0, keepdim=True),
        ).item()

        return max(0, 1 - similarity)

    def _make_decision(self, importance: float, surprise: float) -> RouteDecision:
        """Make routing decision based on importance and surprise."""
        is_critical = importance >= self.critical_threshold
        is_surprising = surprise >= self.surprise_threshold

        if is_critical and is_surprising:
            return RouteDecision.CRYSTAL_HIGH
        elif is_critical:
            return RouteDecision.CRYSTAL
        elif is_surprising:
            return RouteDecision.BOTH  # Hedge our bets
        else:
            return RouteDecision.MORPH

    def _compute_confidence(
        self,
        importance: float,
        surprise: float,
        decision: RouteDecision,
    ) -> float:
        """
        Compute confidence in the routing decision.

        Fixed formula: Uses sigmoid-like scaling so confidence varies meaningfully
        from ~0.3 (uncertain, near thresholds) to ~0.95 (confident, far from thresholds).
        """
        # How far from thresholds (normalized to [0, 1])
        importance_margin = abs(importance - self.critical_threshold) / self.critical_threshold
        surprise_margin = abs(surprise - self.surprise_threshold) / self.surprise_threshold

        # Combined margin (average)
        avg_margin = (importance_margin + surprise_margin) / 2

        # Sigmoid-like scaling: 0 margin → 0.5, high margin → ~0.95
        # Formula: 0.3 + 0.65 * (1 - exp(-3 * margin))
        confidence = 0.3 + 0.65 * (1 - math.exp(-3 * avg_margin))

        return min(0.99, max(0.1, confidence))
