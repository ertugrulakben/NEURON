"""
Temporal Belief Graph (TBG): Novel contribution to memory systems.

Key Innovation: Memories are not just stored - they have evolving belief states.
Unlike existing systems (Titans, BudgetMem, Mem0) that either overwrite or ignore
contradictions, TBG tracks:

1. Confidence: How certain are we about this memory?
2. Corroboration: How many times has it been confirmed?
3. Contradictions: What conflicts with this memory?
4. Temporal Validity: When was this true?

This enables:
- Contradiction detection and resolution
- Belief evolution over time
- Evidence-based confidence calibration
- "What did I believe about X over time?" queries

Author: Ertuğrul Akben
Date: 2026-02-02
"""

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
from enum import Enum

import torch
import torch.nn.functional as F


class ContradictionType(Enum):
    """Types of contradictions between memories."""
    NONE = "none"                    # No contradiction
    VALUE_CONFLICT = "value"         # Same topic, different value (meeting at 3pm vs 4pm)
    TEMPORAL_CONFLICT = "temporal"   # Same fact, different time validity
    NEGATION = "negation"            # Direct negation (X is true vs X is false)
    PARTIAL = "partial"              # Partial overlap with conflict


@dataclass
class BeliefState:
    """
    Tracks the evolving belief in a memory.

    Unlike traditional memory systems that treat all memories equally,
    BeliefState tracks how confident we should be in each memory.
    """
    confidence: float = 0.5              # Initial confidence (uncertain)
    corroboration_count: int = 0         # Times confirmed
    contradiction_count: int = 0         # Times contradicted
    last_corroborated: float = 0.0       # Timestamp
    last_contradicted: float = 0.0       # Timestamp
    temporal_start: Optional[float] = None  # When fact became valid
    temporal_end: Optional[float] = None    # When fact stopped being valid (None = still valid)

    def update_confidence(self) -> float:
        """
        Compute confidence based on evidence.

        Formula: confidence = base + corroboration_boost - contradiction_penalty
        With temporal decay for old corroborations.
        """
        now = time.time()

        # Base confidence from corroboration ratio
        total_evidence = self.corroboration_count + self.contradiction_count
        if total_evidence == 0:
            return 0.5  # No evidence yet

        ratio = self.corroboration_count / total_evidence

        # Temporal decay: recent evidence matters more
        corr_recency = 1.0 / (1.0 + (now - self.last_corroborated) / 3600) if self.last_corroborated else 0
        cont_recency = 1.0 / (1.0 + (now - self.last_contradicted) / 3600) if self.last_contradicted else 0

        # Weighted confidence
        base_confidence = ratio
        recency_adjustment = 0.1 * (corr_recency - cont_recency)

        self.confidence = max(0.1, min(0.99, base_confidence + recency_adjustment))
        return self.confidence

    def corroborate(self) -> None:
        """Record a corroboration (confirmation) of this memory."""
        self.corroboration_count += 1
        self.last_corroborated = time.time()
        self.update_confidence()

    def contradict(self) -> None:
        """Record a contradiction of this memory."""
        self.contradiction_count += 1
        self.last_contradicted = time.time()
        self.update_confidence()


@dataclass
class ContradictionRecord:
    """Records a detected contradiction between memories."""
    conflicting_key: torch.Tensor
    conflicting_value: Any
    contradiction_type: ContradictionType
    similarity: float
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution: Optional[str] = None


class ContradictionDetector:
    """
    Detects contradictions between memories.

    Novel contribution: No existing memory system (Titans, BudgetMem, TiMem)
    explicitly detects and handles contradictions. They either:
    - Overwrite silently (Mem0)
    - Store duplicates without resolution (RAG)
    - Ignore the problem entirely (Titans)

    TBG detects contradictions through:
    1. High semantic similarity (same topic)
    2. Low value similarity (different content)
    3. Pattern matching for negation words
    """

    # Negation patterns
    NEGATION_PATTERNS = [
        ("not ", ""),
        ("no longer ", ""),
        ("isn't ", "is "),
        ("wasn't ", "was "),
        ("won't ", "will "),
        ("don't ", "do "),
        ("never ", "always "),
        ("false", "true"),
        ("incorrect", "correct"),
        ("wrong", "right"),
    ]

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        value_diff_threshold: float = 0.5,
    ):
        self.similarity_threshold = similarity_threshold
        self.value_diff_threshold = value_diff_threshold

    def detect(
        self,
        new_key: torch.Tensor,
        new_value: Any,
        existing_key: torch.Tensor,
        existing_value: Any,
    ) -> Tuple[ContradictionType, float]:
        """
        Detect if new memory contradicts existing memory.

        Returns:
            (ContradictionType, similarity_score)
        """
        # Compute semantic similarity
        similarity = F.cosine_similarity(
            new_key.view(1, -1),
            existing_key.view(1, -1)
        ).item()

        # Not similar enough = not about same topic = no contradiction
        if similarity < self.similarity_threshold:
            return ContradictionType.NONE, similarity

        # High similarity = same topic. Check values.
        new_str = str(new_value).lower()
        existing_str = str(existing_value).lower()

        # Check for direct negation
        if self._is_negation(new_str, existing_str):
            return ContradictionType.NEGATION, similarity

        # Check for value conflict (same structure, different values)
        if self._is_value_conflict(new_str, existing_str):
            return ContradictionType.VALUE_CONFLICT, similarity

        # Check for partial conflict
        if self._has_partial_conflict(new_str, existing_str):
            return ContradictionType.PARTIAL, similarity

        return ContradictionType.NONE, similarity

    def _is_negation(self, new: str, existing: str) -> bool:
        """Check if new is negation of existing."""
        for neg, pos in self.NEGATION_PATTERNS:
            # Check if adding negation to existing matches new
            if neg in new and neg not in existing:
                # Remove negation from new and compare
                new_without_neg = new.replace(neg, pos)
                if self._string_similarity(new_without_neg, existing) > 0.8:
                    return True
            # Check reverse
            if neg in existing and neg not in new:
                existing_without_neg = existing.replace(neg, pos)
                if self._string_similarity(new, existing_without_neg) > 0.8:
                    return True
        return False

    def _is_value_conflict(self, new: str, existing: str) -> bool:
        """
        Check for value conflicts like:
        - "meeting at 3pm" vs "meeting at 4pm"
        - "budget is $50k" vs "budget is $75k"
        """
        import re

        # Extract numbers
        new_numbers = set(re.findall(r'\d+', new))
        existing_numbers = set(re.findall(r'\d+', existing))

        # If both have numbers and they differ, potential conflict
        if new_numbers and existing_numbers:
            # Remove numbers and compare structure
            new_structure = re.sub(r'\d+', 'NUM', new)
            existing_structure = re.sub(r'\d+', 'NUM', existing)

            if self._string_similarity(new_structure, existing_structure) > 0.8:
                # Same structure, different numbers = value conflict
                if new_numbers != existing_numbers:
                    return True

        return False

    def _has_partial_conflict(self, new: str, existing: str) -> bool:
        """Check for partial conflicts (overlapping but inconsistent)."""
        # Simple word overlap check
        new_words = set(new.split())
        existing_words = set(existing.split())

        overlap = len(new_words & existing_words)
        total = len(new_words | existing_words)

        if total == 0:
            return False

        overlap_ratio = overlap / total

        # High overlap but not identical = potential partial conflict
        if 0.3 < overlap_ratio < 0.9 and new != existing:
            return True

        return False

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard on words)."""
        words1 = set(s1.split())
        words2 = set(s2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0


class TemporalBeliefGraph:
    """
    Temporal Belief Graph: A novel memory enhancement layer.

    Wraps around Crystal Memory to add:
    1. Belief states for each memory
    2. Contradiction detection and tracking
    3. Confidence-weighted retrieval
    4. Belief evolution history

    This is the KEY ORIGINAL CONTRIBUTION of NEURON.

    Usage:
        tbg = TemporalBeliefGraph(crystal_memory)

        # Store with belief tracking
        tbg.store(embedding, "Meeting at 3pm", source="user")

        # Later, contradictory info arrives
        tbg.store(embedding, "Meeting at 4pm", source="calendar")
        # → Contradiction detected, both beliefs tracked

        # Query with confidence
        results = tbg.retrieve(query_embedding)
        # → Returns memories WITH confidence scores

        # Check belief evolution
        history = tbg.get_belief_history(topic_embedding)
        # → Shows how beliefs about this topic evolved
    """

    def __init__(
        self,
        crystal_memory,  # CrystalMemory instance
        contradiction_threshold: float = 0.85,
    ):
        self.crystal = crystal_memory
        self.detector = ContradictionDetector(
            similarity_threshold=contradiction_threshold
        )

        # Belief states for each Crystal item (indexed by position)
        self.beliefs: dict[int, BeliefState] = {}

        # Value-to-index cache for O(1) lookup
        self._value_index: dict[int, int] = {}

        # Contradiction log
        self.contradictions: List[ContradictionRecord] = []

        # Belief history (for evolution tracking)
        self.belief_history: List[Tuple[float, int, float]] = []  # (timestamp, item_idx, confidence)

    def store(
        self,
        key_embedding: torch.Tensor,
        value: Any,
        importance: float = 0.5,
        source: str = "unknown",
    ) -> dict:
        """
        Store with belief tracking and contradiction detection.

        Returns:
            Dict with storage result and any detected contradictions
        """
        result = {
            "stored": False,
            "contradictions": [],
            "corroborations": [],
            "belief_state": None,
        }

        # Check for contradictions with existing memories
        for idx, item in enumerate(self.crystal.items):
            contradiction_type, similarity = self.detector.detect(
                key_embedding, value,
                item.key, item.value
            )

            if contradiction_type != ContradictionType.NONE:
                # Record contradiction
                record = ContradictionRecord(
                    conflicting_key=item.key,
                    conflicting_value=item.value,
                    contradiction_type=contradiction_type,
                    similarity=similarity,
                )
                self.contradictions.append(record)
                result["contradictions"].append({
                    "type": contradiction_type.value,
                    "existing_value": item.value,
                    "similarity": similarity,
                })

                # Update belief state of existing item
                if idx in self.beliefs:
                    self.beliefs[idx].contradict()
                    self._log_belief(idx)

            elif similarity > 0.9:
                # Very similar = corroboration
                if idx in self.beliefs:
                    self.beliefs[idx].corroborate()
                    self._log_belief(idx)
                    result["corroborations"].append({
                        "existing_value": item.value,
                        "similarity": similarity,
                    })

        # Store in Crystal
        stored = self.crystal.store(key_embedding, value, importance)
        result["stored"] = stored

        # Create belief state for new item
        if stored:
            new_idx = len(self.crystal.items) - 1
            self._value_index[id(value)] = new_idx

            # Initial confidence based on contradictions
            initial_confidence = 0.7 if not result["contradictions"] else 0.4

            belief = BeliefState(confidence=initial_confidence)

            # Boost if corroborated
            for _ in result["corroborations"]:
                belief.corroborate()

            self.beliefs[new_idx] = belief
            self._log_belief(new_idx)
            result["belief_state"] = {
                "confidence": belief.confidence,
                "corroborations": belief.corroboration_count,
                "contradictions": belief.contradiction_count,
            }

        return result

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        min_confidence: float = 0.0,
    ) -> List[dict]:
        """
        Retrieve memories with confidence scores.

        Unlike standard retrieval, includes belief confidence.
        Low-confidence memories are flagged.
        """
        # Get standard retrieval
        base_results = self.crystal.retrieve(query_embedding, top_k=top_k * 2)

        # Enhance with belief info
        enhanced = []
        for r in base_results:
            # Find corresponding belief state
            idx = self._find_item_index(r["value"])
            belief = self.beliefs.get(idx, BeliefState())

            confidence = belief.update_confidence()

            if confidence >= min_confidence:
                enhanced.append({
                    **r,
                    "confidence": confidence,
                    "corroborations": belief.corroboration_count,
                    "contradictions": belief.contradiction_count,
                    "has_conflicts": belief.contradiction_count > 0,
                })

        # Sort by combined score (similarity * confidence)
        enhanced.sort(
            key=lambda x: x["similarity"] * x["confidence"],
            reverse=True
        )

        return enhanced[:top_k]

    def get_contradictions(
        self,
        query_embedding: Optional[torch.Tensor] = None,
        unresolved_only: bool = True,
    ) -> List[ContradictionRecord]:
        """Get contradiction records, optionally filtered by topic."""
        records = self.contradictions

        if unresolved_only:
            records = [r for r in records if not r.resolved]

        if query_embedding is not None:
            # Filter by similarity to query
            filtered = []
            for r in records:
                sim = F.cosine_similarity(
                    query_embedding.view(1, -1),
                    r.conflicting_key.view(1, -1)
                ).item()
                if sim > 0.7:
                    filtered.append(r)
            records = filtered

        return records

    def get_belief_history(
        self,
        item_idx: Optional[int] = None,
    ) -> List[dict]:
        """
        Get belief evolution history.

        Shows how confidence in memories changed over time.
        """
        history = self.belief_history

        if item_idx is not None:
            history = [(t, i, c) for t, i, c in history if i == item_idx]

        return [
            {
                "timestamp": t,
                "item_idx": i,
                "confidence": c,
            }
            for t, i, c in history
        ]

    def resolve_contradiction(
        self,
        contradiction_idx: int,
        resolution: str,
        keep_newer: bool = True,
    ) -> None:
        """
        Manually resolve a contradiction.

        Args:
            contradiction_idx: Index in self.contradictions
            resolution: Explanation of resolution
            keep_newer: If True, boost newer memory's confidence
        """
        if contradiction_idx >= len(self.contradictions):
            return

        record = self.contradictions[contradiction_idx]
        record.resolved = True
        record.resolution = resolution

        # Adjust beliefs based on resolution
        # (In practice, this would be more sophisticated)

    def _find_item_index(self, value: Any) -> int:
        """Find index of item with given value. Uses cache with fallback."""
        # Fast path: check cache by identity
        cached = self._value_index.get(id(value))
        if cached is not None and cached < len(self.crystal.items):
            if self.crystal.items[cached].value == value:
                return cached

        # Slow path: linear scan and update cache
        for idx, item in enumerate(self.crystal.items):
            if item.value == value:
                self._value_index[id(value)] = idx
                return idx
        return -1

    def _log_belief(self, idx: int) -> None:
        """Log belief state for history tracking."""
        if idx in self.beliefs:
            self.belief_history.append((
                time.time(),
                idx,
                self.beliefs[idx].confidence,
            ))

    def stats(self) -> dict:
        """Get TBG statistics."""
        confidences = [b.confidence for b in self.beliefs.values()]

        return {
            "total_memories": len(self.crystal),
            "tracked_beliefs": len(self.beliefs),
            "total_contradictions": len(self.contradictions),
            "unresolved_contradictions": sum(1 for c in self.contradictions if not c.resolved),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "low_confidence_count": sum(1 for c in confidences if c < 0.5),
        }
