"""
NEURON Configuration.

Centralized configuration for all NEURON components.
"""

from dataclasses import dataclass


@dataclass
class NeuronConfig:
    """
    Configuration for the NEURON memory system.

    Args:
        d_model: Embedding dimension (must match your encoder output).
        crystal_size: Maximum items in Crystal Memory.
        morph_rank: Rank for Morph low-rank decomposition.
        consolidation_interval: Steps between cross-memory consolidation.
        critical_threshold: Importance threshold for Crystal routing (0-1).
        surprise_threshold: Surprise threshold for dual-write routing (0-1).
        contradiction_threshold: Similarity threshold for contradiction detection (0-1).
        similarity_threshold: Minimum similarity for Crystal retrieval (0-1).
        morph_base_decay: Base decay factor for Morph state matrix.
        morph_learning_rate: Learning rate for Morph weight updates.
    """

    # Dimensions
    d_model: int = 768
    crystal_size: int = 10000
    morph_rank: int = 64

    # Scheduling
    consolidation_interval: int = 100

    # Router thresholds
    critical_threshold: float = 0.6
    surprise_threshold: float = 0.5

    # Belief / contradiction
    contradiction_threshold: float = 0.85

    # Crystal
    similarity_threshold: float = 0.5

    # Morph
    morph_base_decay: float = 0.95
    morph_learning_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.crystal_size <= 0:
            raise ValueError(f"crystal_size must be positive, got {self.crystal_size}")
        if self.morph_rank <= 0:
            raise ValueError(f"morph_rank must be positive, got {self.morph_rank}")
        if not 0 < self.critical_threshold < 1:
            raise ValueError(f"critical_threshold must be in (0, 1), got {self.critical_threshold}")
        if not 0 < self.surprise_threshold < 1:
            raise ValueError(f"surprise_threshold must be in (0, 1), got {self.surprise_threshold}")
