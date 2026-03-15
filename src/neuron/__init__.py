"""
NEURON: Neural Encoding with Unified Recurrent Optimized Network

Hybrid memory architecture combining exact recall (Crystal)
with infinite-capacity fuzzy understanding (Morph) for LLMs.

Key Innovation: Temporal Belief Graph (TBG) - tracks belief states,
detects contradictions, and enables confidence-weighted retrieval.
"""

__version__ = "0.3.0"

from neuron.config import NeuronConfig
from neuron.core.neuron import NEURON
from neuron.core.crystal import CrystalMemory
from neuron.core.morph import MorphLayer
from neuron.core.router import ImportanceRouter
from neuron.core.fusion import FusionLayer
from neuron.core.belief import TemporalBeliefGraph, ContradictionType, BeliefState

__all__ = [
    "NEURON",
    "NeuronConfig",
    "CrystalMemory",
    "MorphLayer",
    "ImportanceRouter",
    "FusionLayer",
    "TemporalBeliefGraph",
    "ContradictionType",
    "BeliefState",
]
