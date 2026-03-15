"""
NEURON: Neural Encoding with Unified Recurrent Optimized Network

Main orchestrator class that combines:
- Crystal Memory (exact symbolic storage)
- Morph Layer (neural continuous memory)
- Importance Router (SMTR)
- Fusion Layer (adaptive combination)
- Cross-Memory Consolidation (HCMC)
- Temporal Belief Graph (TBG) - NOVEL CONTRIBUTION

TBG is the key original contribution: tracks belief states, detects
contradictions, and enables confidence-weighted retrieval.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuron.config import NeuronConfig
from neuron.core.belief import TemporalBeliefGraph
from neuron.core.crystal import CrystalMemory
from neuron.core.fusion import CrossMemoryConsolidator, FusionLayer
from neuron.core.morph import MorphLayer
from neuron.core.router import ImportanceRouter, RouteDecision


class NEURON(nn.Module):
    """
    NEURON: Hybrid memory system for LLMs.

    Combines exact recall (Crystal) with infinite-capacity fuzzy
    understanding (Morph), using surprise-modulated routing (SMTR)
    and cross-memory consolidation (HCMC).

    Args:
        config: NeuronConfig instance with all parameters.
        d_model: Embedding dimension (default: 768). Ignored if config is provided.
        crystal_size: Max items in Crystal (default: 10000). Ignored if config is provided.
        morph_rank: Rank for Morph low-rank approximation (default: 64). Ignored if config is provided.
        consolidation_interval: Steps between consolidation (default: 100). Ignored if config is provided.

    Example:
        >>> from neuron import NEURON
        >>> from neuron.config import NeuronConfig
        >>> # Using config object (recommended)
        >>> config = NeuronConfig(d_model=512, crystal_size=5000)
        >>> memory = NEURON(config=config)
        >>> # Or using keyword arguments directly
        >>> memory = NEURON(d_model=512, crystal_size=5000)
    """

    def __init__(
        self,
        config: Optional[NeuronConfig] = None,
        *,
        d_model: int = 768,
        crystal_size: int = 10000,
        morph_rank: int = 64,
        consolidation_interval: int = 100,
    ):
        super().__init__()

        # Build config from kwargs if not provided
        if config is None:
            config = NeuronConfig(
                d_model=d_model,
                crystal_size=crystal_size,
                morph_rank=morph_rank,
                consolidation_interval=consolidation_interval,
            )

        self.config = config
        self.d_model = config.d_model
        self.consolidation_interval = config.consolidation_interval
        self._step_count = 0

        # Core components
        self.crystal = CrystalMemory(
            max_items=config.crystal_size,
            embedding_dim=config.d_model,
            similarity_threshold=config.similarity_threshold,
        )
        self.morph = MorphLayer(
            d_model=config.d_model,
            rank=config.morph_rank,
            base_decay=config.morph_base_decay,
            learning_rate=config.morph_learning_rate,
        )
        self.router = ImportanceRouter(
            d_model=config.d_model,
            critical_threshold=config.critical_threshold,
            surprise_threshold=config.surprise_threshold,
        )
        self.fusion = FusionLayer(
            d_crystal=config.d_model,
            d_morph=config.morph_rank,
            d_output=config.d_model,
        )
        self.consolidator = CrossMemoryConsolidator(
            d_crystal=config.d_model,
            d_morph=config.morph_rank,
        )

        # Temporal Belief Graph - NOVEL CONTRIBUTION
        self.belief_graph = TemporalBeliefGraph(
            crystal_memory=self.crystal,
            contradiction_threshold=config.contradiction_threshold,
        )

        # Context tracking
        self._last_embedding: Optional[torch.Tensor] = None

    def absorb(
        self,
        embedding: torch.Tensor,
        text: Optional[str] = None,
    ) -> dict:
        """
        Absorb new information into memory.

        Args:
            embedding: Content embedding [d_model] or [batch, d_model]
            text: Original text (optional, for pattern matching)

        Returns:
            Dict with routing decision and stats
        """
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        # Route the information
        routing = self.router(
            embedding=embedding,
            text=text,
            context_embedding=self._last_embedding,
        )

        stats = {
            "decision": routing.decision.value,
            "importance": routing.importance,
            "surprise": routing.surprise,
            "stored_crystal": False,
            "updated_morph": False,
            "contradictions": [],  # TBG: detected contradictions
            "belief_confidence": None,  # TBG: confidence in stored memory
        }

        # Execute routing decision
        # Now using TBG for Crystal storage to track beliefs and contradictions
        if routing.decision == RouteDecision.CRYSTAL_HIGH:
            # High priority crystal storage with TBG
            tbg_result = self.belief_graph.store(
                embedding.mean(dim=0),
                text or embedding,
                importance=min(1.0, routing.importance * 1.2),
            )
            stats["stored_crystal"] = tbg_result["stored"]
            stats["contradictions"] = tbg_result["contradictions"]
            if tbg_result["belief_state"]:
                stats["belief_confidence"] = tbg_result["belief_state"]["confidence"]

        elif routing.decision == RouteDecision.CRYSTAL:
            # Normal crystal storage with TBG
            tbg_result = self.belief_graph.store(
                embedding.mean(dim=0),
                text or embedding,
                importance=routing.importance,
            )
            stats["stored_crystal"] = tbg_result["stored"]
            stats["contradictions"] = tbg_result["contradictions"]
            if tbg_result["belief_state"]:
                stats["belief_confidence"] = tbg_result["belief_state"]["confidence"]

        elif routing.decision == RouteDecision.BOTH:
            # Dual write (hedge) with TBG
            tbg_result = self.belief_graph.store(
                embedding.mean(dim=0),
                text or embedding,
                importance=routing.importance * 0.8,
            )
            stats["stored_crystal"] = tbg_result["stored"]
            stats["contradictions"] = tbg_result["contradictions"]
            if tbg_result["belief_state"]:
                stats["belief_confidence"] = tbg_result["belief_state"]["confidence"]
            self.morph(embedding)
            stats["updated_morph"] = True

        else:  # MORPH
            # Morph only
            self.morph(embedding)
            stats["updated_morph"] = True

        # Update context tracking
        self._last_embedding = embedding.mean(dim=0).detach()

        # Periodic consolidation
        self._step_count += 1
        if self._step_count % self.consolidation_interval == 0:
            self._consolidate()

        return stats

    def query(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> dict:
        """
        Query the memory system.

        Args:
            query_embedding: Query embedding
            top_k: Number of Crystal results to retrieve

        Returns:
            Dict with crystal_results, morph_context, and fused_output
        """
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)

        # Get Crystal results
        crystal_results = self.crystal.retrieve(
            query_embedding.mean(dim=0),
            top_k=top_k,
        )

        # Get Morph context
        morph_context = self.morph.get_context_vector()

        # Prepare Crystal output - combine retrieved embeddings weighted by similarity
        if crystal_results:
            # Get embeddings from stored items and weight by similarity
            similarities = torch.tensor([r['similarity'] for r in crystal_results])
            weights = F.softmax(similarities, dim=0)

            # Retrieve stored keys as embeddings
            crystal_embeddings = []
            for r in crystal_results:
                # Find the matching item in crystal to get its key (embedding)
                for item in self.crystal.items:
                    if item.value == r['value']:
                        crystal_embeddings.append(item.key)
                        break

            if crystal_embeddings:
                crystal_stack = torch.stack(crystal_embeddings)
                crystal_output = (weights.unsqueeze(1) * crystal_stack).sum(dim=0)
            else:
                crystal_output = query_embedding.mean(dim=0)
        else:
            crystal_output = torch.zeros(self.d_model)

        # Fuse outputs
        fused = self.fusion(
            crystal_output=crystal_output,
            morph_output=morph_context,
            query=query_embedding.mean(dim=0),
        )

        return {
            "crystal_results": crystal_results,
            "morph_context": morph_context,
            "fused_output": fused,
        }

    def _consolidate(self) -> None:
        """Run cross-memory consolidation."""
        self.consolidator.consolidate(self.crystal, self.morph)

    def reset_session(self) -> None:
        """Reset Morph for new session (keep Crystal)."""
        self.morph.reset()
        self._last_embedding = None

    def reset_all(self) -> None:
        """Full reset (both memories)."""
        self.crystal.clear()
        self.morph.reset()
        self._last_embedding = None
        self._step_count = 0
        # Reset TBG
        self.belief_graph = TemporalBeliefGraph(
            crystal_memory=self.crystal,
            contradiction_threshold=self.config.contradiction_threshold,
        )

    def stats(self) -> dict:
        """Get memory statistics including TBG stats."""
        tbg_stats = self.belief_graph.stats()
        return {
            "crystal_items": len(self.crystal),
            "crystal_capacity": self.crystal.max_items,
            "morph_tokens_seen": self.morph.n_tokens.item(),
            "total_steps": self._step_count,
            # TBG stats
            "tbg_contradictions": tbg_stats["total_contradictions"],
            "tbg_unresolved": tbg_stats["unresolved_contradictions"],
            "tbg_avg_confidence": tbg_stats["avg_confidence"],
        }

    def get_contradictions(self, query_embedding: Optional[torch.Tensor] = None) -> list:
        """
        Get detected contradictions, optionally filtered by topic.

        This is a KEY FEATURE of TBG - no other memory system provides this.
        """
        return self.belief_graph.get_contradictions(
            query_embedding=query_embedding,
            unresolved_only=True,
        )

    def get_belief_history(self, item_idx: Optional[int] = None) -> list:
        """
        Get belief evolution history.

        Shows how confidence in memories changed over time.
        """
        return self.belief_graph.get_belief_history(item_idx)
