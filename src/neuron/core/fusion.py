"""
Fusion Layer: Combines Crystal and Morph outputs.

Also implements Horizontal Cross-Memory Consolidation (HCMC):
- Crystal patterns influence Morph organization
- Morph context improves Crystal retrieval
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """
    Adaptive fusion of Crystal and Morph outputs.

    Uses learned gating to balance:
    - Crystal: Exact facts, specific details
    - Morph: Context, relationships, general understanding

    The gate learns when to trust each source based on query type.

    Args:
        d_crystal: Crystal output dimension
        d_morph: Morph output dimension
        d_output: Final output dimension
    """

    def __init__(
        self,
        d_crystal: int = 768,
        d_morph: int = 64,
        d_output: int = 768,
    ):
        super().__init__()

        # Project to common space
        self.crystal_proj = nn.Linear(d_crystal, d_output)
        self.morph_proj = nn.Linear(d_morph, d_output)

        # Adaptive gate
        self.gate = nn.Sequential(
            nn.Linear(d_output * 2, d_output),
            nn.ReLU(),
            nn.Linear(d_output, 1),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Linear(d_output, d_output)

    def forward(
        self,
        crystal_output: torch.Tensor,
        morph_output: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse Crystal and Morph outputs.

        Args:
            crystal_output: Output from Crystal Memory
            morph_output: Context vector from Morph Layer
            query: Query embedding (for query-aware gating)

        Returns:
            Fused context tensor
        """
        # Project to common space
        c = self.crystal_proj(crystal_output)
        m = self.morph_proj(morph_output)

        # Compute adaptive gate
        combined = torch.cat([c, m], dim=-1)
        gate_value = self.gate(combined)

        # Weighted fusion
        fused = gate_value * c + (1 - gate_value) * m

        # Final projection
        output = self.output_proj(fused)

        return output


class CrossMemoryConsolidator:
    """
    Horizontal Cross-Memory Consolidation (HCMC)

    Bidirectional influence between Crystal and Morph:

    1. Crystal → Morph: Frequently accessed Crystal patterns
       inform Morph's semantic organization

    2. Morph → Crystal: Morph's context understanding
       improves Crystal retrieval ranking

    This runs periodically (not every inference) as a background process.
    """

    def __init__(
        self,
        crystal_weight: float = 0.3,
        morph_weight: float = 0.3,
        d_crystal: int = 768,
        d_morph: int = 64,
    ):
        self.crystal_weight = crystal_weight
        self.morph_weight = morph_weight
        self.d_crystal = d_crystal
        self.d_morph = d_morph

        # Projection layer for dimension alignment (crystal → morph space)
        self._crystal_to_morph = torch.nn.Linear(d_crystal, d_morph, bias=False)
        # Initialize with random orthogonal weights for better gradients
        torch.nn.init.orthogonal_(self._crystal_to_morph.weight)

    def consolidate(
        self,
        crystal_memory,
        morph_layer,
    ) -> dict:
        """
        Run consolidation between Crystal and Morph.

        Args:
            crystal_memory: CrystalMemory instance
            morph_layer: MorphLayer instance

        Returns:
            Stats about consolidation
        """
        stats = {
            "crystal_patterns_extracted": 0,
            "morph_context_used": 0,
            "items_reranked": 0,
        }

        # Extract frequently accessed patterns from Crystal
        frequent_keys = self._extract_frequent_patterns(crystal_memory)
        stats["crystal_patterns_extracted"] = len(frequent_keys)

        # Use patterns to bias Morph centroid
        if frequent_keys:
            pattern_centroid = torch.stack(frequent_keys).mean(dim=0)
            morph_layer.centroid = (
                (1 - self.crystal_weight) * morph_layer.centroid +
                self.crystal_weight * pattern_centroid
            )
            stats["morph_context_used"] = 1

        # Use Morph context to rerank Crystal items
        morph_context = morph_layer.get_context_vector()
        if morph_context.abs().max() > 1e-6:
            self._rerank_crystal_by_context(crystal_memory, morph_context)
            stats["items_reranked"] = len(crystal_memory)

        return stats

    def _extract_frequent_patterns(
        self,
        crystal_memory,
        min_access: int = 3,
    ) -> list[torch.Tensor]:
        """Get keys from frequently accessed Crystal items."""
        frequent = []
        for item in crystal_memory.items:
            if item.access_count >= min_access:
                frequent.append(item.key)
        return frequent

    def _rerank_crystal_by_context(
        self,
        crystal_memory,
        morph_context: torch.Tensor,
    ) -> None:
        """
        Boost importance of Crystal items that align with Morph context.

        Fixed: Uses proper projection instead of truncation to preserve semantics.
        """
        if len(crystal_memory.items) == 0:
            return

        # Project Crystal keys to Morph space for proper comparison
        for item in crystal_memory.items:
            # Project crystal key (d_crystal) to morph space (d_morph)
            with torch.no_grad():
                projected_key = self._crystal_to_morph(item.key.unsqueeze(0)).squeeze(0)

            # Compute similarity in the shared morph space
            similarity = F.cosine_similarity(
                projected_key.unsqueeze(0),
                morph_context.unsqueeze(0),
            ).item()

            # Small boost based on context alignment (only positive adjustments)
            if similarity > 0:
                item.importance = min(
                    1.0,
                    item.importance + self.morph_weight * similarity * 0.1
                )
