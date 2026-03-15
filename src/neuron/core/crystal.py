"""
Crystal Memory: Exact-recall symbolic memory for critical information.

Key Features:
- Key-value store with semantic indexing
- 100% exact retrieval for stored items
- Importance-weighted LRU eviction
- Max capacity with smart overflow handling
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F


@dataclass
class MemoryItem:
    """Single item in Crystal Memory."""
    key: torch.Tensor  # Semantic embedding
    value: Any         # Raw content (text, structured data, etc.)
    importance: float  # Importance score from router
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0


class CrystalMemory:
    """
    Crystal Memory: Symbolic key-value store with semantic retrieval.

    Unlike neural memory, Crystal provides exact recall - what you store
    is exactly what you get back. Used for critical information like
    names, dates, numbers, code snippets, and specific facts.

    Args:
        max_items: Maximum number of items to store (default: 10000)
        embedding_dim: Dimension of key embeddings (default: 768)
        similarity_threshold: Min similarity for retrieval (default: 0.5)
    """

    def __init__(
        self,
        max_items: int = 10000,
        embedding_dim: int = 768,
        similarity_threshold: float = 0.5,
    ):
        self.max_items = max_items
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold

        self.items: list[MemoryItem] = []

    def store(
        self,
        key_embedding: torch.Tensor,
        value: Any,
        importance: float = 0.5,
    ) -> bool:
        """
        Store a new item in Crystal Memory.

        Args:
            key_embedding: Semantic embedding of the content
            value: The actual content to store
            importance: Importance score (0-1)

        Returns:
            True if stored, False if duplicate detected
        """
        # Normalize key
        key = F.normalize(key_embedding.view(1, -1), dim=-1).squeeze()

        # Check for duplicates
        for i, existing in enumerate(self.items):
            similarity = F.cosine_similarity(
                key.unsqueeze(0),
                existing.key.unsqueeze(0)
            ).item()

            if similarity > 0.95:
                # Update existing item
                self.items[i].value = value
                self.items[i].importance = max(existing.importance, importance)
                self.items[i].timestamp = time.time()
                return False

        # Evict if at capacity
        if len(self.items) >= self.max_items:
            self._evict_lru()

        # Store new item
        item = MemoryItem(
            key=key,
            value=value,
            importance=importance,
        )
        self.items.append(item)
        return True

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve most relevant items for a query.

        Args:
            query_embedding: Query semantic embedding
            top_k: Number of items to retrieve

        Returns:
            List of dicts with 'value', 'similarity', 'importance'
        """
        if not self.items:
            return []

        # Normalize query
        query = F.normalize(query_embedding.view(1, -1), dim=-1).squeeze()

        # Compute similarities
        keys = torch.stack([item.key for item in self.items])
        similarities = F.cosine_similarity(query.unsqueeze(0), keys)

        # Get top-k
        k = min(top_k, len(self.items))
        top_sims, top_indices = similarities.topk(k)

        results = []
        for sim, idx in zip(top_sims.tolist(), top_indices.tolist()):
            if sim >= self.similarity_threshold:
                item = self.items[idx]
                item.access_count += 1
                item.timestamp = time.time()

                results.append({
                    "value": item.value,
                    "similarity": sim,
                    "importance": item.importance,
                })

        return results

    def _evict_lru(self) -> None:
        """Evict least recently used item, weighted by importance."""
        if not self.items:
            return

        # Score = recency × importance × access_frequency
        now = time.time()
        scores = []
        for item in self.items:
            recency = 1.0 / (1.0 + now - item.timestamp)
            access_freq = 1.0 + item.access_count
            score = recency * item.importance * access_freq
            scores.append(score)

        # Remove item with lowest score
        min_idx, _ = min(enumerate(scores), key=lambda x: x[1])
        self.items.pop(min_idx)

    def __len__(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        """Clear all items."""
        self.items.clear()
