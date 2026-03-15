"""
Morph Layer: Neural state matrix for infinite-capacity fuzzy memory.

Key Features:
- Fixed-size state matrix (constant memory regardless of input)
- HyperNetwork-based weight updates
- Semantic-aware adaptive decay
- Continuous encoding of context and relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    """
    Small network that generates weight updates from input.

    Maps input embedding to delta-W matrix via low-rank decomposition.
    """

    def __init__(self, d_model: int = 512, rank: int = 64):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        self.down = nn.Linear(d_model, rank)
        self.key_proj = nn.Linear(rank, rank)
        self.val_proj = nn.Linear(rank, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate delta-W from input.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            delta_W: Weight update [rank, rank]
        """
        h = F.gelu(self.down(x))  # [batch, rank]
        k = self.key_proj(h)       # [batch, rank]
        v = self.val_proj(h)       # [batch, rank]

        # Outer product: [batch, rank, 1] @ [batch, 1, rank] = [batch, rank, rank]
        delta_W = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))

        # Average over batch
        return delta_W.mean(dim=0)  # [rank, rank]


class MorphLayer(nn.Module):
    """
    Morph Layer: Continuous neural memory with adaptive decay.

    Unlike Crystal (symbolic key-value), Morph stores information
    as weight modifications in a fixed-size matrix. This allows
    infinite capacity (in principle) but with fuzzy/approximate recall.

    The key equation:
        M(t) = λ × M(t-1) + η × ΔW

    Where:
        M = State matrix
        λ = Decay factor (adaptive based on semantic similarity)
        η = Learning rate
        ΔW = Weight update from HyperNetwork

    Args:
        d_model: Model dimension (default: 512)
        rank: Rank for low-rank decomposition (default: 64)
        base_decay: Base decay factor (default: 0.95)
        learning_rate: Update learning rate (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        rank: int = 64,
        base_decay: float = 0.95,
        learning_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.base_decay = base_decay
        self.learning_rate = learning_rate

        # HyperNetwork for generating weight updates
        self.hypernetwork = HyperNetwork(d_model, rank)

        # State matrix (the "memory")
        self.register_buffer("M", torch.zeros(rank, rank))

        # Context tracking for adaptive decay
        self.register_buffer("centroid", torch.zeros(d_model))
        self.register_buffer("n_tokens", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input and update state.

        Args:
            x: Input tensor [batch, d_model]

        Returns:
            Context vector derived from current state
        """
        # Generate weight update
        delta_W = self.hypernetwork(x)

        # Compute adaptive decay
        decay = self._compute_decay(x)

        # Update state matrix
        self.M = decay * self.M + self.learning_rate * delta_W

        # Update centroid
        self._update_centroid(x)

        # Return context vector
        return self.get_context_vector()

    def _compute_decay(self, x: torch.Tensor) -> float:
        """
        Compute adaptive decay based on semantic similarity.

        Higher similarity to past context = slower decay (keep memory)
        Lower similarity = faster decay (adapt to new topic)

        Fixed formula: Uses exponential scaling so decay actually varies meaningfully.
        - High similarity (0.8+) → decay ≈ 0.95 (preserve memory)
        - Low similarity (0.0)  → decay ≈ 0.90 (faster adaptation)
        - Negative similarity   → decay ≈ 0.80 (topic shift, rapid decay)
        """
        if self.n_tokens < 1:
            return self.base_decay

        # Compute similarity to context centroid
        x_mean = x.mean(dim=0)
        similarity = F.cosine_similarity(
            x_mean.unsqueeze(0),
            self.centroid.unsqueeze(0)
        ).item()

        # Fixed: Exponential decay scaling that actually varies
        # similarity in [-1, 1] maps to decay_factor in [0.8, 1.0]
        # Using: decay = base_decay ^ (2 - similarity)
        # High sim (1.0) → base_decay^1 = 0.95
        # Zero sim (0.0) → base_decay^2 = 0.9025
        # Low sim (-1.0) → base_decay^3 = 0.857
        decay_exponent = 2.0 - similarity  # Range: [1, 3]
        decay = self.base_decay ** decay_exponent

        # Clamp to reasonable range
        return max(0.5, min(0.99, decay))

    def _update_centroid(self, x: torch.Tensor) -> None:
        """Update running average of context."""
        x_mean = x.mean(dim=0)
        alpha = 1.0 / (self.n_tokens.item() + 1)
        self.centroid = (1 - alpha) * self.centroid + alpha * x_mean
        self.n_tokens += x.shape[0]

    def get_context_vector(self) -> torch.Tensor:
        """
        Extract context vector from state matrix.

        Uses SVD to get principal components.
        """
        if self.M.abs().max() < 1e-6:
            return torch.zeros(self.rank, device=self.M.device)

        # SVD decomposition
        U, S, Vh = torch.linalg.svd(self.M)

        # Use top-k components weighted by singular values
        k = min(10, self.rank)
        context = (U[:, :k] * S[:k]).sum(dim=1)

        return context

    def reset(self) -> None:
        """Reset state for new session."""
        self.M.zero_()
        self.centroid.zero_()
        self.n_tokens.zero_()
