from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore


def torch_available() -> bool:
    return TORCH_OK


if TORCH_OK:

    class LearnableTemporalDecay(nn.Module):
        """Exponential decay over horizons — decay rate is learned."""

        def __init__(self, n_steps: int = 5):
            super().__init__()
            self.log_lambda = nn.Parameter(torch.tensor(0.5))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, C)
            lam = torch.sigmoid(self.log_lambda) * 0.9 + 0.05
            t = x.size(1)
            steps = torch.arange(t, device=x.device, dtype=x.dtype)
            w = torch.exp(-lam * (t - 1 - steps))
            w = w / w.sum()
            return x * w.view(1, t, 1)

    class MultiScaleTemporalCNN(nn.Module):
        """Dilated convolutions across accumulation horizons."""

        def __init__(self, in_ch: int = 4, hidden: int = 32):
            super().__init__()
            dilations = [1, 2, 4, 8]
            branches = []
            for d in dilations:
                branches.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, hidden, kernel_size=3, padding=d, dilation=d),
                        nn.GELU(),
                        nn.Conv1d(hidden, hidden, kernel_size=1),
                        nn.GELU(),
                    )
                )
            self.branches = nn.ModuleList(branches)
            self.proj = nn.Linear(hidden * len(dilations), hidden)
            self.decay = LearnableTemporalDecay()

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # x: (B, T, C) -> conv wants (B, C, T)
            x = self.decay(x)
            xc = x.transpose(1, 2)
            outs = [b(xc).mean(dim=2) for b in self.branches]
            h = torch.cat(outs, dim=1)
            h = self.proj(h)
            # Temporal attention weights (interpretable)
            attn_logits = (x ** 2).mean(dim=2)
            attn = F.softmax(attn_logits, dim=1)
            return h, attn

    class CrossModalAttention(nn.Module):
        """Bidirectional quant <-> semantic fusion."""

        def __init__(self, quant_dim: int, sem_dim: int, hidden: int = 64):
            super().__init__()
            self.q_proj = nn.Linear(quant_dim, hidden)
            self.s_proj = nn.Linear(sem_dim, hidden)
            self.cross_qs = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            self.cross_sq = nn.MultiheadAttention(hidden, num_heads=4, batch_first=True)
            self.out = nn.Linear(hidden * 2, hidden)

        def forward(
            self, quant: torch.Tensor, sem: torch.Tensor
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            q = self.q_proj(quant).unsqueeze(1)
            s = self.s_proj(sem).unsqueeze(1)
            qs, w_qs = self.cross_qs(q, s, s, need_weights=True)
            sq, w_sq = self.cross_sq(s, q, q, need_weights=True)
            fused = self.out(torch.cat([qs.squeeze(1), sq.squeeze(1)], dim=1))
            return fused, {"quant_to_sem": w_qs, "sem_to_quant": w_sq}

    class GraphPropagation(nn.Module):
        """Broker + feature similarity message passing."""

        def __init__(self, dim: int, steps: int = 2):
            super().__init__()
            self.steps = steps
            self.msg = nn.Linear(dim, dim)

        def forward(self, h: torch.Tensor, adj: torch.Tensor, node_idx: torch.Tensor) -> torch.Tensor:
            # h: (B, D), adj: (N, N), node_idx: (B,)
            n = adj.size(0)
            if n <= 1:
                return h
            H = torch.zeros(n, h.size(1), device=h.device, dtype=h.dtype)
            H.index_add_(0, node_idx, h)
            counts = torch.zeros(n, device=h.device).index_add_(0, node_idx, torch.ones_like(node_idx, dtype=h.dtype))
            counts = counts.clamp(min=1.0).unsqueeze(1)
            H = H / counts
            for _ in range(self.steps):
                H = torch.matmul(adj, H)
                H = F.gelu(self.msg(H))
            return H[node_idx]

    class NepseMultimodalNet(nn.Module):
        def __init__(self, sem_dim: int = 16, temporal_ch: int = 4, hidden: int = 64):
            super().__init__()
            self.cnn = MultiScaleTemporalCNN(in_ch=temporal_ch, hidden=hidden)
            self.cross = CrossModalAttention(quant_dim=hidden, sem_dim=sem_dim, hidden=hidden)
            self.graph = GraphPropagation(hidden)
            self.head = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden // 2, 1),
            )

        def forward(
            self,
            temporal: torch.Tensor,
            semantic: torch.Tensor,
            adj: torch.Tensor,
            node_idx: torch.Tensor,
        ) -> tuple[torch.Tensor, dict]:
            quant_h, temporal_attn = self.cnn(temporal)
            fused, cross_w = self.cross(quant_h, semantic)
            graph_h = self.graph(fused, adj, node_idx)
            logit = self.head(graph_h).squeeze(-1)
            interpret = {
                "temporal_attn": temporal_attn.detach(),
                "cross_modal": {k: v.detach() for k, v in cross_w.items()},
                "decay_lambda": torch.sigmoid(self.cnn.decay.log_lambda).item(),
            }
            return logit, interpret

    def phase_aware_bce_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (bce * weights).mean()
