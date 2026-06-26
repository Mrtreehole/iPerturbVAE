import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Model components
# =========================

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        act: str = "gelu",
    ):
        super().__init__()
        dims = (in_dim,) + tuple(hidden_dims) + (out_dim,)
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if act == "gelu":
                layers.append(nn.GELU())
            elif act == "relu":
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Unsupported act: {act}")
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompoundEncoder(nn.Module):
    """
    3200 -> 1024 -> 512 -> 256 -> 128
    """
    def __init__(self, in_dim: int = 1024, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dims=(512, 256),
            out_dim=out_dim,
            dropout=dropout,
            use_layernorm=True,
            act="gelu",
        )

    def forward(self, compound_x: torch.Tensor) -> torch.Tensor:
        return self.mlp(compound_x)


class ConditionalDecoder(nn.Module):
    """
    Decoder that takes [z | c] as input.
    """
    def __init__(
        self,
        z_dim: int,
        c_dim: int = 128,
        out_dim: int = 978,
        hidden_dims: Tuple[int, ...] = (512, 1024),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = MLP(
            in_dim=z_dim + c_dim,
            hidden_dims=hidden_dims,
            out_dim=out_dim,
            dropout=dropout,
            use_layernorm=True,
            act="gelu",
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=-1))


# =========================
# Conditional VAE (new)
# =========================

class CompoundConditionalVAE(nn.Module):
    """
    Conditional VAE for ctl+compound -> trt.

    Components:
      - frozen_gene_to_z: gene_ctl -> z0 (typically mu of pretrained gene VAE). No grad.
      - compound_encoder: compound -> c (trainable)
      - posterior_net: [z0 | c] -> (mu, logvar) of q(z | z0, c) (trainable)
      - decoder: [z | c] -> gene_trt recon (trainable)

    Forward returns:
      {
        "z0": z0, "c": c,
        "mu": mu, "logvar": logvar,
        "z": z, "recon": recon
      }

    Usage:
      - training: sample_z=True
      - eval: sample_z=False (use mu for deterministic recon)
    """
    def __init__(
        self,
        frozen_gene_to_z: nn.Module,
        gene_out_dim: int,            # output dim of decoder (e.g., 978)
        z_dim: int,                   # latent dim for conditional VAE
        compound_in_dim: int = 3200,
        compound_emb_dim: int = 128,
        compound_dropout: float = 0.1,
        posterior_hidden_dims: Tuple[int, ...] = (256, 512),
        posterior_dropout: float = 0.1,
        decoder_hidden_dims: Tuple[int, ...] = (512, 1024),
        decoder_dropout: float = 0.1,
        clamp_logvar: Optional[Tuple[float, float]] = (-12.0, 12.0),
    ):
        super().__init__()
        self.z_dim = z_dim
        self.clamp_logvar = clamp_logvar

        self.frozen_gene_to_z = frozen_gene_to_z
        self.compound_encoder = CompoundEncoder(in_dim=compound_in_dim, out_dim=compound_emb_dim, dropout=compound_dropout)

        # q(z | z0, c): output 2*z_dim (mu, logvar)
        self.posterior_net = MLP(
            in_dim=z_dim + compound_emb_dim,
            hidden_dims=posterior_hidden_dims,
            out_dim=2 * z_dim,
            dropout=posterior_dropout,
            use_layernorm=True,
            act="gelu",
        )

        self.decoder = ConditionalDecoder(
            z_dim=z_dim,
            c_dim=compound_emb_dim,
            out_dim=gene_out_dim,
            hidden_dims=decoder_hidden_dims,
            dropout=decoder_dropout,
        )

        self.freeze_gene_to_z()

    def freeze_gene_to_z(self):
        self.frozen_gene_to_z.eval()
        for p in self.frozen_gene_to_z.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_gene_to_z0(self, gene_x: torch.Tensor) -> torch.Tensor:
        # keep no_grad to avoid building graph through frozen part
        return self.frozen_gene_to_z(gene_x)

    def encode_posterior(self, z0: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        h = torch.cat([z0, c], dim=-1)          # (B, z_dim + c_dim)
        out = self.posterior_net(h)             # (B, 2*z_dim)
        mu, logvar = out.chunk(2, dim=-1)       # 各 (B, z_dim)

    # 可选：数值稳定（不想改行为就删掉这两行）
        if self.clamp_logvar is not None:
            lo, hi = self.clamp_logvar
            logvar = logvar.clamp(lo, hi)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps
 
    def forward(self, gene_x: torch.Tensor, compound_x: torch.Tensor, sample_z: bool = True) -> Dict[str, torch.Tensor]:
        z0 = self.encode_gene_to_z0(gene_x)                # (B, z_dim) frozen
        c = self.compound_encoder(compound_x)              # (B, compound_emb_dim) trainable
        mu, logvar = self.encode_posterior(z0, c)          # (B, z_dim), (B, z_dim)
        z = self.reparameterize(mu, logvar) if sample_z else mu
        recon = self.decoder(z, c)
        return {"z0": z0, "c": c, "mu": mu, "logvar": logvar, "z": z, "recon": recon}


# =========================
# Loss & metrics (optional helpers)
# =========================

def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|.) || N(0, I)) averaged over batch.
    """
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return kl.mean()


def smooth_l1_recon_loss(recon: torch.Tensor, target: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    # beta here is SmoothL1's transition point, NOT KL beta.
    return F.smooth_l1_loss(recon, target, beta=beta, reduction="mean")


def r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Per-sample R2 over feature dimension, then mean over batch
    ss_res = ((y_true - y_pred) ** 2).sum(dim=-1)
    y_mean = y_true.mean(dim=-1, keepdim=True)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=-1).clamp_min(eps)
    return (1.0 - ss_res / ss_tot).mean()


def pearsonr_torch(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Per-sample Pearson over feature dimension, then mean over batch
    x = y_true - y_true.mean(dim=-1, keepdim=True)
    y = y_pred - y_pred.mean(dim=-1, keepdim=True)
    num = (x * y).sum(dim=-1)
    den = (x.pow(2).sum(dim=-1) * y.pow(2).sum(dim=-1)).sqrt().clamp_min(eps)
    return (num / den).mean()


class FrozenGeneToZAdapter(nn.Module):
    def __init__(self, pretrained_model: nn.Module, use_mu: bool = True):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.use_mu = use_mu

        self.pretrained_model.eval()
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, gene_x: torch.Tensor) -> torch.Tensor:
        out = self.pretrained_model(gene_x)

        # dict: {"mu": ..., "logvar": ...}
        if isinstance(out, dict):
            mu = out.get("mu", None)
            logvar = out.get("logvar", None)
            if mu is None:
                raise KeyError(f"pretrained_model output dict missing 'mu'. keys={list(out.keys())}")
            if self.use_mu or (logvar is None):
                return mu
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps

        # tuple/list: (mu, logvar) or (mu,)
        if isinstance(out, (tuple, list)):
            if len(out) == 0:
                raise ValueError("pretrained_model returned empty tuple/list")
            mu = out[0]
            if self.use_mu or len(out) == 1:
                return mu
            logvar = out[1]
            eps = torch.randn_like(mu)
            return mu + torch.exp(0.5 * logvar) * eps

        # tensor: treat as mu
        if torch.is_tensor(out):
            return out

        raise TypeError(f"Unsupported output type from pretrained_model: {type(out)}")
