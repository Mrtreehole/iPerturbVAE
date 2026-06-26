# -*- coding: utf-8 -*-
# @Author: treehole 
# @Date:   2026-01-07


from typing import Sequence, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: Sequence[int],
        out_dim: int,
        dropout: float = 0.05,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ):
        super().__init__()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # per-sample KL summed over latent dims, then mean over batch
    kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
    return kl.sum(dim=-1).mean()


def vae_loss(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1e-3,
    recon_loss: str = "mse",
):
    if recon_loss == "mse":
        recon = F.mse_loss(y_hat, y_true)
    elif recon_loss == "l1":
        recon = F.l1_loss(y_hat, y_true, reduction="mean")
    else:
        raise ValueError("recon_loss must be 'mse' or 'l1'")
    kl = kl_divergence(mu, logvar)
    loss = recon + beta * kl
    return loss, {"loss": loss, "recon": recon, "kl": kl}


class GeneSelfAttention(nn.Module):
    """
    输入 gene 向量 (B, gene_dim)；把 gene_dim 当成序列长度，每个基因是一个 token。
    (B,gene_dim)->(B,gene_dim,1)->proj->(B,gene_dim,d_model)->MHSA->输出序列 (B,gene_dim,d_model)
    """
    def __init__(
        self,
        gene_dim: int = 978,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_positional_embedding: bool = True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.gene_dim = gene_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(1, d_model)

        self.use_positional_embedding = use_positional_embedding
        if use_positional_embedding:
            self.pos_emb = nn.Parameter(torch.zeros(1, gene_dim, d_model))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x_gene: torch.Tensor) -> torch.Tensor:
        if x_gene.dim() != 2 or x_gene.size(1) != self.gene_dim:
            raise ValueError(f"x_gene must be (B,{self.gene_dim}), got {tuple(x_gene.shape)}")

        x = x_gene.unsqueeze(-1)      # (B,gene_dim,1)
        x = self.in_proj(x)           # (B,gene_dim,d_model)
        if self.pos_emb is not None:
            x = x + self.pos_emb

        # Pre-LN attention block
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # Pre-LN FFN block
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x  # (B,gene_dim,d_model)


class GeneConvEncoder(nn.Module):
    """
    对注意力输出序列做 1D 卷积编码：
      输入: (B, L=gene_dim, C=d_model)
      转置: (B, C, L)
      Conv1d -> GELU/Dropout -> Conv1d -> ... -> Pool -> (B, channels, 1)
      -> Linear -> (B, latent_dim)
    """
    def __init__(
        self,
        gene_dim: int = 978,
        d_model: int = 128,
        latent_dim: int = 128,
        channels: Sequence[int] = (256, 512),
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_batchnorm: bool = False,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.d_model = d_model
        self.latent_dim = latent_dim

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size建议用奇数，便于 same padding；请传入奇数如 3/5/7")

        pad = kernel_size // 2

        layers = []
        in_ch = d_model
        for out_ch in channels:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_ch, latent_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, gene_dim, d_model)
        if x_seq.dim() != 3 or x_seq.size(1) != self.gene_dim or x_seq.size(2) != self.d_model:
            raise ValueError(
                f"x_seq must be (B,{self.gene_dim},{self.d_model}), got {tuple(x_seq.shape)}"
            )

        x = x_seq.transpose(1, 2)      # (B,d_model,gene_dim)
        x = self.conv(x)               # (B,channels,gene_dim)
        x = self.pool(x).squeeze(-1)   # (B,channels)
        x = self.proj(x)               # (B,latent_dim)
        return x


class GeneOnlyAttnVAE_Conv(nn.Module):
    """
    方案A版本：
      gene(978) -> self-attn -> (B,978,d_model)
      -> Conv1d encoder -> (B,latent_dim)
      -> mu/logvar -> z(latent_dim)
      -> decoder(MLP) -> y_hat(978)

    返回：
      y_hat: (B,978)
      mu/logvar/z: (B,latent_dim)
      attn_seq: (B,978,d_model)
      enc: (B,latent_dim)  (mu/logvar 之前的编码向量)
    """
    def __init__(
        self,
        gene_dim: int = 978,
        latent_dim: int = 128,
        attn_d_model: int = 128,
        attn_heads: int = 8,
        attn_dropout: float = 0.1,
        use_positional_embedding: bool = True,
        # conv encoder
        conv_channels: Sequence[int] = (256, 512),
        conv_kernel_size: int = 5,
        conv_dropout: float = 0.1,
        conv_use_batchnorm: bool = False,
        # decoder
        decoder_hidden: Sequence[int] = (1024, 2048),
        decoder_dropout: float = 0.05,
        decoder_activation: str = "gelu",
        decoder_use_layernorm: bool = True,
    ):
        super().__init__()
        self.gene_dim = gene_dim
        self.latent_dim = latent_dim
        self.attn_d_model = attn_d_model

        self.gene_attn = GeneSelfAttention(
            gene_dim=gene_dim,
            d_model=attn_d_model,
            n_heads=attn_heads,
            dropout=attn_dropout,
            use_positional_embedding=use_positional_embedding,
        )

        self.encoder = GeneConvEncoder(
            gene_dim=gene_dim,
            d_model=attn_d_model,
            latent_dim=latent_dim,
            channels=conv_channels,
            kernel_size=conv_kernel_size,
            dropout=conv_dropout,
            use_batchnorm=conv_use_batchnorm,
        )

        self.to_mu = nn.Linear(latent_dim, latent_dim)
        self.to_logvar = nn.Linear(latent_dim, latent_dim)

        self.decoder = MLP(
            in_dim=latent_dim,
            hidden=decoder_hidden,
            out_dim=gene_dim,
            dropout=decoder_dropout,
            activation=decoder_activation,
            use_layernorm=decoder_use_layernorm,
        )

    def forward(self, x_gene: torch.Tensor) -> Dict[str, torch.Tensor]:
        attn_seq = self.gene_attn(x_gene)      # (B,gene_dim,d_model)
        enc = self.encoder(attn_seq)           # (B,latent_dim)
        mu = self.to_mu(enc)
        logvar = self.to_logvar(enc)
        z = reparameterize(mu, logvar)
        y_hat = self.decoder(z)                # (B,gene_dim)
        return {"y_hat": y_hat, "mu": mu, "logvar": logvar, "z": z, "attn_seq": attn_seq, "enc": enc}


# ---- 可选：一个最小自检 ----
def _smoke_test(device: str = "cpu") -> Tuple[torch.Size, torch.Size]:
    torch.manual_seed(0)
    model = GeneOnlyAttnVAE_Conv(
        gene_dim=978,
        latent_dim=256,
        attn_d_model=128,
        attn_heads=8,
        attn_dropout=0,
        use_positional_embedding=True,
        conv_channels=(256, 512),
        conv_kernel_size=5,
        conv_dropout=0.1,
        conv_use_batchnorm=False,
        decoder_hidden=(256, 512),
        decoder_dropout=0,
        decoder_activation="relu",
        decoder_use_layernorm=True,
    ).to(device)

    x = torch.randn(4, 978, device=device)
    out = model(x)
    y_hat = out["y_hat"]
    loss, logs = vae_loss(y_hat, x, out["mu"], out["logvar"], beta=0.05, recon_loss="mse")
    loss.backward()
    return y_hat.shape, out["attn_seq"].shape


if __name__ == "__main__":
    y_shape, attn_shape = _smoke_test("cpu")
    print("y_hat:", y_shape, "attn_seq:", attn_shape)
