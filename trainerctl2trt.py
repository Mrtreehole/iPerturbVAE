import os
import time
import numpy as np
import torch

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils.dataloader import H5GeneCompoundDataset
from utils.evaluation import compute_r2_pearson

from models.stagevae import GeneOnlyAttnVAE_Conv
from models.conditionvae import (
    CompoundConditionalVAE,
    FrozenGeneToZAdapter,
    kl_standard_normal,
)

import torch.nn.functional as F

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

def setup(device_str: str = "cuda", index: int = 0):
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def make_optimizer(params, lr=1e-4, weight_decay=0.0):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def load_frozen_gene_to_z(device, ckpt_path: str, gene_dim=978, latent_dim=128):
    """
    加载上一阶段 GeneOnlyAttnVAE checkpoint，并冻结为 gene_x -> z 的模块（默认用 mu）。
    兼容 ckpt 保存为 {"model": state_dict} 或直接 state_dict。
    同时兼容 DDP 保存的 "module." 前缀。
    """
    base = GeneOnlyAttnVAE_Conv(
        gene_dim=gene_dim,
        latent_dim=latent_dim,
        attn_d_model=128,
        attn_heads=8,
        attn_dropout=0,
        use_positional_embedding=True,
        conv_channels=(256, 512),
        conv_kernel_size=3,
        conv_dropout=0,
        conv_use_batchnorm=False,
        decoder_hidden=(256, 512),
        decoder_dropout=0,
        decoder_activation="relu",
        decoder_use_layernorm=True,
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt

    # strip DDP prefix if present
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module.") :]: v for k, v in state.items()}

    base.load_state_dict(state, strict=True)

    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    gene_to_z = FrozenGeneToZAdapter(pretrained_model=base, use_mu=True).to(device)
    gene_to_z.eval()
    for p in gene_to_z.parameters():
        p.requires_grad = False
    return gene_to_z


def _move(batch: dict, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def beta_schedule(epoch: int, beta_max: float = 1e-3, warmup_epochs: int = 20) -> float:
    if warmup_epochs <= 0:
        return float(beta_max)
    t = min(1.0, max(0.0, epoch / float(warmup_epochs)))
    return float(beta_max * t)


def train_step(
    batch,
    model,
    optimizer,
    device,
    epoch: int,
    grad_clip=1.0,
    beta_max: float = 1e-3,
    warmup_epochs: int = 20,
):
    model.train()

    batch = _move(batch, device)
    x_gene_ctl = batch["gene_ctl"].float()
    x_gene_trt = batch["gene_trt"].float()
    x_comp = batch["compound"].float()

    out = model(gene_x=x_gene_ctl, compound_x=x_comp, sample_z=True)
    y_hat = out["recon"]

    recon = F.mse_loss(y_hat, x_gene_trt)
    kl = kl_standard_normal(out["mu"], out["logvar"])
    beta = beta_schedule(epoch, beta_max=beta_max, warmup_epochs=warmup_epochs)
    loss = recon + beta * kl

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
    optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "recon": float(recon.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "beta": float(beta),
    }


@torch.no_grad()
def evaluate(
    dataloader,
    model,
    device,
    epoch: int = 200,
    beta_max: float = 1e-3,
    warmup_epochs: int = 20,
):
    model.eval()

    total_recon = 0.0
    total_kl = 0.0
    total_loss = 0.0
    total_samples = 0

    preds_local, trues_local = [], []
    beta = beta_schedule(epoch, beta_max=beta_max, warmup_epochs=warmup_epochs)

    for batch in dataloader:
        batch = _move(batch, device)
        x_gene_ctl = batch["gene_ctl"].float()
        x_gene_trt = batch["gene_trt"].float()
        x_comp = batch["compound"].float()

        out = model(gene_x=x_gene_ctl, compound_x=x_comp, sample_z=False)
        y_hat = out["recon"]

        recon = F.mse_loss(y_hat, x_gene_trt)
        kl = kl_standard_normal(out["mu"], out["logvar"])
        loss = recon + beta * kl

        bs = x_gene_trt.size(0)
        total_recon += float(recon.detach().cpu()) * bs
        total_kl += float(kl.detach().cpu()) * bs
        total_loss += float(loss.detach().cpu()) * bs
        total_samples += bs

        preds_local.append(y_hat.detach().cpu())
        trues_local.append(x_gene_trt.detach().cpu())

    loss_mean = total_loss / max(1, total_samples)
    recon_mean = total_recon / max(1, total_samples)
    kl_mean = total_kl / max(1, total_samples)

    preds_local = (
        torch.cat(preds_local, dim=0)
        if len(preds_local)
        else torch.empty((0, 978), dtype=torch.float32)
    )
    trues_local = (
        torch.cat(trues_local, dim=0)
        if len(trues_local)
        else torch.empty((0, 978), dtype=torch.float32)
    )

    y_pred = preds_local.numpy()
    y_true = trues_local.numpy()
    r2, pearson_corr = compute_r2_pearson(y_true, y_pred)

    return {
        "loss": loss_mean,
        "recon": recon_mean,
        "kl": kl_mean,
        "beta": float(beta),
        "r2": r2,
        "pearson": pearson_corr,
    }


def main():
    device = setup("cuda")

    writer = SummaryWriter(log_dir="./runs/ctl_plus_compound_to_trt_condvae_single")
    os.makedirs("./checkpoints", exist_ok=True)

    # =====================
    # Data split
    # =====================
    h5_path = "/home/liuxiaoping/cbfp_vae/dataprocess/data_qcpass_filtered.h5"
    full_dataset = H5GeneCompoundDataset(h5_path=h5_path, compound_dim=3200)

    spl_dir = "./data_splits/splits_idx"
    os.makedirs(spl_dir, exist_ok=True)

    train_idx_path = os.path.join(spl_dir, "train_indices.npy")
    val_idx_path = os.path.join(spl_dir, "val_indices.npy")
    test_idx_path = os.path.join(spl_dir, "test_indices.npy")

    # 单进程：每次都可复用同一份 split；若文件不存在就生成
    if (not os.path.exists(train_idx_path)) or (not os.path.exists(val_idx_path)) or (not os.path.exists(test_idx_path)):
        n_samples = len(full_dataset)
        indices = np.arange(n_samples)

        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.15, random_state=seed, shuffle=True
        )
        val_size_adjusted = 0.15 / (0.7 + 0.15)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            random_state=seed,
            shuffle=True,
        )

        np.save(train_idx_path, train_idx)
        np.save(val_idx_path, val_idx)
        np.save(test_idx_path, test_idx)

    train_idx = np.load(train_idx_path)
    val_idx = np.load(val_idx_path)
    test_idx = np.load(test_idx_path)

    print(
        f"Dataset split (ctl2trt): Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
    )

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    test_subset = Subset(full_dataset, test_idx)

    # =====================
    # Loaders
    # =====================
    epochs = 200
    batch_size = 64
    num_workers = 32
    patience = 25
    min_delta = 1e-4

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # =====================
    # Model
    # =====================
    pretrained_ckpt = "./checkpoints/best_gene_only_attn_vae_convctl2ctl.pt"
    z_dim = 128

    gene_to_z = load_frozen_gene_to_z(device, pretrained_ckpt, gene_dim=978, latent_dim=z_dim)

    cond_model = CompoundConditionalVAE(
        frozen_gene_to_z=gene_to_z,
        gene_out_dim=978,
        z_dim=z_dim,
        compound_in_dim=3200,
        compound_emb_dim=128,
        compound_dropout=0,
        posterior_hidden_dims=(128, 256,512),
        posterior_dropout=0.05,
        decoder_hidden_dims=(128, 258,512),
        decoder_dropout=0.05,
        clamp_logvar=(-12.0, 12.0),
    ).to(device)

    # 只优化 requires_grad=True 的参数（保证 frozen 部分不会进优化器）
    optimizer = make_optimizer(
        (p for p in cond_model.parameters() if p.requires_grad),
        lr=1e-3,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0

    # =====================
    # Train
    # =====================
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_beta = 0.0

        for batch in train_loader:
            m = train_step(
                batch,
                cond_model,
                optimizer,
                device,
                epoch=epoch,
                grad_clip=1.0,
                beta_max=1e-3,
                warmup_epochs=20,
            )
            total_loss += m["loss"]
            total_recon += m["recon"]
            total_kl += m["kl"]
            total_beta += m["beta"]

        avg_loss = total_loss / max(1, len(train_loader))
        avg_recon = total_recon / max(1, len(train_loader))
        avg_kl = total_kl / max(1, len(train_loader))
        avg_beta = total_beta / max(1, len(train_loader))

        val_metrics = evaluate(val_loader, cond_model, device, epoch=epoch, beta_max=1e-3, warmup_epochs=20)
        scheduler.step(val_metrics["loss"])

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Epoch {epoch}/{epochs}")
        print(
            f"  Train - loss: {avg_loss:.4f} | recon: {avg_recon:.4f} | kl: {avg_kl:.4f} | beta: {avg_beta:.6g}"
        )
        print(
            f"  Valid - loss: {val_metrics['loss']:.4f} | recon: {val_metrics['recon']:.4f} | kl: {val_metrics['kl']:.4f} | "
            f"R2: {val_metrics['r2']:.4f} | Pearson: {val_metrics['pearson']:.4f}"
        )
        print(f"  Time: {elapsed:.2f}s")

        writer.add_scalar("Train/Loss", avg_loss, epoch)
        writer.add_scalar("Train/Recon", avg_recon, epoch)
        writer.add_scalar("Train/KL", avg_kl, epoch)
        writer.add_scalar("Train/Beta", avg_beta, epoch)

        writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/Recon", val_metrics["recon"], epoch)
        writer.add_scalar("Val/KL", val_metrics["kl"], epoch)
        writer.add_scalar("Val/R2", val_metrics["r2"], epoch)
        writer.add_scalar("Val/Pearson", val_metrics["pearson"], epoch)

        if val_metrics["loss"] < best_val - min_delta:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            ckpt = {
                "epoch": epoch,
                "model": cond_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val": best_val,
                "pretrained_ckpt": pretrained_ckpt,
                "z_dim": z_dim,
                "compound_dim": 3200,
                "input": "gene_ctl+compound",
                "target": "gene_trt",
                "loss": "mse+kl(beta_warmup)",
            }
            torch.save(ckpt, "./checkpoints/pc3best_ctl_compound_to_trt_condvae.pt")
            print(f"  New best saved (val_loss={best_val:.4f})")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # =====================
    # Test
    # =====================
    print("\n" + "=" * 70)
    print(f"Best epoch: {best_epoch} | best val loss: {best_val:.6f}")

    ckpt = torch.load("./checkpoints/pc3best_ctl_compound_to_trt_condvae.pt", map_location="cpu")
    cond_model.load_state_dict(ckpt["model"], strict=True)

    test_metrics = evaluate(test_loader, cond_model, device, epoch=best_epoch, beta_max=1e-3, warmup_epochs=20)
    print(
        f"Test - loss: {test_metrics['loss']:.4f} | recon: {test_metrics['recon']:.4f} | kl: {test_metrics['kl']:.4f} | "
        f"R2: {test_metrics['r2']:.4f} | Pearson: {test_metrics['pearson']:.4f}"
    )

    writer.close()


if __name__ == "__main__":
    main()