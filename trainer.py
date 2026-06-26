# -*- coding: utf-8 -*-
# @Author: treehole (modified by assistant)
# @Date:   2026-01-07


import os
import time
import numpy as np
import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils.dataloader import H5GeneCompoundDataset
from utils.evaluation import compute_r2_pearson

# 这里改成你方案A的模型文件/类名
# 例如你把我给的模型保存为 models/geneonlyattnvae_conv.py：
# from models.geneonlyattnvae_conv import GeneOnlyAttnVAE_Conv, vae_loss
from models.stagevae import GeneOnlyAttnVAE_Conv, vae_loss  


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device


def cleanup_ddp():
    dist.destroy_process_group()


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, random_state=seed, shuffle=True
    )
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size_adjusted, random_state=seed, shuffle=True
    )



    print(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def make_optimizer(model, lr=1e-4, weight_decay=0):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _ddp_is_initialized():
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if _ddp_is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


@torch.no_grad()
def _gather_cat_tensor(x: torch.Tensor) -> torch.Tensor:

    if not _ddp_is_initialized():
        return x

    world_size = dist.get_world_size()
    device = x.device

    # gather batch sizes
    b = torch.tensor([x.size(0)], device=device, dtype=torch.long)
    bs = [torch.zeros_like(b) for _ in range(world_size)]
    dist.all_gather(bs, b)
    bs = [int(t.item()) for t in bs]
    max_b = max(bs)

    # pad to max_b
    if x.size(0) < max_b:
        pad = torch.zeros((max_b - x.size(0),) + x.shape[1:], device=device, dtype=x.dtype)
        x_pad = torch.cat([x, pad], dim=0)
    else:
        x_pad = x

    # all_gather padded
    gathered = [torch.zeros_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gathered, x_pad)

    # unpad and cat
    out = []
    for t, n in zip(gathered, bs):
        out.append(t[:n].detach())
    return torch.cat(out, dim=0)


def train_step(batch, model, optimizer, device, beta=0.05, recon_loss="mse", target_key="gene_ctl", grad_clip=1.0):
    model.train()
    x_gene = batch[target_key].to(device, non_blocking=True)

    out = model(x_gene)
    loss, logs = vae_loss(out["y_hat"], x_gene, out["mu"], out["logvar"], beta=beta, recon_loss=recon_loss)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "loss": float(logs["loss"].detach().cpu()),
        "recon": float(logs["recon"].detach().cpu()),
        "kl": float(logs["kl"].detach().cpu()),
    }


@torch.no_grad()
def evaluate(dataloader, model, device, beta=0.05, recon_loss="mse", target_key="gene_ctl"):
    """
    DDP 下：
      - loss/recon/kl：按样本数加权求全局平均（all_reduce sum）
      - r2/pearson：用全量预测/真值 gather 后计算（更严格，但更耗显存/CPU）
    """
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    total_samples = 0

    preds_local, trues_local = [], []

    for batch in dataloader:
        x_gene = batch[target_key].to(device, non_blocking=True)

        out = model(x_gene)
        loss, logs = vae_loss(out["y_hat"], x_gene, out["mu"], out["logvar"], beta=beta, recon_loss=recon_loss)

        bs = x_gene.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_recon += float(logs["recon"].detach().cpu()) * bs
        total_kl += float(logs["kl"].detach().cpu()) * bs
        total_samples += bs

        preds_local.append(out["y_hat"].detach())
        trues_local.append(x_gene.detach())

    # ---- all_reduce for scalar metrics ----
    totals = torch.tensor([total_loss, total_recon, total_kl, total_samples], device=device, dtype=torch.float64)
    totals = _all_reduce_sum(totals)
    total_loss_g, total_recon_g, total_kl_g, total_samples_g = totals.tolist()
    total_samples_g = max(1.0, total_samples_g)

    # ---- gather for r2/pearson ----
    y_pred_local = torch.cat(preds_local, dim=0) if len(preds_local) else torch.empty((0, 978), device=device)
    y_true_local = torch.cat(trues_local, dim=0) if len(trues_local) else torch.empty((0, 978), device=device)

    y_pred = _gather_cat_tensor(y_pred_local).cpu().numpy()
    y_true = _gather_cat_tensor(y_true_local).cpu().numpy()
    r2, pearson_corr = compute_r2_pearson(y_true, y_pred)

    return {
        "loss": total_loss_g / total_samples_g,
        "recon": total_recon_g / total_samples_g,
        "kl": total_kl_g / total_samples_g,
        "r2": r2,
        "pearson": pearson_corr,
    }


def main():
    local_rank, device = setup_ddp()
    is_main = local_rank == 0

    if is_main:
        #train_subset, val_subset, test_subset = split_dataset(full_dataset, seed=seed)##
        writer = SummaryWriter(log_dir="./runs/gene_only_attn_vae_conv_ddp")
        os.makedirs("./checkpoints", exist_ok=True)

    # ===== data =====
    h5_path = "../cbfp_vae/dataprocess/data_qcpass_unique_by_ctl.h5"
    full_dataset = H5GeneCompoundDataset(h5_path=h5_path, compound_dim=3200)

    # ===== split =====
    if is_main:
        
        train_idx = np.load("./data_splits/train_indicesall.npy")
        val_idx = np.load("./data_splits/val_indicesall.npy")
        test_idx = np.load("./data_splits/test_indicesall.npy")
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        test_subset = Subset(full_dataset, test_idx)

    # ===== optional: train fraction =====
    use_train_fraction = 1  # e.g. 0.1
    if is_main:
        train_idx_full = train_subset.indices
        rng = np.random.default_rng(seed)
        n_keep = max(1, int(len(train_idx_full) * use_train_fraction))
        train_idx_small = rng.choice(train_idx_full, size=n_keep, replace=False)
        np.save("./data_splits/train_indices_small.npy", train_idx_small)
        dist.barrier()
        print(f"Using {n_keep}/{len(train_idx_full)} ({use_train_fraction:.0%}) train samples")
    else:
        dist.barrier()
        train_idx_small = np.load("./data_splits/train_indices_small.npy")
    train_subset = Subset(full_dataset, train_idx_small)

    # ===== loaders =====
    epochs = 200
    batch_size = 32
    num_workers = 64
    patience = 15
    min_delta = 1e-4

    train_sampler = DistributedSampler(train_subset, shuffle=True, seed=seed, drop_last=False)
    val_sampler = DistributedSampler(val_subset, shuffle=False, seed=seed, drop_last=False)
    test_sampler = DistributedSampler(test_subset, shuffle=False, seed=seed, drop_last=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # ===== model =====
    model = GeneOnlyAttnVAE_Conv(
        gene_dim=978,
        latent_dim=128,
        attn_d_model=128,
        attn_heads=8,
        attn_dropout=0,
        use_positional_embedding=True,
        conv_channels=(256, 512),
        conv_kernel_size=3,
        conv_dropout=0,
        conv_use_batchnorm=False,#这玩意不能加，否则效果贼78差
        decoder_hidden=(256,512),
        decoder_dropout=0,
        decoder_activation="relu",
        decoder_use_layernorm=True,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = make_optimizer(model, lr=5e-4, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    beta_max = 1e-3
    warmup_epochs = 50
    recon_loss = "mse"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        torch.cuda.synchronize()
        start_time = time.time()

        beta = beta_max * min(1.0, epoch / warmup_epochs)

        total_train_loss = total_train_rec = total_train_kl = 0.0
        for batch in train_loader:
            m = train_step(
                batch, model, optimizer, device,
                beta=beta, recon_loss=recon_loss,
                target_key="gene_ctl", grad_clip=1.0
            )
            total_train_loss += m["loss"]
            total_train_rec += m["recon"]
            total_train_kl += m["kl"]

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        avg_train_rec = total_train_rec / max(1, len(train_loader))
        avg_train_kl = total_train_kl / max(1, len(train_loader))

        val_metrics = evaluate(val_loader, model, device, beta=beta, recon_loss=recon_loss, target_key="gene_ctl")
        scheduler.step(val_metrics["loss"])

        if is_main:
            elapsed = time.time() - start_time
            print("\n" + "=" * 70)
            print(f"Epoch {epoch}/{epochs} | beta={beta:.4f}")
            print(f"  Train - Loss: {avg_train_loss:.4f} | Recon: {avg_train_rec:.4f} | KL: {avg_train_kl:.4f}")
            print(
                f"  Valid - Loss: {val_metrics['loss']:.4f} | Recon: {val_metrics['recon']:.4f} | "
                f"KL: {val_metrics['kl']:.4f} | R2: {val_metrics['r2']:.4f} | Pearson: {val_metrics['pearson']:.4f}"
            )
            print(f"  Time: {elapsed:.2f}s")

            writer.add_scalar("Train/Loss", avg_train_loss, epoch)
            writer.add_scalar("Train/Recon", avg_train_rec, epoch)
            writer.add_scalar("Train/KL", avg_train_kl, epoch)
            writer.add_scalar("Train/beta", beta, epoch)

            writer.add_scalar("Val/Loss", val_metrics["loss"], epoch)
            writer.add_scalar("Val/Recon", val_metrics["recon"], epoch)
            writer.add_scalar("Val/KL", val_metrics["kl"], epoch)
            writer.add_scalar("Val/R2", val_metrics["r2"], epoch)
            writer.add_scalar("Val/Pearson", val_metrics["pearson"], epoch)

            if val_metrics["loss"] < best_val_loss - min_delta:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                patience_counter = 0
                ckpt = {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "beta_max": beta_max,
                    "warmup_epochs": warmup_epochs,
                    "recon_loss": recon_loss,
                    "model_name": "GeneOnlyAttnVAE_Conv",
                }
                torch.save(ckpt, "./checkpoints/512best_gene_only_attn_vae_convctl2ctl.pt")
                print(f"  New best saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    if is_main:
        print("\n" + "=" * 70)
        print(f"Best epoch: {best_epoch} | best val loss: {best_val_loss:.6f}")

        ckpt = torch.load("./checkpoints/512best_gene_only_attn_vae_convctl2ctl.pt", map_location="cpu")
        model.module.load_state_dict(ckpt["model"])

        test_metrics = evaluate(test_loader, model, device, beta=beta_max, recon_loss=recon_loss, target_key="gene_ctl")
        print(
            f"Test - Loss: {test_metrics['loss']:.4f} | Recon: {test_metrics['recon']:.4f} | "
            f"KL: {test_metrics['kl']:.4f} | R2: {test_metrics['r2']:.4f} | Pearson: {test_metrics['pearson']:.4f}"
        )
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
