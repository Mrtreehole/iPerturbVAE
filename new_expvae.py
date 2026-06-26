# -*- coding: utf-8 -*-
import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from models.stagevae import GeneOnlyAttnVAE_Conv


def load_data_from_h5(h5_path, dataset_key="ctls"):
    with h5py.File(h5_path, "r") as f:
        if dataset_key not in f:
            raise KeyError(
                f"Dataset key '{dataset_key}' not found in {h5_path}, "
                f"available keys: {list(f.keys())}"
            )
        data = f[dataset_key][:]
    return data


def build_model(device, latent_dim=128):
    model = GeneOnlyAttnVAE_Conv(
        gene_dim=978,
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
    model.eval()
    return model


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model", ckpt)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def compute_attention_for_batch(model, x_batch, use_cuda=True):
    device = next(model.parameters()).device
    if use_cuda and torch.cuda.is_available():
        x_batch = x_batch.to(device)

    model.eval()
    out = model(x_batch)
    z = out["z"]
    attn_seq = out["attn_seq"]

    batch_size, gene_dim, _ = attn_seq.shape
    latent_dim = z.size(1)
    mi_batch = torch.zeros(batch_size, latent_dim, gene_dim, device=device)

    for i in tqdm(range(latent_dim), desc="latent dims (Mi)", leave=False):
        model.zero_grad()
        score = z[:, i].sum()
        grads = torch.autograd.grad(
            outputs=score,
            inputs=attn_seq,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        alpha = grads.mean(dim=1)
        weighted = attn_seq * alpha.unsqueeze(1)
        mi_i = torch.relu(weighted.sum(dim=2))
        mi_batch[:, i, :] = mi_i

    m_batch = mi_batch.mean(dim=1)
    return mi_batch.detach(), m_batch.detach()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute VAE attention (Mi, M) for ctls/pred fold outputs."
    )
    parser.add_argument("--h5_path", type=str, required=True, help="Input H5 prediction file path")
    parser.add_argument(
        "--dataset_key",
        type=str,
        choices=["ctls", "pred"],
        default="ctls",
        help="Dataset key inside the prediction H5 file",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoints/best_gene_only_attn_vae_convctl2ctl.pt",
        help="Checkpoint path for the gene-only VAE used to compute explanations",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npz path")
    parser.add_argument("--cpu", action="store_true", help="Force CPU computation")
    return parser.parse_args()


def main():
    args = parse_args()

    if (not args.cpu) and torch.cuda.is_available():
        device = torch.device("cuda")
        use_cuda = True
    else:
        device = torch.device("cpu")
        use_cuda = False

    print(f"Loading data from {args.h5_path} (dataset_key={args.dataset_key}) ...")
    data_np = load_data_from_h5(args.h5_path, dataset_key=args.dataset_key)
    num_samples, gene_dim = data_np.shape
    if gene_dim != 978:
        raise ValueError(f"Expected gene_dim=978, but got {gene_dim}")

    data_tensor = torch.from_numpy(data_np).float()

    print("Building model...")
    model = build_model(device=device, latent_dim=args.latent_dim)
    print(f"Loading checkpoint from {args.ckpt_path} ...")
    model = load_checkpoint(model, args.ckpt_path, device)

    all_mi_list = []
    all_m_list = []
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size

    outer_bar = tqdm(range(num_batches), desc="batches", leave=True)
    for batch_index in outer_bar:
        start = batch_index * args.batch_size
        end = min((batch_index + 1) * args.batch_size, num_samples)
        x_batch = data_tensor[start:end]

        mi_batch, m_batch = compute_attention_for_batch(
            model,
            x_batch,
            use_cuda=use_cuda,
        )
        all_mi_list.append(mi_batch.cpu())
        all_m_list.append(m_batch.cpu())

    mi_all = torch.cat(all_mi_list, dim=0).numpy()
    m_all = torch.cat(all_m_list, dim=0).numpy()

    print(f"Saving results to {args.output_path} ...")
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    np.savez_compressed(
        args.output_path,
        Mi=mi_all,
        M=m_all,
        dataset_key=np.array(args.dataset_key),
    )

    print("Done.")


if __name__ == "__main__":
    main()
