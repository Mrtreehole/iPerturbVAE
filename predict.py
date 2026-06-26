import argparse
import os

import h5py
import torch
from torch.utils.data import DataLoader

from utils.dataloader import H5GeneCompoundDataset
from models.stagevae import GeneOnlyAttnVAE_Conv
from models.conditionvae import CompoundConditionalVAE, FrozenGeneToZAdapter


RANDOM_CKPT_TEMPLATE = "best_ctl_compound_to_trt_condvae_fold{fold}.pt"
DISEASE_CKPT_TEMPLATE = "best_ctl_compound_to_trt_condvaesplitbydisease_fold{fold}.pt"


def load_frozen_gene_to_z(device, ckpt_path, gene_dim=978, latent_dim=128):
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
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    base.load_state_dict(state, strict=True)

    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    gene_to_z = FrozenGeneToZAdapter(pretrained_model=base, use_mu=True).to(device)
    gene_to_z.eval()
    for p in gene_to_z.parameters():
        p.requires_grad = False

    return gene_to_z


def build_model(device, pretrained_gene_ckpt, cond_ckpt_path, z_dim, compound_dim):
    gene_to_z = load_frozen_gene_to_z(
        device,
        pretrained_gene_ckpt,
        gene_dim=978,
        latent_dim=z_dim,
    )

    model = CompoundConditionalVAE(
        frozen_gene_to_z=gene_to_z,
        gene_out_dim=978,
        z_dim=z_dim,
        compound_in_dim=compound_dim,
        compound_emb_dim=128,
        compound_dropout=0,
        posterior_hidden_dims=(128, 256, 512),
        posterior_dropout=0.05,
        decoder_hidden_dims=(128, 258, 512),
        decoder_dropout=0.05,
        clamp_logvar=(-12.0, 12.0),
    ).to(device)

    ckpt = torch.load(cond_ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def resolve_ckpt_path(split_type, fold, ckpt_dir):
    if split_type == "random":
        filename = RANDOM_CKPT_TEMPLATE.format(fold=fold)
    else:
        filename = DISEASE_CKPT_TEMPLATE.format(fold=fold)
    return os.path.join(ckpt_dir, filename)


def read_ckpt_metadata(ckpt_path, fallback_pretrained, fallback_z_dim, fallback_compound_dim):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_ckpt = ckpt.get("pretrained_ckpt", fallback_pretrained)
    z_dim = int(ckpt.get("z_dim", fallback_z_dim))
    compound_dim = int(ckpt.get("compound_dim", fallback_compound_dim))
    return pretrained_ckpt, z_dim, compound_dim


@torch.no_grad()
def predict(model, dataloader, device):
    preds = []
    trues = []
    ctls = []

    for batch in dataloader:
        gene_ctl = batch["gene_ctl"].float().to(device)
        gene_trt = batch["gene_trt"].float().to(device)
        compound = batch["compound"].float().to(device)

        out = model(gene_x=gene_ctl, compound_x=compound, sample_z=False)
        y_hat = out["recon"]

        ctls.append(gene_ctl.cpu())
        preds.append(y_hat.cpu())
        trues.append(gene_trt.cpu())

    ctls = torch.cat(ctls, dim=0).numpy()
    preds = torch.cat(preds, dim=0).numpy()
    trues = torch.cat(trues, dim=0).numpy()
    return ctls, preds, trues


def save_h5(ctls, preds, trues, out_path, split_type, fold, ckpt_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("ctls", data=ctls)
        f.create_dataset("pred", data=preds)
        f.create_dataset("true", data=trues)
        f.attrs["split_type"] = split_type
        f.attrs["fold"] = int(fold)
        f.attrs["checkpoint_path"] = ckpt_path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run predictions for random/disease 5-fold checkpoints on the full dataset."
    )
    ap.add_argument("--h5_path", type=str, default="./dataprocess/data_qcpass_filtered.h5")
    ap.add_argument("--split_type", type=str, choices=["random", "disease"], required=True)
    ap.add_argument("--fold", type=int, choices=[1, 2, 3, 4, 5], required=True)
    ap.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    ap.add_argument("--cond_ckpt", type=str, default=None)
    ap.add_argument(
        "--pretrained_gene_ckpt",
        type=str,
        default="./checkpoints/best_gene_only_attn_vae_convctl2ctl.pt",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--compound_dim", type=int, default=3200)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="./results_cv5")
    ap.add_argument("--out_h5", type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.cond_ckpt or resolve_ckpt_path(args.split_type, args.fold, args.ckpt_dir)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pretrained_gene_ckpt, z_dim, compound_dim = read_ckpt_metadata(
        ckpt_path=ckpt_path,
        fallback_pretrained=args.pretrained_gene_ckpt,
        fallback_z_dim=args.z_dim,
        fallback_compound_dim=args.compound_dim,
    )

    dataset = H5GeneCompoundDataset(h5_path=args.h5_path, compound_dim=compound_dim)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(
        device=device,
        pretrained_gene_ckpt=pretrained_gene_ckpt,
        cond_ckpt_path=ckpt_path,
        z_dim=z_dim,
        compound_dim=compound_dim,
    )

    ctls, preds, trues = predict(model, loader, device)

    out_h5 = args.out_h5 or os.path.join(
        args.out_dir,
        f"pred_{args.split_type}_fold{args.fold}.h5",
    )
    save_h5(ctls, preds, trues, out_h5, args.split_type, args.fold, ckpt_path)

    print(f"Saved predictions to: {out_h5}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Shape: {preds.shape}")


if __name__ == "__main__":
    main()
