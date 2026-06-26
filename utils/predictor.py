# -*- coding: utf-8 -*-
# @Author: treehole
# @Date:   2025-11-07
# @Description: 基于测试集的推理类，自动计算 MSE, R², 皮尔森相关系数 (Pearson r)

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Subset, DataLoader
from scipy.stats import pearsonr

from models.model import (
    CompoundMLPEncoder,
    GeneMHEncoderWithM,
    SingleTokenCrossAttention,
    CrossModalGeneBetaVAE
)
from utils.dataloader import H5GeneCompoundDataset


class GeneVAEPredictor:

    def __init__(
        self,
        checkpoint_path: str,
        mask_matrix_path: str,
        h5_path: str,
        test_idx_path: str,
        batch_size: int = 64,
        num_workers: int = 4,
        device: str = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = checkpoint_path
        self.mask_matrix_path = mask_matrix_path
        self.h5_path = h5_path
        self.test_idx_path = test_idx_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        # ---- 与训练保持一致 ----
        self.gene_dim = 978
        self.pathway_dim = 2062
        self.compound_dim = 3200
        self.d_model = 2062
        self.latent_dim = 64
        self.hidden = 512
        self.beta = 0.5
        self.aux_path_recon = False

        # === 初始化 ===
        self._load_mask_matrix()
        self._init_dataset()
        self._init_model()
        self._load_checkpoint()

        self.gene_encoder.eval()
        self.compound_encoder.eval()
        self.cross_attn.eval()
        self.vae.eval()
        print(f"✅ GeneVAEPredictorTestSet initialized on {self.device}")

    # -------------------------------------------------
    def _load_mask_matrix(self):
        mask_matrix = pd.read_csv(self.mask_matrix_path, index_col=0).values
        self.mask_matrix = torch.tensor(mask_matrix, dtype=torch.float32)
        print(f"📘 Loaded mask matrix: {self.mask_matrix.shape}")

    def _init_dataset(self):
        print("📂 Loading full dataset...")
        full_dataset = H5GeneCompoundDataset(h5_path=self.h5_path, compound_dim=self.compound_dim)

        assert os.path.exists(self.test_idx_path), "❌ test_indices.npy not found!"
        test_idx = np.load(self.test_idx_path)
        print(f"📊 Loaded test_idx: {len(test_idx)} samples")

        self.test_subset = Subset(full_dataset, test_idx)
        self.test_loader = DataLoader(
            self.test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        print("✅ Test DataLoader ready.")

    def _init_model(self):
        self.gene_encoder = GeneMHEncoderWithM(
            gene_dim=self.gene_dim,
            pathway_dim=self.pathway_dim,
            d_model=self.d_model,
            use_weight_transform=True,
            weight_transform_side="gene",
            trainable_pathway=False,
            init_pathway_matrix=self.mask_matrix,
        )
        self.compound_encoder = CompoundMLPEncoder(input_dim=self.compound_dim, d_model=self.d_model)
        self.cross_attn = SingleTokenCrossAttention(d_model=self.d_model)
        self.vae = CrossModalGeneBetaVAE(
            gene_dim=self.gene_dim,
            pathway_dim=self.pathway_dim,
            d_model=self.d_model,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
            beta=self.beta,
            aux_path_recon=self.aux_path_recon,
        )

        self.gene_encoder.to(self.device)
        self.compound_encoder.to(self.device)
        self.cross_attn.to(self.device)
        self.vae.to(self.device)

    def _remove_module_prefix(self, state_dict):
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state[k[len("module.") :]] = v
            else:
                new_state[k] = v
        return new_state

    def _load_checkpoint(self):
        assert os.path.exists(self.checkpoint_path), f"❌ Checkpoint not found: {self.checkpoint_path}"
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        print(f"🔄 Loading checkpoint weights (epoch={checkpoint.get('epoch', 'N/A')})")

        self.gene_encoder.load_state_dict(self._remove_module_prefix(checkpoint["gene_encoder"]))
        self.compound_encoder.load_state_dict(self._remove_module_prefix(checkpoint["compound_encoder"]))
        self.cross_attn.load_state_dict(self._remove_module_prefix(checkpoint["cross_attn"]))
        self.vae.load_state_dict(self._remove_module_prefix(checkpoint["vae"]))
        print("✅ Model weights loaded successfully.")

    # -------------------------------------------------
    def compute_metrics(self, y_true, y_pred):
        """
        计算 MSE, 平均 R², 平均 Pearson r。
        """
        # ---- MSE ----
        mse = np.mean((y_true - y_pred) ** 2)

        # ---- R² ----
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2_per_gene = 1 - ss_res / (ss_tot + 1e-8)
        r2_mean = np.mean(r2_per_gene)

        # ---- Pearson r ----
        pearson_per_gene = []
        for i in range(y_true.shape[1]):
            if np.std(y_true[:, i]) < 1e-8 or np.std(y_pred[:, i]) < 1e-8:
                pearson_per_gene.append(0.0)
                continue
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            pearson_per_gene.append(r)
        pearson_mean = float(np.mean(pearson_per_gene))

        return mse, r2_mean, pearson_mean

    # -------------------------------------------------
    def predict_and_save(self, output_dir="./predict_output"):
        """
        对测试集执行推理，并计算 MSE、R²、Pearson r。
        """
        os.makedirs(output_dir, exist_ok=True)
        preds, targets = [], []

        print("🚀 Running inference on test set...")

        with torch.no_grad():
            for batch in self.test_loader:
                x_gene_ctl = batch["gene_ctl"].to(self.device)
                x_gene_trt = batch["gene_trt"].to(self.device)
                x_comp = batch["compound"].to(self.device)

                # 前向传递
                g_feat = self.gene_encoder(x_gene_ctl)
                c_feat = self.compound_encoder(x_comp)
                g_out, c_out = self.cross_attn(g_feat, c_feat)
                fused = torch.cat([g_out, c_out], dim=-1)
                h = self.vae.fuse(fused)
                mu = self.vae.to_mu(h)
                logvar = self.vae.to_logvar(h)
                z = self.vae.reparameterize(mu, logvar)
                recon_gene = self.vae.dec_gene(z)

                preds.append(recon_gene.cpu())
                targets.append(x_gene_trt.cpu())

        pred_gene = torch.cat(preds, dim=0).numpy()
        true_gene = torch.cat(targets, dim=0).numpy()

        # === 保存预测结果 ===
        np.save(f"{output_dir}/pred_gene_expression.npy", pred_gene)
        pd.DataFrame(pred_gene).to_csv(f"{output_dir}/pred_gene_expression.csv", index=False)

        print(f"💾 Saved predictions: {output_dir}/pred_gene_expression.npy & .csv")

        # === 计算指标 ===
        mse, r2, pcc = self.compute_metrics(true_gene, pred_gene)
        print(f"📈 Test results → MSE: {mse:.6f} | R²: {r2:.4f} | Pearson r: {pcc:.4f}")

        # 写出结果文件
        with open(f"{output_dir}/test_metrics.txt", "w") as f:
            f.write("=== Test Metrics ===\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"R²:  {r2:.6f}\n")
            f.write(f"Pearson r: {pcc:.6f}\n")

        print(f"✅ Metrics saved to: {output_dir}/test_metrics.txt")
        print(f"🎯 Prediction complete! Total samples: {pred_gene.shape[0]}")
