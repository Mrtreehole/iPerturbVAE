# -*- coding: utf-8 -*-
# @Author: Dehua Liu (Refactored by Assistant)
# @Date: 2025-10-23

import torch
import h5py
from torch.utils.data import Dataset
import pandas as pd


gene_id_sorted = pd.read_csv('/home/liuxiaoping/cbfp_vae/dataprocess/data/landmark_gene_idx_positions.csv')['idx_position'].tolist()

class H5GeneCompoundDataset(Dataset):
    """
    从 HDF5 文件读取 gene 和 compound 数据。
    支持懒加载 + DDP 分布式训练。
    """
    def __init__(self, h5_path, compound_dim=3200):
        super().__init__()
        self.h5_path = h5_path
        self.compound_dim = compound_dim
        self.file = None  # 懒打开文件

        # 只读取一次长度
        with h5py.File(h5_path, "r") as f:
            self.length = f["comp_ccfps"].shape[0]
        print(f"📂 Loaded HDF5 dataset: {self.length} samples")

    def _get_file(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r", swmr=True)
        return self.file

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()
        comp = f["comp_ccfps"][idx].astype("float32")
        #now we need to get landmark genes only and sort by geneinfo_beta
        gene_ctlfull = f["gene_info_ctl"][idx].astype("float32")
        gene_trtfull = f["gene_info_trt"][idx].astype("float32")
        #now we need to get landmark genes only and sort by geneinfo_beta
        gene_ctl = gene_ctlfull[gene_id_sorted]
        gene_trt = gene_trtfull[gene_id_sorted]

        comp = torch.from_numpy(comp)
        return {
            "compound": comp,      # (3200,)
            "gene_ctl": torch.from_numpy(gene_ctl),
            "gene_trt": torch.from_numpy(gene_trt),
        }
