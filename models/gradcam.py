# vae_attention_cross.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZToConvFeature(nn.Module):
    """
    仅用于解释：把 latent 向量 z 映射成一组 1D conv 特征图 A。
    不参与原模型训练。
    """
    def __init__(
        self,
        z_dim: int,
        conv_channels: int = 64,
        conv_len: int = 16,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 建议用奇数，方便 same padding")

        self.z_dim = z_dim
        self.conv_channels = conv_channels
        self.conv_len = conv_len

        # 先从 z 得到一个长度为 conv_len 的序列
        self.expand = nn.Linear(z_dim, conv_channels * conv_len)
        # 在序列上做一层 conv1d，得到最终的 A
        pad = kernel_size // 2
        self.conv = nn.Conv1d(conv_channels, conv_channels,
                              kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()

    def forward(self, z: torch.Tensor):
        """
        z: [B, D] -> A: [B, C, L]
        """
        B, D = z.shape
        x = self.expand(z)                 # [B, C*L]
        x = x.view(B, self.conv_channels, self.conv_len)  # [B, C, L]
        A = self.conv(x)                  # [B, C, L]
        A = self.act(A)
        return A


class CrossVAEAttentionGenerator:
    """
    用条件 VAE 的 z_cond，在一个“解释用 conv 模块”上实现
    Liu 论文 3.2 节的梯度 attention[^12]。

    注意：
    - 这里不走 GeneOnlyAttnVAE_Conv 的 encoder/decoder，只是用 z_cond 本身。
    - 如果你一定要把 A 放在 GeneOnlyAttnVAE_Conv 里，我可以再帮你替换成真正的 conv 层 hook，
      但目前你的 gene-only VAE 的 decoder 是 MLP，没有 conv，所以先用这个“解释分支”比较自然。
    """

    def __init__(self, z_dim: int,
                 conv_channels: int = 64,
                 conv_len: int = 16,
                 kernel_size: int = 3,
                 device: torch.device = torch.device("cpu")):
        self.device = device
        self.z_dim = z_dim

        # z -> A 的解释用 conv 模块
        self.z_to_feat = ZToConvFeature(
            z_dim=z_dim,
            conv_channels=conv_channels,
            conv_len=conv_len,
            kernel_size=kernel_size,
        ).to(device)

        # 用于保存前向的特征 A 和反向的梯度 dA
        self._features = None
        self._gradients = None

        # 在 conv 层上注册 hooks
        self._fwd_hook = self.z_to_feat.conv.register_forward_hook(
            self._forward_hook_fn
        )
        self._bwd_hook = self.z_to_feat.conv.register_full_backward_hook(
            self._backward_hook_fn
        )

    def _forward_hook_fn(self, module, inputs, output):
        # output = A: [B, C, L]
        self._features = output

    def _backward_hook_fn(self, module, grad_input, grad_output):
        # grad_output[0] = d(score)/dA: [B, C, L]
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def _compute_Mi_from_current_grad(self):
        """
        按 Liu 论文式(2)(3)从当前 batch 的 A 和 ∂z_i/∂A 计算 M_i[^12]。

        这里 A: [B, C, L] 当成 “一维空间”，等价于 h×w = 1×L 的情形。
        返回：Mi: [B, 1, L]
        """
        A = self._features          # [B, C, L]
        dA = self._gradients        # [B, C, L]
        assert A is not None and dA is not None, "features/gradients 未捕获到"

        B, C, L = A.shape

        # α_k = 1/T ∑_p ∑_q ∂z_i/∂A_k(p,q)[^12]
        # 一维情况下就是对 L 做平均
        alpha = dA.view(B, C, -1).mean(dim=-1)    # [B, C]
        alpha = alpha.view(B, C, 1)               # [B, C, 1]

        # M_i = ReLU(∑_k α_k A_k)[^12]
        weighted = (alpha * A).sum(dim=1, keepdim=True)  # [B, 1, L]
        Mi = F.relu(weighted)
        return Mi

    @torch.no_grad()
    def _normalize_map(self, M, eps: float = 1e-8):
        """
        把注意力图做 [0,1] 归一化，方便可视化。
        输入: [B, 1, L]
        """
        B = M.size(0)
        M_flat = M.view(B, -1)
        minv = M_flat.min(dim=1, keepdim=True)[0]
        maxv = M_flat.max(dim=1, keepdim=True)[0]
        M_norm = (M_flat - minv) / (maxv - minv + eps)
        return M_norm.view_as(M)

    def get_attention_from_cond_z(self, z_cond: torch.Tensor):
        """
        输入：
          z_cond: 条件 VAE 输出的 latent，形状 [B, D]

        输出：
          Mi_all: [D, B, 1, L]，每个 latent 维的注意力曲线 M_i[^12]
          M_overall: [B, 1, L]，整体 attention M = 平均 M_i[^12]
        """
        device = self.device
        z = z_cond.detach().to(device).clone()
        z.requires_grad_(True)

        B, D = z.shape
        assert D == self.z_dim, f"z_dim mismatch: got {D}, expected {self.z_dim}"

        Mi_list = []

        # 按 Liu：对每个 z_i 单独取 score = ∑_b z[b,i]，反传到 A[^12]
        for dim_idx in range(D):
            # 清空梯度与缓存
            if z.grad is not None:
                z.grad.zero_()
            self._features = None
            self._gradients = None

            # forward: z -> A (conv feature)
            _ = self.z_to_feat(z)      # A 会被 forward hook 捕获

            # score: 对 batch 求和，使之成为标量
            score = z[:, dim_idx].sum()
            score.backward(retain_graph=True)

            Mi = self._compute_Mi_from_current_grad()  # [B, 1, L]
            Mi_list.append(Mi.detach().clone())

        # [D, B, 1, L]
        Mi_all = torch.stack(Mi_list, dim=0)

        # overall: M = 1/D ∑ M_i[^12]
        M_overall = Mi_all.mean(dim=0)  # [B, 1, L]

        # 归一化到 [0,1]
        M_overall = self._normalize_map(M_overall)
        Mi_all_flat = Mi_all.view(-1, 1, Mi_all.size(-1))
        Mi_all_norm = self._normalize_map(Mi_all_flat).view_as(Mi_all)

        return Mi_all_norm, M_overall
