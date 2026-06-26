# -*- coding: utf-8 -*-
# @Author: treehole
# @Date:   2026-4-3
# @Description: 根据可解释性结果 计算rges，计算代码改编自那篇nc
import numpy as np
import pandas as pd
def rank_ties_random(a, seed=None):
    """
    等价于 R: rank(a, ties.method="random")
    返回 1..n 的唯一秩；相同值的元素在这些秩里被随机打散。
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a)
    n = len(a)

    order = np.argsort(a, kind="mergesort")  # 稳定排序
    sorted_a = a[order]

    ranks = np.empty(n, dtype=float)
    cur_rank = 1
    i = 0
    while i < n:
        j = i
        while j < n and sorted_a[j] == sorted_a[i]:
            j += 1

        block = order[i:j]
        block_ranks = np.arange(cur_rank, cur_rank + (j - i), dtype=float)
        rng.shuffle(block_ranks)
        ranks[block] = block_ranks

        cur_rank += (j - i)
        i = j

    return ranks
def cmap_score_new(sig_up, sig_down, drug_signature):
    # drug_signature: DataFrame with columns ["ids", "rank"]
    # sig_up/sig_down: DataFrame where the 1st column contains gene ids

    # copy to avoid modifying caller
    drug_signature = drug_signature.copy()

    num_genes = drug_signature.shape[0]

    # R: drug_signature[,"rank"] <- rank(drug_signature[,"rank"])
    # pandas rank(method='average') approximates R's default (ties=average)
    drug_signature["rank"] = drug_signature["rank"].rank(method="average", ascending=True)

    # ---- merge: by.x="ids", by.y=1 ----
    # Extract ids from first column (R by.y = 1)
    if isinstance(sig_up, pd.DataFrame):
        sig_up_ids = sig_up.iloc[:, 0]
    else:
        sig_up_ids = pd.Series(sig_up)

    if isinstance(sig_down, pd.DataFrame):
        sig_down_ids = sig_down.iloc[:, 0]
    else:
        sig_down_ids = pd.Series(sig_down)

    # Make a 2-col DF for merging
    sig_up_df = pd.DataFrame({"ids": sig_up_ids})
    sig_down_df = pd.DataFrame({"ids": sig_down_ids})

    up_tags_rank = pd.merge(drug_signature[["ids", "rank"]], sig_up_df, on="ids", how="inner")
    down_tags_rank = pd.merge(drug_signature[["ids", "rank"]], sig_down_df, on="ids", how="inner")

    up_tags_position = np.sort(up_tags_rank["rank"].to_numpy())
    down_tags_position = np.sort(down_tags_rank["rank"].to_numpy())

    num_tags_up = len(up_tags_position)
    num_tags_down = len(down_tags_position)

    ks_up = 0.0
    ks_down = 0.0

    # ---- compute ks_up ----
    if num_tags_up > 1:
        # R:
        # a_up = max_{j=1..num_tags_up} ( j/num_tags_up - up_tags_position[j]/num_genes )
        # b_up = max_{j=1..num_tags_up} ( up_tags_position[j]/num_genes - (j-1)/num_tags_up )
        j = np.arange(1, num_tags_up + 1, dtype=float)  # 1..num_tags_up
        up_pos = up_tags_position.astype(float)

        a_up = np.max(j / num_tags_up - up_pos / num_genes)
        b_up = np.max(up_pos / num_genes - (j - 1) / num_tags_up)

        ks_up = a_up if a_up > b_up else -b_up
    else:
        ks_up = 0.0

    # ---- compute ks_down ----
    if num_tags_down > 1:
        j = np.arange(1, num_tags_down + 1, dtype=float)
        down_pos = down_tags_position.astype(float)

        a_down = np.max(j / num_tags_down - down_pos / num_genes)
        b_down = np.max(down_pos / num_genes - (j - 1) / num_tags_down)

        ks_down = a_down if a_down > b_down else -b_down
    else:
        ks_down = 0.0

    # ---- connectivity_score logic (keep identical structure) ----
    if (ks_up == 0.0) and (ks_down != 0.0):      # only down gene inputed
        connectivity_score = -ks_down
    elif (ks_up != 0.0) and (ks_down == 0.0):  # only up gene inputed
        connectivity_score = ks_up
    elif np.sum(np.sign([ks_down, ks_up])) == 0:
        # different signs
        connectivity_score = ks_up - ks_down
    else:
        connectivity_score = ks_up - ks_down

    return float(connectivity_score)

def build_X_from_idx(dfM, idx, ic50_list, idx_mode="auto"):
    idx = list(idx)
    ic50_list = list(ic50_list)
    if len(idx) != len(ic50_list):
        raise ValueError(f"len(idx)={len(idx)} != len(ic50_list)={len(ic50_list)}")

    valid_rows = []
    valid_ic50 = []

    for i, (idv, ic50v) in enumerate(zip(idx, ic50_list)):
        if idv is None or (isinstance(idv, float) and np.isnan(idv)):
            continue
        if ic50v is None or (isinstance(ic50v, float) and np.isnan(ic50v)):
            continue
        valid_rows.append(idv)
        valid_ic50.append(ic50v)

    if idx_mode == "auto":
        try:
            df_index = dfM.index
            can_loc = all(v in df_index for v in valid_rows)
            use_loc = bool(can_loc)
        except Exception:
            use_loc = False
    elif idx_mode == "loc":
        use_loc = True
    elif idx_mode == "iloc":
        use_loc = False
    else:
        raise ValueError("idx_mode must be one of: 'auto', 'loc', 'iloc'")

    if use_loc:
        X = dfM.loc[valid_rows].values
    else:
        X = dfM.iloc[valid_rows].values

    ic50_array = np.asarray(valid_ic50, dtype=float)
    
    return X, ic50_array
