import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

def compute_r2_pearson(y_true, y_pred, var_eps=1e-8, min_ss_tot=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # per-gene R2
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    y_mean = np.mean(y_true, axis=0)
    ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)

    valid = ss_tot > min_ss_tot
    r2_per_gene = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    r2_per_gene[valid] = 1.0 - ss_res[valid] / (ss_tot[valid] + var_eps)
    mean_r2 = float(np.nanmean(r2_per_gene))

    # per-gene Pearson
    pearson_vals = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    for i in np.where(valid)[0]:
        if np.std(y_pred[:, i]) < 1e-8:  # 预测为常数：相关性没意义
            continue
        pearson_vals[i] = pearsonr(y_true[:, i], y_pred[:, i])[0]
    mean_pearson = float(np.nanmean(pearson_vals))

    return mean_r2, mean_pearson

def compute_r2_pearson_samplewise(y_true, y_pred, var_eps=1e-8, min_ss_tot=1e-6):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_samples = y_true.shape[0]

    r2_vals = np.full(n_samples, np.nan)
    pearson_vals = np.full(n_samples, np.nan)

    for i in range(n_samples):

        yt = y_true[i]
        yp = y_pred[i]

        # R2
        ss_res = np.sum((yt - yp) ** 2)
        yt_mean = np.mean(yt)
        ss_tot = np.sum((yt - yt_mean) ** 2)

        if ss_tot > min_ss_tot:
            r2_vals[i] = 1.0 - ss_res / (ss_tot + var_eps)

        # Pearson
        if np.std(yt) > 1e-8 and np.std(yp) > 1e-8:
            pearson_vals[i] = pearsonr(yt, yp)[0]

    mean_r2 = float(np.nanmean(r2_vals))
    mean_pearson = float(np.nanmean(pearson_vals))

    return mean_r2, mean_pearson