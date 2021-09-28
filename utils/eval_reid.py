# encoding: utf-8
"""
Based on code from:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

Adapted and extended by:
@author: mikwieczorek
"""

import numpy as np

from tqdm import tqdm

k_list = [1, 5, 10, 20, 50]


def top_k_retrieval(row_matches: np.ndarray, k: list):
    results = []
    for kk in k:
        results.append(np.any(row_matches[:kk]))
    return [int(item) for item in results]


def eval_func(
    indices, q_pids, g_pids, q_camids, g_camids, max_rank=50, respect_camids=False
):
    """
    Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = indices.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    topk_results = []  # Store topk retureval
    single_performance = []
    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        if respect_camids:
            remove = [
                (gpid == q_pid) & (q_camid in gcamid)
                for gpid, gcamid in zip(g_pids[order], g_camids[order])
            ]
        else:
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.0) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        # Save AP for each query to allow finding worst performing samples
        single_performance.append(list([q_idx, q_pid, AP]))
        # Get topk accuracy for topk
        topk_results.append(top_k_retrieval(orig_cmc, k_list))

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    all_topk = np.vstack(topk_results)
    all_topk = np.mean(all_topk, 0)

    return all_cmc, mAP, all_topk, np.array(single_performance)
