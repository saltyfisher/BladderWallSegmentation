import numpy as np
import cv2 as cv
from tqdm import tqdm
from scipy.spatial import distance


def compute_score(img, u_idx, k_idx):
    color_diff = img[u_idx[0], u_idx[1]] - img[k_idx[0], k_idx[1]]
    dst_diff = np.linalg.norm(np.array([u_idx[0], u_idx[1]]).T.reshape(-1, 1) - np.stack(k_idx), axis=0)
    dst_diff = (dst_diff - np.min(dst_diff)) / (np.max(dst_diff) - np.min(dst_diff) + 1e-6)
    score = color_diff + 0.5 * dst_diff

    return score

def compute_process_simmilarity(process):
    process_idx = np.argsort(process)
    best_idx = np.arange(process.size)

    simmilarity = distance.hamming(process_idx, best_idx)
    return simmilarity

def find_most_sim(flag, ngh_idx, process_idx, sample_idx, img, u_idx, k_idx, threshold_sim):
    best_sim_idx = -1
    best_sim = np.inf
    if ngh_idx.size == 0:
        return best_sim_idx, best_sim

    for j in ngh_idx:
        if flag[j] == 0:
            continue
        subidx = process_idx[j, sample_idx]
        score = compute_score(img, u_idx, [k_idx[0][subidx], k_idx[1][subidx]])
        simmilarity = compute_process_simmilarity(score)
        if simmilarity <= threshold_sim:
            if simmilarity < best_sim:
                best_sim_idx = j
                best_sim = simmilarity
    return best_sim_idx, best_sim

def generate_trimap_hierarchy_local(img, trimap):
    trimap = np.pad(trimap, (1, 1), mode='constant', constant_values=0)
    img = np.pad(img, (1, 1), mode='constant', constant_values=0)
    n_num = np.nonzero(trimap == 128)[0].size
    with tqdm(total=n_num) as pbar:
        while np.nonzero(trimap == 128)[0].size != 0:
            dst = cv.distanceTransform((trimap != 255).astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_3)
            dst = np.round(dst, 2) * (trimap == 128)
            min_dst = np.min(dst[dst != 0])
            u_idx = np.nonzero(dst == min_dst)
            for i in range(u_idx[0].shape[0]):
                ngh_idx = np.meshgrid(np.arange(u_idx[0][i] - 1, u_idx[0][i] + 2),
                                      np.arange(u_idx[1][i] - 1, u_idx[1][i] + 2))
                ngh_idx = np.stack([np.delete(x.reshape(-1), 4) for x in ngh_idx])
                fg_idx = ngh_idx[:, trimap[ngh_idx[0, :], ngh_idx[1, :]] == 255]
                bg_idx = np.nonzero(trimap == 0)

                if fg_idx.size == 0:
                    trimap[u_idx[0][i], u_idx[1][i]] = 0
                    continue

                fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)

                bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)

                if np.min(fu_score) <= np.min(bu_score):
                    trimap[u_idx[0][i], u_idx[1][i]] = 255
                else:
                   trimap[u_idx[0][i],  u_idx[1][i]] = 0
                pbar.update()
    trimap = trimap[1:-1, 1:-1]
    return trimap

def generate_trimap_hierarchy(img, trimap):
    n_num = np.nonzero(trimap == 128)[0].size
    with tqdm(total=n_num) as pbar:
        while np.nonzero(trimap == 128)[0].size != 0:
            # print(np.nonzero(trimap == 128)[0].size)
            dst = cv.distanceTransform((trimap != 255).astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_3)
            dst = np.round(dst, 2) * (trimap == 128)
            min_dst = np.min(dst[dst != 0])
            u_idx = np.nonzero(dst == min_dst)

            for i in range(u_idx[0].shape[0]):
                fg_idx = np.nonzero(trimap == 255)
                bg_idx = np.nonzero(trimap == 0)

                fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)

                bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)

                if np.min(fu_score) <= np.min(bu_score):
                    trimap[u_idx[0][i], u_idx[1][i]] = 255
                else:
                    trimap[u_idx[0][i], u_idx[1][i]] = 0
                pbar.update()
    return trimap

def generate_trimap_hierarchy_DP(img, trimap):
    sample_num = 50
    threshold_sim = 0.5
    n_num = np.nonzero(trimap == 128)[0].size
    with tqdm(total=n_num) as pbar:
        while np.nonzero(trimap == 128)[0].size != 0:
            print(np.nonzero(trimap == 128)[0].size)
            dst = cv.distanceTransform((trimap != 255).astype(np.uint8), distanceType=cv.DIST_L2, maskSize=cv.DIST_MASK_3)
            dst = np.round(dst, 2) * (trimap == 128)
            min_dst = np.min(dst[dst != 0])
            u_idx = np.nonzero(dst == min_dst)
            u_mask = np.zeros_like(trimap)
            u_mask[u_idx[0], u_idx[1]] = 128
            u_num = u_idx[0].size
            fg_idx = np.nonzero(trimap == 255)
            bg_idx = np.nonzero(trimap == 0)
            f_num = fg_idx[0].size
            b_num = bg_idx[0].size
            uf_process_idx = np.zeros((u_num, f_num), dtype=np.int)
            # uf_simmilarity = np.ones(u_num, u_num) * -1
            ub_process_idx = np.zeros((u_num, b_num), dtype=np.int)
            # ub_simmilarity = np.ones(u_num, u_num) * -1
            f_sample_idx = np.linspace(0, f_num-1, sample_num, dtype=np.int)
            b_sample_idx = np.linspace(0, b_num-1, sample_num, dtype=np.int)
            uf_flag = np.zeros(u_num)
            ub_flag = np.zeros(u_num)
            for i in range(u_idx[0].shape[0]):
                ngh_idx = np.meshgrid(np.arange(u_idx[0][i] - 1, u_idx[0][i] + 2),
                                          np.arange(u_idx[1][i] - 1, u_idx[1][i] + 2))
                ngh_idx = np.stack([np.delete(x.reshape(-1), 4) for x in ngh_idx])
                ngh_idx = ngh_idx[:, u_mask[ngh_idx[0, :], ngh_idx[1, :]] == 128]
                idx = np.argwhere((u_idx[0] == ngh_idx[0, :]) & (u_idx[1] == ngh_idx[1, :])).reshape(-1)
                best_sim_idx_f, best_sim_f = \
                    find_most_sim(uf_flag, idx, uf_process_idx, f_sample_idx, img, [u_idx[0][i], u_idx[1][i]], fg_idx, threshold_sim)
                if best_sim_idx_f != -1:
                    fg_subidx = uf_process_idx[best_sim_idx_f, :f_sample_idx[1]]
                    fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]],
                                             [fg_idx[0][fg_subidx], fg_idx[1][fg_subidx]])
                    uf_process_idx[i, :] = uf_process_idx[best_sim_idx_f, :]
                else:
                    fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)
                    uf_process_idx[i, :] = np.argsort(fu_score)
                    uf_flag[i] = 1
                best_sim_idx_b, best_sim_b = \
                    find_most_sim(ub_flag, idx, ub_process_idx, b_sample_idx, img, [u_idx[0][i], u_idx[1][i]], bg_idx, threshold_sim)
                if best_sim_idx_f != -1:
                    bg_subidx = ub_process_idx[best_sim_idx_b, :f_sample_idx[1]]
                    bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]],
                                             [bg_idx[0][bg_subidx], bg_idx[1][bg_subidx]])
                    ub_process_idx[i, :] = ub_process_idx[best_sim_idx_b]
                else:
                    bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)
                    ub_process_idx[i, :] = np.argsort(bu_score)
                    ub_flag[i] = 1

                if np.min(fu_score) <= np.min(bu_score):
                    trimap[u_idx[0][i], u_idx[1][i]] = 255
                else:
                    trimap[u_idx[0][i], u_idx[1][i]] = 0
                pbar.update()
    return trimap

def generate_trimap_DP(img, trimap):
    sample_num = 50
    threshold_sim = 0.5

    trimap = np.pad(trimap, (1, 1), mode='constant', constant_values=0)
    img = np.pad(img, (1, 1), mode='constant', constant_values=0)
    u_idx = np.nonzero(trimap == 128)

    u_num = u_idx[0].size
    fg_idx = np.nonzero(trimap == 255)
    bg_idx = np.nonzero(trimap == 0)
    f_num = fg_idx[0].size
    b_num = bg_idx[0].size
    uf_process_idx = np.zeros((u_num, f_num), dtype=np.int)
    # uf_simmilarity = np.ones(u_num, u_num) * -1
    ub_process_idx = np.zeros((u_num, b_num), dtype=np.int)
    # ub_simmilarity = np.ones(u_num, u_num) * -1
    f_sample_idx = np.linspace(0, f_num-1, sample_num, dtype=np.int)
    b_sample_idx = np.linspace(0, b_num-1, sample_num, dtype=np.int)
    uf_flag = np.zeros(u_num)
    ub_flag = np.zeros(u_num)
    with tqdm(total=u_idx[0].shape[0]) as pbar:
        for i in range(u_idx[0].shape[0]):
            ngh_idx = np.meshgrid(np.arange(u_idx[0][i] - 1, u_idx[0][i] + 2),
                                      np.arange(u_idx[1][i] - 1, u_idx[1][i] + 2))
            ngh_idx = np.stack([np.delete(x.reshape(-1), 4) for x in ngh_idx])
            ngh_idx = ngh_idx[:, trimap[ngh_idx[0, :], ngh_idx[1, :]] == 128]
            idx = np.argwhere((u_idx[0] == ngh_idx[0, :]) & (u_idx[1] == ngh_idx[1, :])).reshape(-1)
            best_sim_idx_f, best_sim_f = \
                find_most_sim(uf_flag, idx, uf_process_idx, f_sample_idx, img, [u_idx[0][i], u_idx[1][i]], fg_idx, threshold_sim)
            if best_sim_idx_f != -1:
                fg_subidx = uf_process_idx[best_sim_idx_f, :f_sample_idx[1]]
                fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]],
                                         [fg_idx[0][fg_subidx], fg_idx[1][fg_subidx]])
                uf_process_idx[i, :] = uf_process_idx[best_sim_idx_f, :]
            else:
                fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)
                uf_process_idx[i, :] = np.argsort(fu_score)
                uf_flag[i] = 1
            best_sim_idx_b, best_sim_b = \
                find_most_sim(ub_flag, idx, ub_process_idx, b_sample_idx, img, [u_idx[0][i], u_idx[1][i]], bg_idx, threshold_sim)
            if best_sim_idx_f != -1:
                bg_subidx = ub_process_idx[best_sim_idx_b, :f_sample_idx[1]]
                bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]],
                                         [bg_idx[0][bg_subidx], bg_idx[1][bg_subidx]])
                ub_process_idx[i, :] = ub_process_idx[best_sim_idx_b]
            else:
                bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)
                ub_process_idx[i, :] = np.argsort(bu_score)
                ub_flag[i] = 1

            if np.min(fu_score) <= np.min(bu_score):
                trimap[u_idx[0][i], u_idx[1][i]] = 255
            else:
                trimap[u_idx[0][i], u_idx[1][i]] = 0
            pbar.update()

    trimap = trimap[1:-1, 1:-1]
    return trimap

def generate_trimap_local(img, trimap):
    window_size = 10
    trimap = np.pad(trimap, (window_size, window_size), mode='constant', constant_values=0)
    img = np.pad(img, (window_size, window_size), mode='constant', constant_values=0)
    u_idx = np.nonzero(trimap == 128)
    with tqdm(total=u_idx[0].size) as pbar:
        for i in range(u_idx[0].shape[0]):
            ngh_idx = np.meshgrid(np.arange(u_idx[0][i] - window_size, u_idx[0][i] + window_size + 1),
                                  np.arange(u_idx[1][i] - window_size, u_idx[1][i] + window_size + 1))
            ngh_idx = np.stack([x.reshape(-1) for x in ngh_idx])
            fg_idx = ngh_idx[:, trimap[ngh_idx[0, :], ngh_idx[1, :]] == 255]
            bg_idx = ngh_idx[:, trimap[ngh_idx[0, :], ngh_idx[1, :]] == 0]

            if fg_idx.size == 0:
                trimap[u_idx[0][i], u_idx[1][i]] = 0
                continue
            if bg_idx.size == 0:
                trimap[u_idx[0][i], u_idx[1][i]] = 255
                continue

            fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)

            bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)

            if np.min(fu_score) <= np.min(bu_score):
                trimap[u_idx[0][i], u_idx[1][i]] = 255
            else:
               trimap[u_idx[0][i],  u_idx[1][i]] = 0
            pbar.update()
    trimap = trimap[window_size:-window_size, window_size:-window_size]
    return trimap

def generate_trimap(img, trimap):
    u_idx = np.nonzero(trimap == 128)
    fg_idx = np.nonzero(trimap == 255)
    bg_idx = np.nonzero(trimap == 0)
    with tqdm(total=u_idx[0].size) as pbar:
        for i in range(u_idx[0].shape[0]):

            fu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], fg_idx)

            bu_score = compute_score(img, [u_idx[0][i], u_idx[1][i]], bg_idx)

            if np.min(fu_score) <= np.min(bu_score):
                trimap[u_idx[0][i], u_idx[1][i]] = 255
            else:
                trimap[u_idx[0][i], u_idx[1][i]] = 0
            pbar.update()
    return trimap

