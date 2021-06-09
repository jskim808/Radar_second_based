
import numpy as np
import copy
import pickle
import random
import pdb

nusc_cls = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle',
            'bicycle', 'traffic_cone', 'barrier'}

def dbg_instance(src_nusc, dst_nusc):
    src_cnt, dst_cnt = {}, {}
    src_sum, dst_sum = 0, 0
    for nusc_cl in nusc_cls:
        src_cnt[nusc_cl] = 0
        dst_cnt[nusc_cl] = 0
    for src_info in src_nusc:
        for cls in src_info['gt_names']:
            if cls in nusc_cls:
                src_cnt[cls] += 1
                src_sum += 1
    for dst_info in dst_nusc:
        for cls in dst_info['gt_names']:
            if cls in nusc_cls:
                dst_cnt[cls] += 1
                dst_sum += 1
    for nusc_cl in nusc_cls:
        print('%20s %6d %6d %4f %4f' % \
                (nusc_cl, src_cnt[nusc_cl], dst_cnt[nusc_cl], 
                 src_cnt[nusc_cl]/src_sum, dst_cnt[nusc_cl]/dst_sum))

def sampling_sample(cls_list, cls_name, visit, num_samp):
    samp_idx = np.array([], dtype=np.int)
    cls_visit = visit[cls_list]
    not_use_cls = (cls_visit == 0).nonzero()[0]
    if len(not_use_cls) <= num_samp:
        num_samp -= len(not_use_cls)
        visit[cls_list[not_use_cls]] = True
        samp_idx = np.hstack((samp_idx, cls_list[not_use_cls]))
        not_use_cls = np.array(range(len(cls_list)))
    if len(not_use_cls) > num_samp:
        random.shuffle(not_use_cls)
        rand_samp = not_use_cls[:num_samp]
        visit[cls_list[rand_samp]] = True
        samp_idx = np.hstack((samp_idx, cls_list[rand_samp]))
    else:
        pdb.set_trace()
        abcd = 1
    return samp_idx
   
def ds_sampling(_nusc_infos):
    nusc_infos = copy.deepcopy(_nusc_infos)
    NUM_INPUT = len(nusc_infos)
    ds_dict = {}
    num_stage2_total, visit_cnt = 0, 0
    for nusc_cl in nusc_cls:
        ds_dict[nusc_cl] = []

    # duplicate sample
    for i in range(NUM_INPUT):
        uniq_gts = np.unique(nusc_infos[i]['gt_names'])
        uniq_masks = np.zeros_like(uniq_gts, dtype=bool)
        # masking necessary cls
        for u_idx, uniq_gt in enumerate(uniq_gts):
            uniq_masks[u_idx] = uniq_gt in nusc_cls
        uniq_gts = uniq_gts[uniq_masks]
        num_stage2_total += len(uniq_gts)
        # duplicate current idxs sample
        append_list = [i]
        for l in range(len(uniq_gts) - 1):
            nusc_infos.append(nusc_infos[i].copy())
            append_list.append(len(nusc_infos)-1)
        for uniq_gt in uniq_gts:
            for ap in append_list:
                ds_dict[uniq_gt].append(ap)

    # sampling samples
    ds_list = []
    num_samp = num_stage2_total // len(ds_dict.keys())
    visit = np.zeros(len(nusc_infos))
    for key in ds_dict.keys():
        ds_dict[key] = np.array(ds_dict[key])
        idx = sampling_sample(ds_dict[key], key, visit, num_samp)
        ds_list = ds_list + idx.tolist()
    ds_list = np.array(ds_list)

    # makes new nusc_infos
    ds_nusc_infos = []
    for ds_idx in ds_list:
        ds_nusc_infos.append(nusc_infos[ds_idx])

    # for compare instance number
    dbg_instance(_nusc_infos, ds_nusc_infos) 

    return ds_nusc_infos

if __name__ == '__main__':
    with open('train_nusc_infos.pkl', 'rb') as f:
        train_nusc_infos = pickle.load(f)
    with open('val_nusc_infos.pkl', 'rb') as f:
        val_nusc_infos = pickle.load(f)
    ds_sampling(train_nusc_infos)
