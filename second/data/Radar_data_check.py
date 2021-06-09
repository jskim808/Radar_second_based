from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.nuscenes import NuScenes
import numpy as np
import os

root_path = '/mnt/sdd/jhyoo/dataset/NUSCENES/'
GRIF_path = '/mnt/sdd/jhyoo/dataset/NUSCENES/ns_radar_train/radar/'
version = 'v1.0-trainval'


nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

sample0 = nusc.sample[0]
sample1 = nusc.sample[1]
samples = [sample0, sample1]
radar_channels = ['RADAR_FRONT','RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']
max_sweeps = 10
use_idx = [0,1,2,5,6,7,8,9]


for i in range(3):
    import pdb; pdb.set_trace()
    sample = nusc.sample[i]
    all_pc = np.zeros((0, 18))
    sample_tk = sample['token']
    for radar_channel in radar_channels:
        radar_data_token = sample['data'][radar_channel]
        radar_sample_data = nusc.get('sample_data',radar_data_token)
        for _ in range(max_sweeps):
            radar_path = root_path+radar_sample_data['filename']
            current_pc = RadarPointCloud.from_file(radar_path).points.T
            all_pc = np.vstack((all_pc,current_pc))
            if radar_sample_data['next'] == '':
                break
            else:
                radar_sample_data = nusc.get('sample_data', radar_sample_data['next'])
    radar_pc_base = all_pc[:,use_idx]
    # print(f'Sample {i} sample_token = {sample_tk}, point_num = {point_num}, total={np.sum(point_num)}')
    base_sample_data = nusc.get('sample_data',sample['data']['RADAR_FRONT'])
    Jisong_path = base_sample_data['filename'].split('/')[-1].split('.')[0]
    my_points = np.load(f'{root_path}samples/RADAR_version2_6Sweeps/{Jisong_path}.npy')
    radar_bin = str(np.char.zfill(str(i),6))+'.bin'
    points = np.fromfile(GRIF_path+radar_bin, dtype=np.float32, count=-1).reshape([-1, 6])
    print(all_pc.shape)
    print(points.shape)
    # import pdb; pdb.set_trace()
    print(f'Sample {i} sample_token = {sample_tk}, GRIF_point_num = {len(points)}')
'''

point_info = {'point_num' = [], 'point_rcs' = [] ,'point_vx'= [] , 'point_vy'= []}
for i,sample in enumerate(nusc.sample):
    radar_bin = str(np.char.zfill(str(i),6))+'.bin'
    points = np.fromfile(GRIF_path+radar_bin, dtype=np.float32, count=-1).reshape([-1, 6])
    point_num = len(points)
    point_rcs = points[][][]
    import pdb; pdb.set_trace()
'''