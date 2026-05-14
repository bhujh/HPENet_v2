import os
import pickle
import logging
import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS


@DATASETS.register_module()
class RadarClassi(Dataset):
    classes = ['valid',
               'invalid']
    num_classes = 2
    class2color = {
        "valid": [0, 0, 255],
        "invalid": [255, 0, 0],
    }
    cmap = [*class2color.values()]
    gravity_dim = 2
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """
    def __init__(
        self,
        data_root: str = "data/RadarClassi/radarfull",
        test_area: int = 5,
        voxel_size: float = 0.04,
        voxel_max=None,
        split: str = "train",
        transform=None,
        loop: int = 1,
        presample: bool = False,
        variable: bool = False,
        shuffle: bool = True,
    ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle

        raw_root = os.path.join(data_root, 'raw')
        self.raw_root = raw_root
        data_list = sorted(os.listdir(raw_root))
        data_list = [item[:-4] for item in data_list]
        np.random.seed(100)  # 固定种子保证可复现
        np.random.shuffle(data_list)
        n = len(data_list)
        if split == "train":
            self.data_list = data_list[:int(n * 0.83)]
        else:
            self.data_list = data_list[int(n * 0.83):]

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f'radar_{split}_area{test_area}_{voxel_size:.3f}_{str(voxel_max)}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading RadarClassi {split} split on Test Area {test_area}'):
                data_path = os.path.join(raw_root, item + '.ply')
                # cdata = np.load(data_path).astype(np.float32)
                plydata = PlyData.read(data_path)
                # print(plydata["vertex"].data.dtype.names)
                cdata = plydata["vertex"].data.view(np.float32).reshape(-1, 7)
                # print(cdata.shape)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                if voxel_size:
                    coord, feat, label = cdata[:,0:3], cdata[:, 3:6], cdata[:, 6:7]
                    uniq_idx = voxelize(coord, voxel_size)
                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

        stats_file = os.path.join(processed_root, f'feat_stats_area{test_area}.pth')
        if split == 'train':
            counts = np.zeros(self.num_classes, dtype=np.int32)
            feat_sum = np.zeros(3, dtype=np.float64)
            feat_sq_sum = np.zeros(3, dtype=np.float64)
            z_sum = 0.0
            z_sq_sum = 0.0
            n_total = 0

            for item in tqdm(self.data_list, desc='Scanning training data'):
                data_path = os.path.join(self.raw_root, item + '.ply')
                plydata = PlyData.read(data_path)
                cdata = plydata["vertex"].data.view(np.float32).reshape(-1, 7)
                cdata[:, :3] -= np.min(cdata[:, :3], 0)
                labels = cdata[:, 6].astype(np.int32)
                for cls_idx in range(self.num_classes):
                    counts[cls_idx] += (labels == cls_idx).sum()
                feat = np.nan_to_num(cdata[:, 3:6], nan=0.0)
                z = np.nan_to_num(cdata[:, 2], nan=0.0)
                feat_sum += feat.sum(axis=0)
                feat_sq_sum += (feat ** 2).sum(axis=0)
                z_sum += z.sum()
                z_sq_sum += (z ** 2).sum()
                n_total += feat.shape[0]

            self.num_per_class = counts
            feat_mean = feat_sum / n_total
            feat_std = np.sqrt(np.maximum(feat_sq_sum / n_total - feat_mean ** 2, 1e-5))
            z_mean = z_sum / n_total
            z_std = np.sqrt(np.maximum(z_sq_sum / n_total - z_mean ** 2, 1e-5))

            self.feat_mean = torch.tensor(feat_mean, dtype=torch.float32)
            self.feat_std = torch.tensor(feat_std, dtype=torch.float32)
            self.z_mean = torch.tensor(z_mean, dtype=torch.float32)
            self.z_std = torch.tensor(z_std, dtype=torch.float32)

            os.makedirs(processed_root, exist_ok=True)
            torch.save({
                'feat_mean': self.feat_mean, 'feat_std': self.feat_std,
                'z_mean': self.z_mean, 'z_std': self.z_std
            }, stats_file)
            logging.info(f'Class counts: {counts}')
            logging.info(f'Feature mean: {feat_mean}, Feature std: {feat_std}')
            logging.info(f'Z mean: {z_mean:.2f}, Z std: {z_std:.2f}')
        else:
            if os.path.exists(stats_file):
                stats = torch.load(stats_file)
                self.feat_mean = stats['feat_mean']
                self.feat_std = stats['feat_std']
                self.z_mean = stats['z_mean']
                self.z_std = stats['z_std']

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        if self.presample:
            coord, feat, label = np.split(self.data[data_idx], [3, 6], axis=1)
            coord = np.nan_to_num(coord, nan=0.0)
            feat = np.nan_to_num(feat, nan=0.0)
            label = np.nan_to_num(label, nan=0.0)
        else:
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.ply')
            # cdata = np.load(data_path).astype(np.float32)
            plydata = PlyData.read(data_path)
            cdata = plydata["vertex"].data.view(np.float32).reshape(-1, 7)
            cdata[:, :3] -= np.min(cdata[:, :3], 0)
            cdata = np.nan_to_num(cdata, nan=0.0)
            coord, feat, label = cdata[:, :3], cdata[:, 3:6], cdata[:, 6:7]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable, shuffle=self.shuffle)
            # TODO: do we need to -np.min in cropped data?
        label = label.squeeze(-1).astype(np.long)
        data = {'pos': coord, 'x': feat, 'y': label}
        # pre-process.
        if self.transform is not None:
            data = self.transform(data)

        if 'heights' not in data.keys():
            data['heights'] =  torch.from_numpy(coord[:, self.gravity_dim:self.gravity_dim+1].astype(np.float32))

        if hasattr(self, 'feat_mean'):
            if not hasattr(self, '_stats_logged'):
                # logging.info(f'Normalization stats: feat_mean={self.feat_mean.tolist()}, '
                #              f'feat_std={self.feat_std.tolist()}, '
                #              f'z_mean={self.z_mean.item():.2f}, z_std={self.z_std.item():.2f}')
                self._stats_logged = True
            data['x'] = (data['x'] - self.feat_mean.to(data['x'].device)) / self.feat_std.to(data['x'].device).clamp(min=1e-5)
            data['heights'] = (data['heights'] - self.z_mean.to(data['heights'].device)) / self.z_std.to(data['heights'].device).clamp(min=1e-5)

        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
        # return 1   # debug


"""debug
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""
