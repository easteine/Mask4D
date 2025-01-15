import os
import random

import numpy as np
import torch
import yaml
from mask_4d.utils.data_util import data_prepare
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from plyfile import PlyData


class RScanSemanticDatasetModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.things_ids = []
        self.color_map = []

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        with open(self.cfg.DATASET.META_FILE, 'r') as f:
            metadata = json.load(f)
            
        train_scans = [scan for scan in metadata if scan['type'] == 'train']
        val_scans = [scan for scan in metadata if scan['type'] == 'validation']
        test_scans = [scan for scan in metadata if scan['type'] == 'test']

            
        train_set = SemanticDataset(
            self.cfg.DATASET.PATH,
            train_scans,
            self.cfg.KITTI.CONFIG,
            split="train",
        )
        
        train_mask_set = MaskSemanticDataset(
            dataset=train_set,
            split="train",
            min_pts=self.cfg.DATASET.MIN_POINTS,
            space=self.cfg.DATASET.SPACE,
            num_pts=self.cfg.DATASET.SUB_NUM_POINTS,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )
        
        self.train_seq_mask = SequenceMaskDataset(
            train_mask_set,
            n_scans=self.cfg.TRAIN.N_SCANS,
            interval=self.cfg.TRAIN.INTERVAL,
        )

        val_set = SemanticDataset(
            self.cfg.DATASET.PATH,
            val_scans, 
            self.cfg.KITTI.CONFIG,
            split="valid"
        )
        
        self.val_mask_set = MaskSemanticDataset(
            dataset=val_set,
            split="valid",
            min_pts=self.cfg.DATASET.MIN_POINTS,
            space=self.cfg.DATASET.SPACE,
            num_pts=0,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )

        test_set = SemanticDataset(
            self.cfg.DATASET.PATH,
            test_scans,
            self.cfg.KITTI.CONFIG,
            split="test"
        )
        
        self.test_mask_set = MaskSemanticDataset(
            dataset=test_set,
            split="test",
            min_pts=self.cfg.DATASET.MIN_POINTS,
            space=self.cfg.DATASET.SPACE,
            num_pts=0,
            voxel_size=self.cfg.BACKBONE.VOXEL_SIZE,
            voxel_max=self.cfg.BACKBONE.VOXEL_MAX,
        )

        # self.things_ids = train_set.things_ids
        # self.color_map = train_set.color_map

    def train_dataloader(self):
        dataset = self.train_seq_mask
        collate_fn = SphericalSequenceCollation()
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.train_iter = iter(self.train_loader)
        return self.train_loader

    def val_dataloader(self):
        dataset = self.val_mask_set
        collate_fn = SphericalBatchCollation()
        self.valid_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.valid_iter = iter(self.valid_loader)
        return self.valid_loader

    def test_dataloader(self):
        dataset = self.test_mask_set
        collate_fn = SphericalBatchCollation()
        self.test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.cfg.TRAIN.BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=self.cfg.TRAIN.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            timeout=0,
        )
        self.test_iter = iter(self.test_loader)
        return self.test_loader


class SemanticDataset(Dataset):
    def __init__(self, data_path, scan_list, cfg_path, split="train"):
        self.data_path = data_path
        self.scan_list = scan_list
        self.split = split


        yaml_path = cfg_path
        with open(yaml_path, "r") as stream:
            semyaml = yaml.safe_load(stream)

        self.color_map = semyaml["color_map_learning"]
        self.labels = semyaml["labels"]
        self.learning_map = semyaml["learning_map"]
        self.inv_learning_map = semyaml["learning_map_inv"]


        self.things = semyaml['things']
        self.stuff = semyaml['stuff']
        self.things_ids = list(self.things.keys())

        # Build list of all scan paths
        self.scan_paths = []
        self.reference_paths = []
        for scan in scan_list:
            ref_scan = scan['reference']
            ref_path = os.path.join(data_path, ref_scan)
            self.reference_paths.append(ref_path)
            
            # Add paths for each rescan
            for rescan in scan['scans']:
                rescan_path = os.path.join(data_path, rescan['reference'])
                self.scan_paths.append({
                    'path': rescan_path,
                    'reference': ref_path,
                    'transform': np.array(rescan['transform']).reshape(4,4) if 'transform' in rescan else None,
                })


    def __len__(self):
        return len(self.im_idx)

    def __getitem__(self, index):
        scan_data = self.scan_paths[index]
        scan_path = scan_data['path']
        
        # Load point cloud and instance labels
        pc_path = os.path.join(scan_path, 'pointcloud/pointcloud.scan.ply')
        inst_path = os.path.join(scan_path, 'pointcloud/pointcloud.instances.ply')

        xyz, colors = read_ply(str(pc_path))
        intensity = np.asarray(colors)[:, 0]

        # Extract instance and semantic labels
        # Assuming instance labels are stored in vertex labels
        if self.split == "test":
            sem_labels = np.zeros((xyz.shape[0], 1), dtype=int)
            ins_labels = np.zeros((xyz.shape[0], 1), dtype=int)
        else:
            inst_data = PlyData.read(str(inst_path))
            vertex_data = inst_data['vertex']

            # Extract RIO27 semantic labels and instance IDs
            sem_labels = vertex_data['RIO27'].reshape(-1, 1)
            sem_labels = np.vectorize(self.learning_map.__getitem__)(sem_labels)

            ins_labels = vertex_data['objectId'].reshape(-1, 1)

            # rgb = np.column_stack([vertex_data[x] for x in ['red', 'green', 'blue']])

            # Apply transformation if available
            if scan_data['transform'] is not None:
                xyz = (scan_data['transform'][:3, :3] @ xyz.T + 
                      scan_data['transform'][:3, 3:]).T

        return (xyz, sem_labels, ins_labels, intensity, scan_path, scan_data['transform'])


class MaskSemanticDataset(Dataset):
    def __init__(
        self,
        dataset,
        split,
        min_pts,
        space,
        num_pts,
        voxel_size,
        voxel_max,
    ):
        self.dataset = dataset
        self.num_points = num_pts
        self.split = split
        self.min_points = min_pts
        self.th_ids = dataset.things_ids
        self.voxel_size = np.array(voxel_size)
        self.voxel_max = voxel_max
        # no spatial bounds for 3RScan since they are contained scans

        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        xyz, sem_labels, ins_labels, intensity, fname, pose = data

        feats = np.concatenate((xyz, np.expand_dims(intensity, axis=1)), axis=1)

        # Subsample
        if self.split == "train" and len(xyz) > self.num_points:
            idx = np.random.choice(np.arange(len(xyz)), self.num_points, replace=False)
            xyz = xyz[idx]
            sem_labels = sem_labels[idx]
            ins_labels = ins_labels[idx]
            feats = feats[idx]
            intensity = intensity[idx]

        # SphereFormer
        sp_xyz = xyz.copy()
        sp_coords, sp_xyz, sp_feats, sp_labels, sp_idx_recons = data_prepare(
            sp_xyz,
            feats,
            sem_labels,
            self.split,
            self.voxel_size,
            self.voxel_max,
        )

        if self.split == "test":
            return (
                xyz,
                sem_labels,
                ins_labels,
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                fname,
                pose,
                sp_coords,
                sp_xyz,
                sp_feats,
                sp_labels,
                sp_idx_recons,
            )

        stuff_masks = np.array([]).reshape(0, xyz.shape[0])
        stuff_masks_ids = []
        things_masks = np.array([]).reshape(0, xyz.shape[0])
        things_cls = np.array([], dtype=int)
        things_masks_ids = []

        stuff_labels = np.asarray(
            [0 if s in self.th_ids else s for s in sem_labels[:, 0]]
        )
        stuff_cls, st_cnt = np.unique(stuff_labels, return_counts=True)
        # filter small masks
        keep_st = np.argwhere(st_cnt > self.min_points)[:, 0]
        stuff_cls = stuff_cls[keep_st][1:]
        if len(stuff_cls):
            stuff_masks = np.array(
                [np.where(stuff_labels == i, 1.0, 0.0) for i in stuff_cls]
            )
            stuff_masks_ids = [0 for m in stuff_masks]
        # things masks
        ins_sems = np.where(ins_labels == 0, 0, sem_labels)
        _ins_labels = ins_sems + ((ins_labels << 16) & 0xFFFF0000).reshape(-1, 1)
        things_ids, th_idx, th_cnt = np.unique(
            _ins_labels[:, 0], return_index=True, return_counts=True
        )
        # filter small instances
        keep_th = np.argwhere(th_cnt > self.min_points)[:, 0]
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        # remove instances with wrong sem class
        keep_th = np.array(
            [i for i, idx in enumerate(th_idx) if sem_labels[idx] in self.th_ids],
            dtype=int,
        )
        things_ids = things_ids[keep_th]
        th_idx = th_idx[keep_th]
        if len(th_idx):
            things_masks = np.array(
                [np.where(_ins_labels[:, 0] == i, 1.0, 0.0) for i in things_ids]
            )
            things_cls = np.array([sem_labels[i] for i in th_idx]).squeeze(1)
            things_masks_ids = [t for t in things_ids]

        masks = torch.from_numpy(np.concatenate((stuff_masks, things_masks)))
        masks_cls = torch.from_numpy(np.concatenate((stuff_cls, things_cls)))
        stuff_masks_ids.extend(things_masks_ids)
        masks_ids = torch.tensor(stuff_masks_ids)

        assert (
            masks.shape[0] == masks_cls.shape[0]
        ), f"not same number masks and classes: masks {masks.shape[0]}, classes {masks_cls.shape[0]} "

        return (
            xyz,
            sem_labels,
            ins_labels,
            masks,
            masks_cls,
            masks_ids,
            fname,
            pose,
            sp_coords,
            sp_xyz,
            sp_feats,
            sp_labels,
            sp_idx_recons,
        )


class SequenceMaskDataset(Dataset):
    def __init__(self, dataset, n_scans, interval):
        super().__init__()
        self.dataset = dataset
        self.n_scans = n_scans
        self.interval = interval

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        fname = data[6]
        _fname = fname.split("/")
        scan = int(_fname[-1][:-4])
        seq = _fname[-3]
        max_scan = self.dataset.dataset.n_scans[seq] - 1

        # select n_scans randomly between scan-interval/2 and scan+interval/2
        s_before = random.sample(
            set(np.arange(scan - int(self.interval / 2), scan)), round(self.n_scans / 2)
        )
        s_after = random.sample(
            set(np.arange(scan, scan + int(self.interval / 2))), self.n_scans // 2
        )
        scans = s_before + s_after
        scans.sort(reverse=True)
        while scans[-1] < 0:
            scans.pop()
        scans.sort()
        while scans[-1] > max_scan:
            scans.pop()
        idx = index + np.array(scans) - scan
        np.append(idx, np.array([index]))
        idx.sort()
        out_data = []
        for i in idx:
            out_data.append(self.dataset[i])

        return out_data


class SphericalBatchCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "sem_label",
            "ins_label",
            "masks",
            "masks_cls",
            "masks_ids",
            "fname",
            "pose",
            "sp_coord",
            "sp_xyz",
            "sp_feat",
            "sp_label",
            "sp_idx_recons",
        ]

    def __call__(self, data):
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*data))}


class SphericalSequenceCollation:
    def __init__(self):
        self.keys = [
            "pt_coord",
            "sem_label",
            "ins_label",
            "masks",
            "masks_cls",
            "masks_ids",
            "fname",
            "pose",
            "sp_coord",
            "sp_xyz",
            "sp_feat",
            "sp_label",
            "sp_idx_recons",
        ]

    def __call__(self, data):
        _data = data[0]  # always bs=0
        return {self.keys[i]: list(x) for i, x in enumerate(zip(*_data))}


def read_ply(file_path):
    ply_data = PlyData.read(file_path)
    
    # Extract vertex data
    vertex = ply_data['vertex']
    
    # Get x, y, z coordinates
    x = vertex['x']
    y = vertex['y']
    z = vertex['z']
    xyz = np.column_stack((x, y, z))
    
    # Get color data if available
    colors = None
    if 'red' in vertex.dtype.names:
        red = vertex['red']
        green = vertex['green']
        blue = vertex['blue']
        colors = np.column_stack((red, green, blue))
    
    return xyz, colors