from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class MyPushtDataset(BaseImageDataset):
    """
    专门为您的自定义数据采集器 (collect_data.py) 编写的数据集加载类。
    处理的 Key: front_image, wrist_image, robot_eef_pose, action
    处理的图像形状: 已经是在采集时转换好的 (C, H, W)，所以无需再 moveaxis。
    """
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path, mode='r')
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # 将低维动作和位置进行归一化
        data = {
            'action': self.replay_buffer['action'],
            'robot_eef_pose': self.replay_buffer['robot_eef_pose']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # 将图像使用 0-1 的范围归一化
        normalizer['front_image'] = get_image_range_normalizer()
        normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        import cv2
        # 取决于显存大小，这里我们将(360, 480)的图像缩防到(96, 96)以便高效训练
        # 采集时已经是 (3, 360, 480) [T, C, H, W]
        # 使用 numpy 和 cv2 来循环批量 resize
        T = sample['front_image'].shape[0]
        
        front_resized = np.zeros((T, 3, 96, 96), dtype=np.float32)
        wrist_resized = np.zeros((T, 3, 96, 96), dtype=np.float32)
        
        for i in range(T):
            # 取出一帧，变为 (360, 480, 3) 用 cv2 处理
            f_img = np.moveaxis(sample['front_image'][i], 0, -1)
            w_img = np.moveaxis(sample['wrist_image'][i], 0, -1)
            
            f_img = cv2.resize(f_img, (96, 96), interpolation=cv2.INTER_AREA)
            w_img = cv2.resize(w_img, (96, 96), interpolation=cv2.INTER_AREA)
            
            # 再变回 (3, 96, 96)
            front_resized[i] = np.moveaxis(f_img, -1, 0)
            wrist_resized[i] = np.moveaxis(w_img, -1, 0)
        
        data = {
            'obs': {
                'front_image': front_resized / 255.0,
                'wrist_image': wrist_resized / 255.0,
                'robot_eef_pose': sample['robot_eef_pose'].astype(np.float32),
            },
            'action': sample['action'].astype(np.float32)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
