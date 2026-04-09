import copy
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from filelock import FileLock
from typing import Dict, Optional
from datetime import datetime

import zarr
from threadpoolctl import threadpool_limits
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

import random
import scipy.interpolate as si
import scipy.spatial.transform as st


def get_val_mask(n_episodes, val_ratio, seed=0):
    """Return a boolean mask of length n_episodes, with `val_ratio` fraction set to True."""
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask

class SequenceSampler:
    """
    Minimal sequence sampler that picks single frames from each episode.
    Only returning:
    - current camera image
    - current tactile
    """
    def __init__(
        self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        tactile_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray] = None,
        action_padding: bool = False,
        repeat_frame_prob: float = 0.0,
        max_duration: Optional[float] = None
    ):
        episode_ends = replay_buffer.episode_ends[:]
        try:
            gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
            gripper_width_threshold = 0.08
        except KeyError:
            gripper_width = np.full(episode_ends[-1], 9999.0)
            gripper_width_threshold = -9999.0

        self.repeat_frame_prob = repeat_frame_prob

        # Build indices for frames (one index per frame).
        indices = []
        for i in range(len(episode_ends)):
            if episode_mask is not None and not episode_mask[i]:
                continue
            start_idx = 0 if i == 0 else episode_ends[i - 1]
            end_idx = episode_ends[i]

            if max_duration is not None:
                end_idx = min(end_idx, int(max_duration * 60))

            for current_idx in range(start_idx, end_idx, 5):
                indices.append(current_idx)

        # Extract only relevant keys (camera, tactile) into a dict for quick access
        self.replay_buffer = dict()
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        for key in tactile_keys:
            self.replay_buffer[key] = replay_buffer[key]

        self.indices = indices
        self.rgb_keys = rgb_keys
        self.tactile_keys = tactile_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps

        self.ignore_rgb_is_applied = False

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        """Return a single camera+tactile 'frame' given the global frame index."""
        current_idx = self.indices[idx]
        result = dict()

        obs_keys = self.rgb_keys + self.tactile_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.tactile_keys

        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            assert this_latency_steps == 0, \
                f"Expected 0-latency for {key}, got {this_latency_steps}!"

            # Just extract [current_idx : current_idx+1]
            output = input_arr[current_idx:current_idx + 1]
            if output.shape[0] < this_horizon:
                # basic padding, though for 1 frame it shouldn't matter
                padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                output = np.concatenate([padding, output], axis=0)

            result[key] = output

        # optional "repeat_frame_prob" logic
        if self.repeat_frame_prob != 0.0 and random.random() < self.repeat_frame_prob:
            for key in obs_keys:
                result[key][:-1] = result[key][-1:]

        return result

    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply


class TactileAutoencoderDataset(BaseDataset):
    """
    A dataset that:
      1) Randomly picks ~10% episodes for validation (val_ratio=0.1).
      2) From the remaining episodes, picks a 'train_ratio' fraction (e.g. 0.5, 1.0, etc.).
         If train_ratio=1.0 => use all non-validation episodes.
      3) Returns single-frame samples from whichever episodes are designated for training or validation.
    """
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str] = None,
        pose_repr: dict = {},
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,      # fraction of episodes to hold out for val
        train_ratio: float = 1.0,    # fraction of *non-val* episodes to use for train
        max_duration: Optional[float] = None,
        mask_ratio: float = 0.5,
        tactile_mask_ratio: float = 0.3,
        transforms: list = None,
    ):
        super().__init__()

        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        self.mask_ratio = mask_ratio
        self.tactile_mask_ratio = tactile_mask_ratio

        # (1) Load the replay buffer
        if cache_dir is None:
            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store,
                    store=zarr.MemoryStore()
                )
        else:
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),
                                            writemap=True, metasync=False,
                                            sync=False, map_async=True, lock=False) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        import shutil
                        shutil.rmtree(cache_path)
                        raise e
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(zarr.group(store))

        self.num_robot = 0
        rgb_keys = []
        lowdim_keys = []
        tactile_keys = []
        key_horizon = {}
        key_down_sample_steps = {}
        key_latency_steps = {}

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            typ = attr.get('type', 'low_dim')
            if typ == 'rgb':
                rgb_keys.append(key)
            elif typ == 'low_dim':
                lowdim_keys.append(key)
            elif typ == 'tactile':
                tactile_keys.append(key)

            if key.endswith('eef_pos'):
                self.num_robot += 1
            horizon = attr['horizon']
            key_horizon[key] = horizon
            key_latency_steps[key] = attr['latency_steps']
            key_down_sample_steps[key] = attr['down_sample_steps']

        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        n_episodes = replay_buffer.n_episodes

        val_mask = get_val_mask(
            n_episodes=n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        base_train_mask = ~val_mask

        rng = np.random.default_rng(seed=seed+999)  # a second RNG
        if train_ratio < 1.0:
            train_episodes = np.where(base_train_mask)[0]  # episodes that are currently train
            n_keep = int(len(train_episodes) * train_ratio)
            chosen = rng.choice(train_episodes, size=n_keep, replace=False)
            final_train_mask = np.zeros(n_episodes, dtype=bool)
            final_train_mask[chosen] = True
            train_mask = final_train_mask
        else:
            train_mask = base_train_mask

        self.num_val_episodes = val_mask.sum()
        self.num_train_episodes = train_mask.sum()

        self.sampler_lowdim_keys = []
        for k in lowdim_keys:
            if 'wrt' not in k:
                self.sampler_lowdim_keys.append(k)

        for k in replay_buffer.keys():
            if k.endswith('_demo_start_pose') or k.endswith('_demo_end_pose'):
                self.sampler_lowdim_keys.append(k)
                query_key = k.split('_')[0] + '_eef_pos'
                key_horizon[k] = shape_meta['obs'][query_key]['horizon']
                key_latency_steps[k] = shape_meta['obs'][query_key]['latency_steps']
                key_down_sample_steps[k] = shape_meta['obs'][query_key]['down_sample_steps']

        # Build the final sampler with the final train_mask
        sampler = SequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            tactile_keys=tactile_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration
        )

        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.tactile_keys = tactile_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

        self.key_transform_map = nn.ModuleDict()
        self.image_shape = None
        for rk in self.rgb_keys:
            shape = tuple(obs_shape_meta[rk]['shape'])
            self.image_shape = shape[1:]  # e.g., (224, 224)
            break

        self.rgb_transform = (
            nn.Identity() if (not transforms) else nn.Sequential(*transforms)
        )
        for rk in self.rgb_keys:
            self.key_transform_map[rk] = self.rgb_transform

    def get_validation_dataset(self):
        """
        Return a copy dataset that samples from the chosen validation episodes only.
        We reuse the same 'val_mask', ignoring the train-ratio mask.
        """
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            tactile_keys=self.tactile_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,  # <--- only val episodes
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration
        )
        # Count how many val episodes we have in val_set
        val_set.num_val_episodes = self.val_mask.sum()
        return val_set

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True

        data = self.sampler.sample_sequence(idx)

        # --- Process Camera (Use only the first rgb key) ---
        for rk in self.rgb_keys:
            rgb = data[rk].astype(np.float32) / 255.0
            cur_np = rgb[0]  # shape (H, W, 3)
            cur_tensor = torch.from_numpy(cur_np).permute(2, 0, 1)  # (3, H, W)
            # apply any transforms
            cur_tensor = self.key_transform_map[rk](cur_tensor)
            break  # only use the first camera

        rgb_cur = cur_tensor

        # --- Process Tactile (Use only the first tactile key) ---
        for tk in self.tactile_keys:
            tact = data[tk].astype(np.float32)
            tact_cur_np = tact[0]  # shape (12, 64)
            tact_cur = torch.from_numpy(tact_cur_np)
            break

        return {
            'rgb_cur': rgb_cur,         # (3, 224, 224) or after transforms
            'tactile_cur': tact_cur,    # (12, 64)
        }
