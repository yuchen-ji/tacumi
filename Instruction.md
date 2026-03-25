# 一些常用指令，方便后续操作

### 1 生成训练数据

```bash
# 首先运行SLAM
python run_slam_pipeline.py traning_data/demo_session_1

# 然后生成训练数据
python scripts_slam_pipeline/07_generate_replay_buffer.py -o training_data/demo_session_1/dataset.zarr.zip training_data/demo_session_1
```

### 2 在服务器上训练模型

```bash
# 将代码/数据传送到服务器，使用rsync可以断点续传
rsync -avh --info=progress2 --partial --append-verify \
  -e "ssh -T -o Compression=no -c aes128-gcm@openssh.com" \
  umi2.tar.gz  dhu:/raid/users/lsm

# 使用docker镜像站，渡渡鸟镜像站下载镜像

```

### 3 移植处理触觉模态的代码，from touch in the wild
#### 3.1 2个配置文件
主要通过这2个配置文件来确定后续代码是否需要使用tactile数据。
1. umi.yaml
```yaml
# TODO: 这里是和触觉相关的配置文件
# 当使用触觉数据时，需要取消注释，(requires tactile.npy in demo data)
camera0_tactile:
  shape: [12, 64]
  horizon: ${task.img_obs_horizon} # int
  latency_steps: 0 # float
  down_sample_steps: ${task.obs_down_sample_steps} # int
  type: tactile
  ignore_by_policy: False
```
2. train_diffusion_transformer_umi_workspace.yaml
```yaml
# TODO: 这里是和触觉相关的配置文件
# 当使用触觉数据时，需要将use_tactile设置为true
# 并且取消注释camera0_tactile in task/umi.yaml
use_tactile: false
tactile_model_choice: "simple_cnn"  # or "resnet18"
```

#### 3.2 修改的代码
1. 07_generate_replay_buffer.py
```python
# ADDED 26.03.22 新增对触觉数据的支持
# Load tactile data if available (tactile.npy alongside raw_video.mp4)
for cam_id, camera in enumerate(cameras):
    video_path_rel = camera['video_path']
    video_path_abs = demos_path.joinpath(video_path_rel).absolute()
    video_start, video_end = camera['video_start_end']

    npy_path = video_path_abs.parent.joinpath('tactile.npy')
    if npy_path.is_file():
        tactile_arr = np.load(npy_path, allow_pickle=True)
        tactile_slice = tactile_arr[video_start:video_end]
        episode_data[f'camera{cam_id}_tactile'] = tactile_slice.astype(np.float32)
        print("tactile data added")
# ADDED END
```
2. sampler.py
这个文件主要是从数据集中采样，被`umi_dataset.py`调用
```python
# 新增多处处理tactile的代码，会根据是否有tactile数据来处理
# 同样支持没有tactile数据的情况
if tactile_keys is None:
    tactile_keys = []
```

3. umi_dataset.py
```python
# 在数据集中，增加了对tactile模态的支持
# 当tactile模态缺省时，也同样支持
tactile_keys = list()
# ......
elif type == 'tactile':
    tactile_keys.append(key)
```

4. transformer_obs_encoder.py
```python
# 新增了simplecnn作为 tactile encoder
# 同时，支持让transformer的encoder加上tactile feature。所有feature cat到一起
if self.use_tactile:
    for key in self.tactile_keys:
        tactile_data = obs_dict[key]
        B, T = tactile_data.shape[:2]
        assert B == batch_size
        tactile_data = tactile_data.reshape(B * T, *tactile_data.shape[2:])

        left_tactile = tactile_data[:, :, :32].clamp(0, 1)
        right_tactile = tactile_data[:, :, 32:].clamp(0, 1)

        left_index = (left_tactile * 255).long().clamp(0, 255)
        right_index = (right_tactile * 255).long().clamp(0, 255)

        left_color = self.viridis_map[left_index]
        right_color = self.viridis_map[right_index]
        tactile_images = torch.cat([left_color, right_color], dim=1)
        tactile_images = tactile_images.permute(0, 3, 1, 2)

        device = next(self.key_model_map[key].parameters()).device
        tactile_images = tactile_images.to(device)

        raw_feature = self.key_model_map[key](tactile_images)
        assert raw_feature.dim() == 2
        assert raw_feature.size(0) == B * T

        emb = raw_feature.reshape(B, T, self.n_emb)
        embeddings.append(emb)
```

5. real_inference_util.py
```python
# 推理代码时，在获取数据阶段，新增了对触觉数据的获取
# get_real_obs_dict() 和 get_real_umi_obs_dict()
# 需要在配置文件中启用对tactile的支持
elif type == 'tactile':
    this_data_in = env_obs[key]
    obs_dict_np[key] = this_data_in
```

