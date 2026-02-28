# 一些常用指令，方便后续操作

### 1 生成训练数据

```bash
# 首先运行SLAM
python run_slam_pipeline.py traning_data/demo_session_1

# 然后生成训练数据
python scripts_slam_pipeline/07_generate_replay_buffer.py -o training_data/demo_session_1/dataset.zarr.zip training_data/demo_session_1
```

## 2 在服务器上训练模型

```bash
# 将代码/数据传送到服务器，使用rsync可以断点续传
rsync -avh --info=progress2 --partial --append-verify \
  -e "ssh -T -o Compression=no -c aes128-gcm@openssh.com" \
  umi2.tar.gz  dhu:/raid/users/lsm

# 使用docker镜像站，渡渡鸟镜像站下载镜像

```

