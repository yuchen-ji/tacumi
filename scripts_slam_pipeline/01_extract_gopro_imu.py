"""
python scripts_slam_pipeline/01_extract_gopro_imu.py session_dir1 session_dir2 ...
"""

"""
会遍历一个 session 目录下 demos/ 里的每个视频目录，调用 Docker 容器中的 GoPro 元数据提取脚本，
从 raw_video.mp4 中提取 IMU 数据，并生成 imu_data.json。

它不是处理视频画面本身，而是批量从 GoPro 视频里抽取 IMU / 传感器元数据
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib #路径处理库，比 os.path 更直观
import click
import subprocess#用来执行外部命令，比如调用 Docker,主要是docker pull 与 docker run
import multiprocessing
import concurrent.futures #用来并行执行多个任务，后面会用它来同时处理多个视频目录
from tqdm import tqdm

# %%
@click.command()
@click.option('-d', '--docker_image', default="chicheng/openicc:latest") #默认的 Docker 镜像，里面应该预装了 GoPro IMU 提取工具
@click.option('-n', '--num_workers', type=int, default=None) #如果不传，后面会自动设置成 CPU 核心数
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
#如果你加上：-np，就表示不要先执行 docker pull。默认情况下脚本会先拉取镜像，确保用的是最新版本
@click.argument('session_dir', nargs=-1)
def main(docker_image, num_workers, no_docker_pull, session_dir):
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # pull docker
    if not no_docker_pull:
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        """
         subprocess.run(cmd) 会执行 docker pull 命令，拉取指定的 Docker 镜像
         去远程仓库把 chicheng/openicc:latest 这个镜像准备到本地，供后面的 docker run 使用。

         本地没有 → 下载下来
         本地有旧版本 → 更新
         本地已有最新 → 很快结束
        """
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    for session in session_dir:
        input_dir = pathlib.Path(os.path.expanduser(session)).joinpath('demos')
        #input_dir 是：session_dir1/demos

        input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
        # input_video_dirs 是一个列表，包含了 demos/ 下面每个视频目录的路径，比如 session_dir1/demos/demo_123456_2026.03.30_10.20.30.123456
        # x.parent 是对应的视频目录路径
        # #demos/mapping/raw_video.mp4
        # #demos/demo_xxx/raw_video.mp4
        # #demos/gripper_calibration_xxx/raw_video.mp4
        print(f'Found {len(input_video_dirs)} video dirs')

        with tqdm(total=len(input_video_dirs)) as pbar:#创建一个总数为“视频目录数量”的进度条。每当一个视频处理完成，进度条就会更新
            # one chunk per thread, therefore no synchronization(同步) needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # ThreadPoolExecutor 是一个线程池，可以同时运行多个线程来处理任务。max_workers=num_workers 表示线程池中最多有 num_workers 个线程。
            # 这里我们用线程池来同时处理多个视频目录，每个线程负责一个视频目录的 IMU 提取任务。因为每个视频目录的处理相对独立
                futures = set()  #futures 用来保存当前已经提交、但还没结束的任务
                for video_dir in tqdm(input_video_dirs):
                    # video_dir 是 demo_Cxxxxx_2025 这种结构
                    video_dir = video_dir.absolute()
                    if video_dir.joinpath('imu_data.json').is_file():
                        print(f"imu_data.json already exists, skipping {video_dir.name}")
                        continue
                    mount_target = pathlib.Path('/data')

                    #video_path 和 json_path 不是宿主机路径，而是Docker 容器内部路径
                    video_path = mount_target.joinpath('raw_video.mp4')#输入视频在 /data/raw_video.mp4

                    json_path = mount_target.joinpath('imu_data.json')#输出 JSON 写到 /data/imu_data.json

                    # run imu extractor
                    cmd = [
                        'docker',
                        'run',
                        '--rm', # delete after finish
                        '--volume', str(video_dir) + ':' + '/data', # 将本地视频目录挂载到容器的/data目录
                        docker_image,
                        'node',
                        '/OpenImuCameraCalibrator/javascript/extract_metadata_single.js',
                        str(video_path),
                        str(json_path)
                    ]
                    """
                    它实际会执行一个类似这样的命令：
                    docker run --rm \
                        --volume /宿主机/某个video_dir:/data \
                        chicheng/openicc:latest \
                        node /OpenImuCameraCalibrator/javascript/extract_metadata_single.js \
                        /data/raw_video.mp4 \
                        /data/imu_data.json
                    """
                    stdout_path = video_dir.joinpath('extract_gopro_imu_stdout.txt')
                    stderr_path = video_dir.joinpath('extract_gopro_imu_stderr.txt')

                    if len(futures) >= num_workers:
                        # limit number of inflight tasks
                        completed, futures = concurrent.futures.wait(futures, 
                            return_when=concurrent.futures.FIRST_COMPLETED)
                        pbar.update(len(completed))
                    """
                    虽然已经用了线程池，但作者还手动控制了“正在飞行中的任务数量”。
                    当当前尚未完成的任务数已经达到 num_workers 时：
                    不再继续提交新任务
                    先等至少一个任务完成
                    完成多少个，就更新进度条多少格
                    然后继续提交新任务
                    这样做的好处是可以更平滑地更新进度条，而不是等所有任务都提交完才开始等待和更新。 
                    """

                    futures.add(executor.submit(
                        # 为什么不将 video_dir 作为lambda的参数传递？
                        lambda x, stdo, stde: subprocess.run(x, 
                            cwd=str(video_dir),
                            stdout=stdo.open('w'),
                            stderr=stde.open('w')), 
                        cmd, stdout_path, stderr_path))
                    # print(' '.join(cmd))

                completed, futures = concurrent.futures.wait(futures)
                pbar.update(len(completed))

        print("Done! Result:")
        print([x.result() for x in completed])

# %%
if __name__ == "__main__":
    main()
