"""
python scripts_slam_pipeline/00_process_videos.py -i session_dir1/demos/mapping
"""
"""
对一个已经准备好的 mapping 视频目录，调用 Docker 里的 ORB_SLAM3 程序，
结合视频和 IMU 数据运行单目惯性 SLAM，输出相机轨迹 mapping_camera_trajectory.csv，并保存地图文件 map_atlas.osa。

"""

"""
输入的是：raw_video.mp4 与 imu_data.json

所以它是在做一种 视觉 + IMU 的 SLAM，也就是：

从视频画面里找特征点
结合 IMU 的加速度、角速度信息
估计相机怎么移动
再把环境地图建出来

所以最后它会输出：

mapping_camera_trajectory.csv：相机运动轨迹
map_atlas.osa：地图文件


SLAM 关心的“静态”，是：物体在现实世界里是不是固定的
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import numpy as np
import cv2   #把 mask 保存成 PNG
from umi.common.cv_util import draw_predefined_mask

# %%
@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for mapping video')
@click.option('-m', '--map_path', default=None, help='ORB_SLAM3 *.osa map atlas file')
@click.option('-d', '--docker_image', default="chicheng/orb_slam3:latest")
@click.option('-np', '--no_docker_pull', is_flag=True, default=False, help="pull docker image from docker hub")
@click.option('-nm', '--no_mask', is_flag=True, default=False, help="Whether to mask out gripper and mirrors. Set if map is created with bare GoPro no on gripper.")
def main(input_dir, map_path, docker_image, no_docker_pull, no_mask):
    video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
    for fn in ['raw_video.mp4', 'imu_data.json']:
        assert video_dir.joinpath(fn).is_file()  #强制保证：这个目录已经准备好了“视频 + IMU 数据”，可以拿去跑 SLAM。

    if map_path is None:
        map_path = video_dir.joinpath('map_atlas.osa')#如果用户没给 --map_path，默认输出到当前 video_dir/map_atlas.osa
    else:
        map_path = pathlib.Path(os.path.expanduser(map_path)).absolute()
    map_path.parent.mkdir(parents=True, exist_ok=True)  #确保这个地图文件所在目录存在

    # pull docker
    if not no_docker_pull:    #  如果你本地已有镜像且不想更新，可以加 -np
        print(f"Pulling docker image {docker_image}")
        cmd = [
            'docker',
            'pull',
            docker_image
        ]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            print("Docker pull failed!")
            exit(1)

    mount_target = pathlib.Path('/data')
    csv_path = mount_target.joinpath('mapping_camera_trajectory.csv')
    video_path = mount_target.joinpath('raw_video.mp4')
    json_path = mount_target.joinpath('imu_data.json')
    mask_path = mount_target.joinpath('slam_mask.png')

    """
    这里的 mount_target 是 Docker 容器内部的路径，/data/raw_video.mp4 就是容器里视频文件的路径
    这个路径和宿主机的路径 video_dir 是通过 Docker 的 --volume 参数 连接起来的
    也就是说，容器里的 /data/raw_video.mp4 实际上就是宿主机的 video_dir/raw_video.mp4
    """
    if not no_mask:
        mask_write_path = video_dir.joinpath('slam_mask.png')
        slam_mask = np.zeros((2028, 2704), dtype=np.uint8) #先创建了一张全黑图：大小是 2028 x 2704，像素值全是 0

        slam_mask = draw_predefined_mask(
            #slam_mask, color=255, mirror=True, gripper=False, finger=True)       #保留镜子的版本
            slam_mask, color=255, mirror=False, gripper=False, finger=True)       #去掉镜子，保留夹爪的版本
        
        # 白色区域：要被 mask 掉 / 忽略 的区域
        # 黑色区域：SLAM 正常使用 的区域 
        # False是黑色，True是白色
        cv2.imwrite(str(mask_write_path.absolute()), slam_mask)   #把生成的 mask 写到：video_dir/slam_mask.png

    map_mount_source = pathlib.Path(map_path)
    map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)

    """
    假设宿主机上的地图输出路径是：/home/user/session/demos/mapping/map_atlas.osa
    那么：
    map_mount_source 是宿主机路径
    map_mount_target 是容器内路径，例如：/map/map_atlas.osa
    
    """

    # run SLAM
    cmd = [
        'docker',
        'run',
        '--rm', # delete after finish
        '--volume', str(video_dir) + ':' + '/data', #把当前视频目录挂载到容器里的 /data，这样容器里就能访问到视频和 IMU 数据了
        '--volume', str(map_mount_source.parent) + ':' + str(map_mount_target.parent),#把地图输出目录挂载到容器里的 /map，这样容器里的 SLAM 程序就能把地图文件写到这个目录了
        docker_image,
        '/ORB_SLAM3/Examples/Monocular-Inertial/gopro_slam',
        '--vocabulary', '/ORB_SLAM3/Vocabulary/ORBvoc.txt',
        '--setting', '/ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1_720.yaml',
        '--input_video', str(video_path),
        '--input_imu_json', str(json_path),
        '--output_trajectory_csv', str(csv_path),
        '--save_map', str(map_mount_target)
    ]
    if not no_mask:
        cmd.extend([
            '--mask_img', str(mask_path)
        ])

    stdout_path = video_dir.joinpath('slam_stdout.txt')
    stderr_path = video_dir.joinpath('slam_stderr.txt')

    # 这里cwd的作用是让子进程使用video_dir作为当前工作目录
    # 子进程中，所有相对路径都是相对于video_dir的，绝对路径不受影响
    result = subprocess.run(
        cmd,
        cwd=str(video_dir),
        stdout=stdout_path.open('w'),
        stderr=stderr_path.open('w')
    )
    """
    执行前面构造好的 docker run ... 命令
    当前工作目录设为 video_dir
    标准输出写到 slam_stdout.txt
    标准错误写到 slam_stderr.txt
    最后在终端打印一个 CompletedProcess 结果对象
    """
    print(result)


# %%
if __name__ == "__main__":
    main()
