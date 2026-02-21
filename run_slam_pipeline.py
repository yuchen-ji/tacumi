"""
Main script for UMI SLAM pipeline.
python run_slam_pipeline.py <session_dir>
"""

import sys
import os

# 这几行代码解决import时的路径问题
ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

# %%
@click.command()
@click.argument('session_dir', nargs=-1)    # 接收的应该是一个元组 ('example_demo_session', )
@click.option('-c', '--calibration_dir', type=str, default=None)
def main(session_dir, calibration_dir):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = pathlib.Path(__file__).parent.joinpath('example', 'calibration')
    else:
        calibration_dir = pathlib.Path(calibration_dir)
    assert calibration_dir.is_dir()

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        # 将原始视频重命名，并移动到特定文件夹下
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_imu ###########")
        # 将本地视频文件挂在到slam docker上，运行slam程序，输出文件：
        # 1. imu_data.json
        # 2. extract_gopro_imu_stdout.txt   标准输出 useless
        # 3. extract_gopro_imu_stderr.txt   标准错误 useless
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        assert mapping_dir.is_dir()
        map_path = mapping_dir.joinpath('map_atlas.osa')    # 二进制文件
        if not map_path.is_file():
            cmd = [
                'python', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path)
            ]
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
        aruco_config = calibration_dir.joinpath('aruco_config.yaml')
        assert camera_intrinsics.is_file()
        assert aruco_config.is_file()

        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    session_dir = ("my_demo_session2",)
    main(session_dir)
    # main()

# %%
