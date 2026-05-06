"""
python scripts_slam_pipeline/05_run_calibrations.py session_dir1 session_dir2 ...
"""

"""
它不是再去处理视频本身，而是利用前面步骤已经得到的轨迹和 ArUco 检测结果，去做两类标定（calibration）：
(1)SLAM 坐标系和 tag 坐标系之间的标定
(2)gripper 工作范围的标定

拿前面得到的轨迹 CSV 和 tag_detection.pkl，再进一步算出一些标定结果 JSON。
"""

"""
为什么要做 slam-tag 标定

因为：

SLAM 给你的是一个SLAM 自己内部的地图坐标系
ArUco tag 检测给你的是一个相对于 tag 的坐标参考

这两个坐标系天然不是同一个。

所以必须通过标定求出一个变换，让系统知道：

“SLAM 坐标系里的点，怎么转换到 tag 坐标系里去”
"""

""""
第一步：做 slam-tag 标定,使用：

demos/mapping/tag_detection.pkl
demos/mapping/camera_trajectory.csv
或 mapping_camera_trajectory.csv

生成：demos/mapping/tx_slam_tag.json

第二步：做 gripper range 标定

对每个：demos/gripper_calibration*目录，使用：tag_detection.pkl

生成：gripper_range.json
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

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    script_dir = pathlib.Path(__file__).parent.parent.joinpath('scripts')
    
    for session in session_dir:
        session = pathlib.Path(session)
        demos_dir = session.joinpath('demos')
        mapping_dir = demos_dir.joinpath('mapping')
        slam_tag_path = mapping_dir.joinpath('tx_slam_tag.json')
            
        # run slam tag calibration
        script_path = script_dir.joinpath('calibrate_slam_tag.py')
        assert script_path.is_file()
        tag_path = mapping_dir.joinpath('tag_detection.pkl')
        assert tag_path.is_file()
        csv_path = mapping_dir.joinpath('camera_trajectory.csv')
        if not csv_path.is_file():
            csv_path = mapping_dir.joinpath('mapping_camera_trajectory.csv')
            print("camera_trajectory.csv not found! using mapping_camera_trajectory.csv")
        assert csv_path.is_file()
        
        cmd = [
            'python', str(script_path),
            '--tag_detection', str(tag_path),
            '--csv_trajectory', str(csv_path),
            '--output', str(slam_tag_path),
            '--keyframe_only'
        ]
        subprocess.run(cmd)
        
        # 个人理解，可能对于使用两个gripper操作的需要分别标定
        script_path = script_dir.joinpath('calibrate_gripper_range.py')
        assert script_path.is_file()

        for gripper_dir in demos_dir.glob("gripper_calibration*"):
            gripper_range_path = gripper_dir.joinpath('gripper_range.json')
            tag_path = gripper_dir.joinpath('tag_detection.pkl')
            assert tag_path.is_file()
            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path)
            ]
            subprocess.run(cmd)

            
# %%
if __name__ == "__main__":
    main()
