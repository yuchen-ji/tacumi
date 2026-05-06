"""
python scripts_slam_pipeline/00_process_videos.py session_dir1 session_dir2 ...
"""

"""把一个 session_dir 目录下杂乱的 MP4 视频，按照 SLAM / 数据采集流程重新整理到标准结构里，并根据视频类型自动分类、重命名、移动，
   再在原位置创建软链接，方便后续 SLAM 和数据处理脚本的使用
"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)  #把项目根目录加入 Python 模块搜索路径
os.chdir(ROOT_DIR) #把当前工作目录切到项目根目录。这样后面如果有相对路径，都是相对于项目根目录来解释。

# %%
import pathlib  #路径处理库，比 os.path 更直观
import click  #命令行参数处理库，方便定义和解析命令行参数
import shutil  #主要用来做文件移动
from exiftool import ExifToolHelper  #用来读取视频的元数据，比如拍摄时间和相机序列号
from umi.common.timecode_util import mp4_get_start_datetime #这是项目内部函数，用于读取视频的开始时间，返回一个 datetime 对象 

# %%
@click.command(help='Session directories. Assumming mp4 videos are in <session_dir>/raw_videos') #把 main 变成一个命令行函数
@click.argument('session_dir', nargs=-1)#表示这个命令可以接收任意多个位置参数，参数名叫 session_dir
def main(session_dir):
    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()
        """
        os.path.expanduser(session)：展开 ~
        pathlib.Path(...)：转为 Path 对象
        .absolute()：转成绝对路径
        
        
        比如输入：~/data/session_01
        1. os.path.expanduser 会把 ~ 展开成 /home/user/data/session_01
        2. pathlib.Path 会把字符串转成 Path 对象，方便后续路径操作
        3. .absolute() 确保路径是绝对路径，虽然 expanduser 后通常已经是绝对路径了，但加上这个更保险

        """
        # 把 raw_videos 这个子目录名接到 session 路径后面，这是我们假设的输入视频目录结构
        input_dir = session.joinpath('raw_videos')  #输入视频目录：session/raw_videos

        output_dir = session.joinpath('demos')  #输出整理目录：session/demos

        # create raw_videos if don't exist 
        # input_dir = 'session_dir/raw_videos'
        if not input_dir.is_dir():
            input_dir.mkdir()
            print(f"{input_dir.name} subdir don't exits! Creating one and moving all mp4 videos inside.")
            for mp4_path in list(session.glob('**/*.MP4')) + list(session.glob('**/*.mp4')):
                out_path = input_dir.joinpath(mp4_path.name)
                shutil.move(mp4_path, out_path)
        
        # create mapping video if don't exist
        mapping_vid_path = input_dir.joinpath('mapping.mp4')
        if (not mapping_vid_path.exists()) and not(mapping_vid_path.is_symlink()):
            max_size = -1
            max_path = None
            # 通过比较文件的大小，来判断哪一个是mapping视频
            for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                size = mp4_path.stat().st_size
                if size > max_size:
                    max_size = size
                    max_path = mp4_path
            shutil.move(max_path, mapping_vid_path)
            print(f"raw_videos/mapping.mp4 don't exist! Renaming largest file {max_path.name}.")
        
        # create gripper calibration video if don't exist
        gripper_cal_dir = input_dir.joinpath('gripper_calibration')
        if not gripper_cal_dir.is_dir():
            gripper_cal_dir.mkdir()
            print("raw_videos/gripper_calibration don't exist! Creating one with the first video of each camera serial.")
            
            serial_start_dict = dict() #记录某个相机序列号当前找到的最早时间
            serial_path_dict = dict() #记录该最早时间对应的视频路径

            # 将拍摄时间最早的视频认为是gripper calibration视频（除了mapping视频外最早的）
            # 这里需要注意的是，可能会有多个相机的标定视频，因此需要按相机序列号分组
            with ExifToolHelper() as et:
                for mp4_path in list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4')):
                    if mp4_path.name.startswith('map'):
                        continue
                    
                    start_date = mp4_get_start_datetime(str(mp4_path))  #从视频中读取拍摄开始时间
                    meta = list(et.get_metadata(str(mp4_path)))[0]
                    cam_serial = meta['QuickTime:CameraSerialNumber']  #取 QuickTime 元数据里的相机序列号
                    
                    if cam_serial in serial_start_dict:
                        if start_date < serial_start_dict[cam_serial]:
                            serial_start_dict[cam_serial] = start_date
                            serial_path_dict[cam_serial] = mp4_path
                    else:
                        serial_start_dict[cam_serial] = start_date
                        serial_path_dict[cam_serial] = mp4_path
            
            for serial, path in serial_path_dict.items():
                print(f"Selected {path.name} for camera serial {serial}")
                out_path = gripper_cal_dir.joinpath(path.name)
                shutil.move(path, out_path)

        # look for mp4 video in all subdirectories in input_dir
        input_mp4_paths = list(input_dir.glob('**/*.MP4')) + list(input_dir.glob('**/*.mp4'))
        print(f'Found {len(input_mp4_paths)} MP4 videos')

        with ExifToolHelper() as et:
            for mp4_path in input_mp4_paths:
                if mp4_path.is_symlink():
                    print(f"Skipping {mp4_path.name}, already moved.")
                    continue  #因为脚本处理完一个视频后，会把原位置变成一个软链接。所以下次再跑时，看到软链接就知道这个视频已经处理过了，直接跳过。
                

                start_date = mp4_get_start_datetime(str(mp4_path))
                meta = list(et.get_metadata(str(mp4_path)))[0]
                cam_serial = meta['QuickTime:CameraSerialNumber']
                out_dname = 'demo_' + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")

                # special folders
                if mp4_path.name.startswith('mapping'):
                    out_dname = "mapping"
                elif mp4_path.name.startswith('gripper_cal') or mp4_path.parent.name.startswith('gripper_cal'):
                    out_dname = "gripper_calibration_" + cam_serial + '_' + start_date.strftime(r"%Y.%m.%d_%H.%M.%S.%f")
                
                # create directory
                this_out_dir = output_dir.joinpath(out_dname)
                """
                假设：
                output_dir = /data/session1/demos
                out_dname = demo_123456_2026.03.30_10.20.30.123456
                那么这一行之后：
                this_out_dir = /data/session1/demos/demo_123456_2026.03.30_10.20.30.123456
                """

                this_out_dir.mkdir(parents=True, exist_ok=True)

                """"
                mkdir()：创建目录
                parents=True：如果上级目录不存在，也一起创建
                exist_ok=True：如果目录已经存在，也不要报错
                """
                
                # move videos
                vfname = 'raw_video.mp4'
                out_video_path = this_out_dir.joinpath(vfname)
                shutil.move(mp4_path, out_video_path)

                # create symlink back from original location
                # relative_to's walk_up argument is not avaliable until python 3.12
                dots = os.path.join(*['..'] * len(mp4_path.parent.relative_to(session).parts))
                rel_path = str(out_video_path.relative_to(session))
                symlink_path = os.path.join(dots, rel_path)                
                mp4_path.symlink_to(symlink_path)

# %%
"""
如果用户没有传参数,比如只是运行：python 00_process_videos.py
那么 len(sys.argv) == 1，它就主动显示帮助信息，相当于执行：python 00_process_videos.py --help
如果传了参数,就正常用 click 解析命令行并执行 main()
"""
if __name__ == '__main__':
    if len(sys.argv) == 1:
        main.main(['--help'])
    else:
        main()
