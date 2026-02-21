# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import click
from tqdm import tqdm
import yaml
import json
import av
import numpy as np
import cv2
import pickle

from umi.common.cv_util import (
    parse_aruco_config, 
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags,
    draw_predefined_mask
)

# %%
# @click.command()
# @click.option('-i', '--input', required=True)
# @click.option('-o', '--output', required=True)
# @click.option('-ij', '--intrinsics_json', required=True)
# @click.option('-ay', '--aruco_yaml', required=True)
# @click.option('-n', '--num_workers', type=int, default=4)
def main(input, output, intrinsics_json, aruco_yaml, num_workers):
    cv2.setNumThreads(num_workers)

    # load aruco config
    aruco_config = parse_aruco_config(yaml.safe_load(open(aruco_yaml, 'r')))
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']

    # load intrinsics
    raw_fisheye_intr = parse_fisheye_intrinsics(json.load(open(intrinsics_json, 'r')))

    results = list()
    with av.open(os.path.expanduser(input)) as in_container:
        in_stream = in_container.streams.video[0]
        in_stream.thread_type = "AUTO"
        in_stream.thread_count = num_workers

        in_res = np.array([in_stream.height, in_stream.width])[::-1]
        fisheye_intr = convert_fisheye_intrinsics_resolution(
            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)

        for i, frame in tqdm(enumerate(in_container.decode(in_stream)), total=in_stream.frames):
            img = frame.to_ndarray(format='rgb24')
            frame_cts_sec = frame.pts * in_stream.time_base
            # avoid detecting tags in the mirror
            # 对于我们的版本，没有mirror。这里的代码保持不变，draw_predefined_mask函数实现里的mirror不进行任何处理
            img = draw_predefined_mask(img, color=(0,0,0), mirror=True, gripper=False, finger=False)
            tag_dict = detect_localize_aruco_tags(
                img=img,
                aruco_dict=aruco_dict,
                marker_size_map=marker_size_map,
                fisheye_intr_dict=fisheye_intr,
                refine_subpix=True
            )
            result = {
                'frame_idx': i,
                'time': float(frame_cts_sec),
                'tag_dict': tag_dict
            }
            results.append(result)
    
    # dump
    pickle.dump(results, open(os.path.expanduser(output), 'wb'))

# %%
if __name__ == "__main__":
    # main()

    # Example usage:
    # 使用前需要先注释掉main()函数中的@click.command()装饰器
    input = 'my_demo_session/demos/gripper_calibration_C3441327029233_2025.08.20_18.12.18.172700/raw_video.mp4'
    output = 'my_demo_session/demos/gripper_calibration_C3441327029233_2025.08.20_18.12.18.172700/tag_detection.pkl'
    intrinsics_json = 'example/calibration/gopro_intrinsics_2_7k.json'
    aruco_yaml = 'example/calibration/aruco_config.yaml'
    num_workers = 4
    main(
        input=input,
        output=output,
        intrinsics_json=intrinsics_json,
        aruco_yaml=aruco_yaml,
        num_workers=num_workers
    )