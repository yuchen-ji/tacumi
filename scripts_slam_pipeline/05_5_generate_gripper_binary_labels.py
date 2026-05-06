
"""
python scripts_slam_pipeline/05_5_generate_gripper_binary_labels.py session_dir
"""



"""
第一步：从视觉检测结果恢复连续宽度信号

tag_detection.pkl -> widths

第二步：把宽度信号变得更稳定

宽度 -> 补 NaN -> 中值滤波 -> 均值平滑

第三步：根据一阶差分判断趋势
宽度增大：open
宽度减小：close
宽度基本不变：根据平台位置判断，或者继承前一状态


第四步：对标签做后处理

去掉很短的错误片段，得到更平滑的标签序列

"""
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pickle
import pathlib
import click
import numpy as np
from tqdm import tqdm

from umi.common.cv_util import get_gripper_width


# %%
def moving_average(x, k=5):
    x = np.asarray(x, dtype=np.float64)
    if k <= 1:
        return x.copy()
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(x_pad, kernel, mode='valid')

def median_filter_1d(x, k=5):
    x = np.asarray(x, dtype=np.float64)
    if k <= 1:
        return x.copy()
    pad = k // 2
    x_pad = np.pad(x, (pad, pad), mode='edge')
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x_pad[i:i+k])
    return out


def fill_nan_by_interp(x):
    """
    对 NaN 做线性插值；若全是 NaN，则返回原数组。
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    n = len(out)
    valid = np.where(~np.isnan(out))[0]
    if len(valid) == 0:
        return out
    if len(valid) == 1:
        out[:] = out[valid[0]]
        return out

    all_idx = np.arange(n)
    out[np.isnan(out)] = np.interp(
        all_idx[np.isnan(out)],
        all_idx[~np.isnan(out)],
        out[~np.isnan(out)]
    )
    return out


def remove_short_segments(labels, min_len=3):
    """
    去毛刺：把长度小于 min_len 的短片段并入前后相邻主段。
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(labels)
    if n == 0:
        return labels

    start = 0
    while start < n:
        end = start + 1
        while end < n and labels[end] == labels[start]:
            end += 1

        seg_len = end - start
        if seg_len < min_len:
            left_label = labels[start - 1] if start > 0 else None
            right_label = labels[end] if end < n else None

            if left_label is None and right_label is None:
                pass
            elif left_label is None:
                labels[start:end] = right_label
            elif right_label is None:
                labels[start:end] = left_label
            else:
                if left_label == right_label:
                    labels[start:end] = left_label
                else:
                    left_len = 0
                    i = start - 1
                    while i >= 0 and labels[i] == left_label:
                        left_len += 1
                        i -= 1

                    right_len = 0
                    i = end
                    while i < n and labels[i] == right_label:
                        right_len += 1
                        i += 1

                    labels[start:end] = left_label if left_len >= right_len else right_label

        start = end

    return labels


def normalize_widths(widths, min_width, max_width):
    widths = np.asarray(widths, dtype=np.float64)
    denom = max(max_width - min_width, 1e-8)
    w_norm = (widths - min_width) / denom
    return np.clip(w_norm, 0.0, 1.0)


def label_by_trend(
    widths_norm,
    smooth_k=7,
    diff_eps=0.006,
    stable_band=0.15,
    min_segment_len=5,
    default_open=True
):
    
    """
    输入:
        widths_norm: 归一化后的夹爪宽度序列

    返回:
        labels: 最终二值标签
        x_filled: 对 NaN 线性插值后的序列
        x_med: 中值滤波后的序列
        x_smooth: 均值平滑后的序列
 
    """

    """
    规则：
    - 持续变大段 -> 1 (open)
    - 持续变小段 -> -1 (close)
    - 稳定段：
        - 接近大宽度平台 -> 1
        - 接近小宽度平台 -> -1
        - 中间模糊区域 -> 继承上一帧状态
    """
    x = np.asarray(widths_norm, dtype=np.float64)

    # 先插值补 NaN，再平滑
    # x_filled = fill_nan_by_interp(x)

    # if np.all(np.isnan(x)):
    #     init = 1 if default_open else -1
    #     return np.full(len(x), init, dtype=np.int64), x, x_filled

    #x_smooth = moving_average(x_filled, k=smooth_k)


    x_filled = fill_nan_by_interp(x)
    if np.all(np.isnan(x)):
        init = 1 if default_open else -1
    #    return np.full(len(x), init, dtype=np.int64), x, x_filled
        labels = np.full(len(x), init, dtype=np.int64)
        return labels, x_filled, x_filled.copy(), x_filled.copy()


# 先中值滤波去掉孤立尖刺，再均值平滑
    x_med = median_filter_1d(x_filled, k=5)

    x_smooth = moving_average(x_med, k=smooth_k)

    n = len(x_smooth)
    labels = np.ones(n, dtype=np.int64)

    # 初始状态
    if default_open:
        labels[0] = 1 if x_smooth[0] >= 0.5 else -1
    else:
        labels[0] = -1 if x_smooth[0] < 0.5 else 1

    high_th = 1.0 - stable_band
    low_th = stable_band

    for t in range(1, n):
        d = x_smooth[t] - x_smooth[t - 1]

        if d > diff_eps:
            labels[t] = 1          # open
        elif d < -diff_eps:
            labels[t] = -1          # close
        else:
            if x_smooth[t] >= high_th:
                labels[t] = 1      # 稳定大宽度 -> open
            elif x_smooth[t] <= low_th:
                labels[t] = -1      # 稳定小宽度 -> close
            else:
                labels[t] = labels[t - 1]

    labels = remove_short_segments(labels, min_len=min_segment_len)
    return labels, x_smooth, x_med, x_smooth


def compute_widths_from_tag_detection(
    tag_detection_results,
    left_id,
    right_id,
    nominal_z=0.072
):
    widths = []
    for dt in tag_detection_results:
        tag_dict = dt['tag_dict']
        width = get_gripper_width(tag_dict, left_id, right_id, nominal_z=nominal_z)
        if width is None:
            width = np.nan
        widths.append(width)
    return np.asarray(widths, dtype=np.float64)


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# %%
@click.command()
@click.argument('session_dir', type=click.Path(exists=True))
@click.option(
    '--nominal_z',
    type=float,
    default=0.072,
    help='nominal Z value for gripper finger tag'
)
@click.option(
    '--smooth_k',
    type=int,
    default=7,
    help='moving average kernel size'
)
@click.option(
    '--diff_eps',
    type=float,
    default=0.006,
    help='difference threshold on normalized width for trend detection'
)
@click.option(
    '--stable_band',
    type=float,
    default=0.15,
    help='stable small/large platform band on normalized width'
)
@click.option(
    '--min_segment_len',
    type=int,
    default=5,
    help='minimum segment length for debouncing'
)
@click.option(
    '--calib_dir_name',
    type=str,
    default='gripper_calibration',
    help='directory name prefix for gripper calibration folders'
)
@click.option(
    '--range_json_name',
    type=str,
    default='gripper_range.json',
    help='range json filename'
)
@click.option(
    '--overwrite',
    is_flag=True,
    default=False,
    help='overwrite existing outputs'
)
def main(
    session_dir,
    nominal_z,
    smooth_k,
    diff_eps,
    stable_band,
    min_segment_len,
    calib_dir_name,
    range_json_name,
    overwrite
):
    session_dir = pathlib.Path(session_dir).expanduser().resolve()
    demos_dir = session_dir / 'demos'
    assert demos_dir.is_dir(), f'demos directory not found: {demos_dir}'

    # 1) 找 calibration 结果
    calib_json_candidates = []
    for p in demos_dir.glob(f'{calib_dir_name}*/{range_json_name}'):
        if p.is_file():
            calib_json_candidates.append(p)

    if len(calib_json_candidates) == 0:
        raise FileNotFoundError(
            f'No {range_json_name} found under {demos_dir}/{calib_dir_name}*'
        )

    calib_json_path = sorted(calib_json_candidates)[0]
    print(f'Using calibration file: {calib_json_path}')

    with open(calib_json_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)

    left_id = calib['left_finger_tag_id']
    right_id = calib['right_finger_tag_id']
    min_width = float(calib['min_width'])
    max_width = float(calib['max_width'])

    if not np.isfinite(min_width) or not np.isfinite(max_width) or max_width <= min_width:
        raise ValueError(
            f'Invalid gripper range: min_width={min_width}, max_width={max_width}'
        )

    print(
        f'left_id={left_id}, right_id={right_id}, '
        f'min_width={min_width:.6f}, max_width={max_width:.6f}'
    )

    # 2) 找所有 demo：包含 raw_video.mp4 且非 mapping / gripper_calibration*
    demo_dirs = []
    for p in demos_dir.glob('*/raw_video.mp4'):
        demo_dir = p.parent
        name = demo_dir.name
        if name == 'mapping':
            continue
        if name.startswith(calib_dir_name):
            continue
        demo_dirs.append(demo_dir)

    demo_dirs = sorted(demo_dirs)
    print(f'Found {len(demo_dirs)} demo dirs')

    if len(demo_dirs) == 0:
        print('No demo directories found.')
        return

    # 3) 逐个 demo 处理
    n_success = 0
    n_skip = 0
    n_fail = 0

    for demo_dir in tqdm(demo_dirs, desc='Processing demos'):
        try:
            tag_pkl = demo_dir / 'tag_detection.pkl'
            if not tag_pkl.is_file():
                print(f'[WARN] missing tag_detection.pkl, skip: {demo_dir}')
                n_skip += 1
                continue

            widths_json_path = demo_dir / 'gripper_widths.json'
            labels_npy_path = demo_dir / 'gripper_binary_labels.npy'
            labels_json_path = demo_dir / 'gripper_binary_labels.json'

            if (
                (not overwrite)
                and widths_json_path.exists()
                and labels_npy_path.exists()
                and labels_json_path.exists()
            ):
                print(f'[INFO] outputs already exist, skip: {demo_dir.name}')
                n_skip += 1
                continue

            with open(tag_pkl, 'rb') as f:
                tag_detection_results = pickle.load(f)

            widths = compute_widths_from_tag_detection(
                tag_detection_results=tag_detection_results,
                left_id=left_id,
                right_id=right_id,
                nominal_z=nominal_z
            )

            widths_norm = normalize_widths(
                widths,
                min_width=min_width,
                max_width=max_width
            )

            labels, widths_filled, widths_median, widths_smooth = label_by_trend(
                widths_norm,
                smooth_k=smooth_k,
                diff_eps=diff_eps,
                stable_band=stable_band,
                min_segment_len=min_segment_len,
                default_open=True
            )

            # save_json(widths_json_path, {
            #     'left_finger_tag_id': left_id,
            #     'right_finger_tag_id': right_id,
            #     'min_width': min_width,
            #     'max_width': max_width,
            #     'nominal_z': nominal_z,
            #     'n_frames': int(len(widths)),
            #     'widths_raw': [None if np.isnan(v) else float(v) for v in widths],
            #     'widths_norm': [None if np.isnan(v) else float(v) for v in widths_norm],
            #     'widths_filled': [float(v) for v in widths_filled],
            #     'widths_smooth': [float(v) for v in widths_smooth]
            # })
            save_json(widths_json_path, {
                'left_finger_tag_id': left_id,
                'right_finger_tag_id': right_id,
                'min_width': min_width,
                'max_width': max_width,
                'nominal_z': nominal_z,
                'n_frames': int(len(widths)),
                'widths_raw': [None if np.isnan(v) else float(v) for v in widths],
                'widths_norm': [None if np.isnan(v) else float(v) for v in widths_norm],
                'widths_filled': [float(v) for v in widths_filled],
                'widths_median': [float(v) for v in widths_median],
                'widths_smooth': [float(v) for v in widths_smooth]
                })

            np.save(labels_npy_path, labels.astype(np.int64))

            save_json(labels_json_path, {
                'label_definition': {
                    '-1': 'close',
                    '1': 'open'
                },
                'rule': {
                    'decreasing_trend': -1,
                    'stable_small_width': -1,
                    'increasing_trend': 1,
                    'stable_large_width': 1
                },
                'params': {
                    'smooth_k': smooth_k,
                    'diff_eps': diff_eps,
                    'stable_band': stable_band,
                    'min_segment_len': min_segment_len
                },
                'n_frames': int(len(labels)),
                'labels': labels.astype(int).tolist()
            })

            n_success += 1

        except Exception as e:
            print(f'[ERROR] failed on {demo_dir}: {e}')
            n_fail += 1

    print('\n Done!')
    print(f'success: {n_success}')
    print(f'skip:    {n_skip}')
    print(f'fail:    {n_fail}')


# %%
if __name__ == '__main__':
    main()