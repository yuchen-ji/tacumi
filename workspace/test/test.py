import pathlib
import os
import pickle
import collections
import numpy as np
import time

# # 测试文件的挂载位置
# input_dir = "example_demo_session/demos/mapping"
# video_dir = pathlib.Path(os.path.expanduser(input_dir)).absolute()
# map_path = video_dir.joinpath('map_atlas.osa')
# map_mount_source = pathlib.Path(map_path)
# map_mount_target = pathlib.Path('/map').joinpath(map_mount_source.name)
# print(map_mount_target.parent)  # /map
# print(map_mount_target.name)  # map_atlas.osa


# 读取pkl文件
# with open('example_demo_session/demos/mapping/tag_detection.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data)

# gripper_ids = [1, 1, 1]
# counter = collections.Counter(gripper_ids)
# print(len(counter))

# vid_idx_cam_idx_map = np.full(10, fill_value=-1, dtype=np.int32)
# print(vid_idx_cam_idx_map)


# import scipy.interpolate as si

# t = np.array([0.0, 1.0, 2.0])
# x = np.array([
#     [0,0,0],
#     [1,0,0],
#     [2,0,0],
# ])

# gripper_interp = si.interp1d(
#     t, x,
#     axis=0, bounds_error=False,
#     fill_value=(x[0], x[-1]))

# print(gripper_interp(3))

# forward = np.array([1,0,0])
# up = np.array([0,0,1])
# right = np.cross(forward, up)
# print(right)

# dt = 1/30
# diff = 10 % dt
# print(diff)


# # 获取当前时间戳
# start_timestamp = time.time()
# print(start_timestamp)
# end_timestamp = time.time()
# print(end_timestamp)


# """
# np.all的用法
# """

# a = np.array([
#     [1, 2, 3],
#     [0, -2, -3]
# ])
# # 输出的结果和第0维一样
# b = np.all(a, axis=0)   # [False，True, True]
# print(b)
# print(a[0])


# """
# np.nonzero的用法
# """
# a = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
# b = np.nonzero(a)
# print(b)
# print(b[0])
# print(b[0][-1])


# """
# scipy.interpolate.Interp1d的用法
# """
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate

# x = np.arange(0, 10)
# y = np.exp(-x / 3.0)
# f = interpolate.interp1d(x, y, axis=0, bounds_error=False, fill_value=(x[0], x[-1]))

# xnew = np.arange(0, 18.3, 0.1)
# ynew = f(xnew)  # use interpolation function returned by `interp1d`
# plt.plot(x, y, "o", xnew, ynew, "-")
# plt.show()
