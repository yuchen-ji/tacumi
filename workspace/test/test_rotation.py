import scipy.spatial.transform as st
import numpy as np


# init = st.Rotation.from_euler('xyz', [10, 0, 0], degrees=True).as_matrix()

# rot = st.Rotation.from_euler('xyz', [0, 10, 0], degrees=True).as_matrix()

# left_rot = rot @ init
# print(left_rot)

# right_rot = init @ rot
# print(right_rot)

# 旋转矩阵的左乘和右乘，是完全不同的结果。
# 左乘对于固定坐标系，右乘对于随体坐标系。参考：https://www.singleye.net/2023/10/%E5%B7%A6%E4%B9%98/%E5%8F%B3%E4%B9%98%E6%97%8B%E8%BD%AC/
# 如果只涉及一个轴的旋转[注意是一个轴的旋转，而不是一次旋转]，那么结果是一样的。

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3, :3] = st.Rotation.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix()

tx_flangerot45_tip = np.identity(4)
tx_flangerot45_tip[:3, 3] = np.array([0, 0, 0.2045])

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_tip
print(tx_flange_tip)



tx_flange_tip = np.identity(4)
tx_flange_tip[:3, :3] = st.Rotation.from_euler('z', [np.pi/4]).as_matrix()
tx_flange_tip[:3, 3] = np.array([0, 0, 0.2045])
print(tx_flange_tip)
# tx_tip_flange = np.linalg.inv(tx_flange_tip)