import rtde_control
import rtde_receive
import time


def round_list(vals, digits=3):
    return [round(v, digits) for v in vals]


# 连接到机器人，获取控制和接收接口
robot_ip = "192.168.12.21"
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

# 获取当前机器人的关键角度
# joint_positions = rtde_r.getActualQ()
# print(f"Current joint positions: {round_list(joint_positions)}")

# 获取当前机器人的TCP位置
# rtde_c.setTcp([0.0, 0.0, 0.15, 0.0, 0.0, 0.0])
# rtde_c.setPayload(0.0, [0.0, 0.0, 0.0])
# tool_frame = rtde_c.getTCPOffset()
# print(f"Current tool frame offset: {round_list(tool_frame)}")

tcp_pose = rtde_r.getActualTCPPose()
print(f"Current TCP pose: {round_list(tcp_pose)}")

# tcp_force = rtde_r.getActualTCPForce()
# print(f"Current TCP force: {round_list(tcp_force)}")

# 这个需要配备力传感器才有效
# tcp_force2 = rtde_r.getFtRawWrench()
# print(f"Current raw wrench: {round_list(tcp_force2)}")

# 移动机器人到一个新的关节位置
# target_q = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
# rtde_c.moveJ(target_q)

# 移动机器人到一个新的TCP位置 （上升0.1m）
tcp_pose[2] += 0.05  # Move up by 0.1 meters
rtde_c.moveL(tcp_pose, 0.25, 0.5, True)
time.sleep(0.2)
# rtde_c.stopL(0.5)

# 断开连接
rtde_c.disconnect()
rtde_r.disconnect()