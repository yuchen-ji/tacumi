## 启动服务端程序

### 1. 使用 polymetis 方式
```bash
# 启动机器人和franka hand
sudo pkill -9 -f run_server
python polymetis/polymetis/python/scripts/launch_robot.py robot_client=franka_hardware
python polymetis/polymetis/python/scripts/launch_gripper.py gripper=franka_hand

# 启动服务端rpc程序
python workspace/launch_franka_interface_server_fairo.py
```