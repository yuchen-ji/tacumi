import numpy as np
import serial
import threading
import cv2
import time
from scipy.ndimage import gaussian_filter

# import matplotlib.pyplot as plt
# import seaborn as sns
# os.system('cls')

# contact_data_norm = np.zeros((8, 16))
contact_data_norm = np.zeros((16, 16))
WINDOW_WIDTH = contact_data_norm.shape[1] * 30
WINDOW_HEIGHT = contact_data_norm.shape[0] * 30
cv2.namedWindow("Contact Data_left", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Contact Data_left", WINDOW_WIDTH, WINDOW_HEIGHT)
THRESHOLD = 4
NOISE_SCALE = 60
flag = False  # 确保在主线程第一次访问前已定义，避免 NameError


def readThread(serDev):
    global contact_data_norm, flag
    data_tac = []
    num = 0
    t1 = 0
    backup = None
    flag = False    # 用于判断初始化是否完成
    current = None
    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode("utf-8").strip()
            except:
                line = ""
            # 当读完一个frame，即16个matrix_row之后，line=""，作为分隔符，表示读完了整板的数据
            if len(line) < 10:
                if current is not None and len(current) == 16:  # len(current) == 16 表示读取了16个matrix_row
                    backup = np.array(current)
                    print("fps", 1 / (time.time() - t1))
                    t1 = time.time()
                    data_tac.append(backup)
                    num += 1
                    if num > 30:
                        break
                current = []
                continue
            if current is not None: # current=[]也不等于None
                str_values = line.split()
                int_values = [int(val) for val in str_values]
                matrix_row = int_values
                current.append(matrix_row)

    data_tac = np.array(data_tac)
    # median是基线，后续的每一帧都要减去这个基线
    median = np.median(data_tac, axis=0)
    print("median", median)
    flag = True
    print("Finish Initialization!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    while True:
        if serDev.in_waiting > 0:
            try:
                line = serDev.readline().decode("utf-8").strip()
                # print("fps",1/(time.time()-t1))
                # t1 =time.time()
            except:
                line = ""
            if len(line) < 10:
                if current is not None and len(current) == 16:
                    backup = np.array(current)
                    # print(backup)
                current = []
                if backup is not None:
                    # 当前的contact_data需要减去基线和阈值
                    contact_data = backup - median - THRESHOLD
                    print(contact_data)
                    contact_data = np.clip(contact_data, 0, 100)

                    # 如果小于阈值，则认为是噪声
                    if np.max(contact_data) < THRESHOLD:
                        contact_data_norm = contact_data / NOISE_SCALE
                    # 如果不是噪声，则进行归一化
                    else:
                        # contact_data_norm = np.log(contact_data + 1) / np.log(2.0)
                        contact_data_norm = contact_data / np.max(contact_data)

                continue

            if current is not None:
                str_values = line.split()
                int_values = [int(val) for val in str_values]
                matrix_row = int_values
                current.append(matrix_row)
                continue


# PORT = "left_gripper_right_finger"
PORT = "/dev/ttyUSB0"
BAUD = 2000000
# serDev = serial.Serial(PORT,2000000)
serDev = serial.Serial(PORT, BAUD)    # 获得串口的句柄
exitThread = False
serDev.flush()
serialThread = threading.Thread(target=readThread, args=(serDev,))
serialThread.daemon = True  # 当主进程退出时，强制结束线程
serialThread.start()


def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)


def temporal_filter(new_frame, prev_frame, alpha=0.5):  # 原先alpha=0.2
    """
    Apply temporal smoothing filter.
    'alpha' determines the blending factor.
    A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame


# Initialize previous frame buffer
prev_frame = np.zeros_like(contact_data_norm)


if __name__ == "__main__":
    print("receive data test")
    while True:
        for i in range(300):
            if flag:
                # contact_data_norm = apply_gaussian_blur(contact_data_norm, sigma=0.1)
                # temp_filtered_data = temporal_filter(contact_data_norm, prev_frame)
                temp_filtered_data = contact_data_norm
                prev_frame = temp_filtered_data

                # 这里只是将读取到的接触数据可视化（转化为colormap）
                # Scale to 0-255 and convert to uint8
                temp_filtered_data_scaled = (temp_filtered_data * 255).astype(np.uint8)
                # Apply color map
                colormap = cv2.applyColorMap(
                    temp_filtered_data_scaled, cv2.COLORMAP_VIRIDIS
                )

                cv2.imshow("Contact Data_left", colormap)
                cv2.waitKey(1)
            time.sleep(0.01)
