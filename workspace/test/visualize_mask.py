# this file is to verify the slam_mask.png correctness
# even if the slam_mask.png is strange, it shows the correct area when applied to the video


import cv2


mask_path = "my_demo_session/demos/demo_C3441327029233_2025.08.20_18.44.21.394000/slam_mask.png"
vid_path = "my_demo_session/demos/demo_C3441327029233_2025.08.20_18.44.21.394000/raw_video.mp4"

mask_img = cv2.imread(mask_path)
vid_cap = cv2.VideoCapture(vid_path)

while True:
    ret, vid_frame = vid_cap.read()
    if not ret:
        break

    combined_frame = cv2.bitwise_and(vid_frame, ~mask_img)
    # combined_frame = mask_img
    # 设置窗口大小
    cv2.namedWindow("Masked Video", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Masked Video", 800, 600)
    cv2.imshow("Masked Video", combined_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

vid_cap.release()
cv2.destroyAllWindows()