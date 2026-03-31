import pyrealsense2 as rs
import numpy as np
import cv2

# -------------------------
# 클릭한 여러 점을 저장할 리스트
# -------------------------
clicked_points = []

# -------------------------
# 마우스 클릭 이벤트 함수
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_points

    # 왼쪽 클릭: 점 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Added point: ({x}, {y})")

    # 오른쪽 클릭: 마지막 점 삭제
    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            removed = clicked_points.pop()
            print(f"Removed point: {removed}")

# -------------------------
# RealSense 설정
# -------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

window_name = "RealSense Multi Click"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # 저장된 모든 점 표시
        for i, (x, y) in enumerate(clicked_points):
            # 점 그리기
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

            # 점 번호 표시
            label = f"{i}"
            cv2.putText(color_image, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 거리 읽기
            distance = depth_frame.get_distance(x, y)

            # 거리도 같이 표시
            dist_text = f"{distance:.2f}m"
            cv2.putText(color_image, dist_text, (x + 8, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 전체 안내 문구
        cv2.putText(color_image, "Left click: add point | Right click: remove last | q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF # 어떤키 눌렀나 저장

        if key == ord('q'):
            break
        elif key == ord('c'):
            clicked_points.clear()
            print("All points cleared")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()