import pyrealsense2 as rs
import numpy as np
import cv2

# -------------------------
# 마우스로 클릭한 좌표를 저장할 변수
# -------------------------
clicked_x = None # 아직 클릭안하면 값이 없음(None)
clicked_y = None

# -------------------------
# 마우스 클릭 이벤트 함수
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 좌측버튼 클릭시만 실행
        clicked_x = x
        clicked_y = y
        print(f"Clicked at: ({clicked_x}, {clicked_y})")

# -------------------------
# RealSense 설정
# -------------------------
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

# 윈도우 만들기 + 마우스 이벤트 연결
window_name = "RealSense Click Depth"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 컬러 프레임을 numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 클릭한 점이 있으면 표시
        if clicked_x is not None and clicked_y is not None:
            # 클릭 위치에 빨간 점 그리기
            cv2.circle(color_image, (clicked_x, clicked_y), 5, (0, 0, 255), -1)

            # 해당 위치의 거리 읽기
            distance = depth_frame.get_distance(clicked_x, clicked_y)

            # 글자 표시
            text1 = f"Clicked: ({clicked_x}, {clicked_y})"
            text2 = f"Distance: {distance:.3f} m"

            cv2.putText(color_image, text1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(color_image, text2, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow(window_name, color_image)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()