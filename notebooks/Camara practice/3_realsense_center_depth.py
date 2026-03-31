import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 파이프라인 생성
pipeline = rs.pipeline()
config = rs.config()

# 2. 컬러 스트림과 깊이 스트림 설정 (초당 30장)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 3. 스트림 시작
pipeline.start(config)

try:
    while True:
        # 4. 프레임 받기 (color & depth frame 하나씩)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # 프레임이 없으면 다음 반복
        if not color_frame or not depth_frame:
            continue

        # 5. numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 6. 화면 중앙 좌표 계산
        height, width, _ = color_image.shape
        center_x = width // 2
        center_y = height // 2

        # 7. 중앙 좌표의 거리값 읽기 (단위: meter)
        distance = depth_frame.get_distance(center_x, center_y)

        # 8. 십자선 그리기
        cv2.line(color_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(color_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

        # 9. 거리 표시
        text = f"Distance at center: {distance:.3f} m"
        cv2.putText(color_image, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 10. 컬러 화면 출력
        cv2.imshow("RealSense Color + Center Depth", color_image)

        # q 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()