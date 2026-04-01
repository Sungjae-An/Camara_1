import pyrealsense2 as rs
import numpy as np
import cv2

clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            removed = clicked_points.pop()
            print(f"Removed point: {removed}")


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

window_name = "RealSense Polygon Test"
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

        # 중앙점
        height, width, _ = color_image.shape
        center_x = width // 2
        center_y = height // 2

        # 다각형 내부 채우기
        if len(clicked_points) >= 3:
            overlay = color_image.copy() # 현 화면 복사
            polygon = np.array(clicked_points, dtype=np.int32) # 클릭한 점 리스트를 Numpy 배열(array)로 변환
            cv2.fillPoly(overlay, [polygon], (0, 255, 255)) # overlay 위에 polygon 모양으로 노란색으로 채우기
            alpha = 0.25 # 투명도 (색칠한 쪽이 25%, 원본이 75%)
            color_image = cv2.addWeighted(overlay, alpha, color_image, 1 - alpha, 0) # overlay와 원본을 섞어서 반투명하게 함

        # 점 표시
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

            label = f"{i}"
            cv2.putText(color_image, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            distance = depth_frame.get_distance(x, y)
            dist_text = f"{distance:.2f}m"
            cv2.putText(color_image, dist_text, (x + 8, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 선 연결
        for i in range(len(clicked_points) - 1):
            x1, y1 = clicked_points[i]
            x2, y2 = clicked_points[i + 1]
            cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 닫기
        if len(clicked_points) >= 3:
            x1, y1 = clicked_points[-1]
            x2, y2 = clicked_points[0]
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 중앙점 표시
        cv2.circle(color_image, (center_x, center_y), 6, (255, 0, 255), -1)

        # 중앙점 안/밖 판정
        if len(clicked_points) >= 3:
            polygon = np.array(clicked_points, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, (center_x, center_y), False)

            if result > 0:
                status_text = "CENTER: INSIDE"
                status_color = (0, 255, 0)
            elif result == 0:
                status_text = "CENTER: ON EDGE"
                status_color = (0, 255, 255)
            else:
                status_text = "CENTER: OUTSIDE"
                status_color = (0, 0, 255)

            cv2.putText(color_image, status_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.putText(color_image,
                    "Left: add | Right: remove last | c: clear | q: quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            clicked_points.clear()
            print("All points cleared")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()