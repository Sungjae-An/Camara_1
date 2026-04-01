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

window_name = "RealSense Closed Polygon"
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

        # 점 표시
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

            label = f"{i}"
            cv2.putText(
                color_image,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

            distance = depth_frame.get_distance(x, y)
            dist_text = f"{distance:.2f}m"
            cv2.putText(
                color_image,
                dist_text,
                (x + 8, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1
            )

        # 순서대로 선 연결
        for i in range(len(clicked_points) - 1):
            x1, y1 = clicked_points[i]
            x2, y2 = clicked_points[i + 1]
            cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 마지막 점과 첫 점 연결 -> 닫힌 다각형
        if len(clicked_points) >= 3:
            x1, y1 = clicked_points[-1]
            x2, y2 = clicked_points[0]
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 안내 문구
        cv2.putText(
            color_image,
            "Left: add | Right: remove last | c: clear | q: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

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