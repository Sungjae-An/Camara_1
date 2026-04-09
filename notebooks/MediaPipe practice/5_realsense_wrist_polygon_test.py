import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# 위험영역 점 저장
clicked_points = []


def mouse_callback(event, x, y, flags, param): # “마우스 이벤트가 발생할 때마다 실행되는 함수
    global clicked_points # global 선언

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y)) # append -> 추가
        print(f"Added point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points: # 리스트가 비어있지 않으면
            removed = clicked_points.pop() # pop -> 리스트의 마지막 요소 제거
            print(f"Removed point: {removed}")


def get_stable_depth(depth_frame, x, y, window_size=5):
    depths = []
    half = window_size // 2

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px = x + dx
            py = y + dy

            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0:
                depths.append(d)

    if len(depths) == 0:
        return 0.0

    return float(np.median(depths))


# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

window_name = "RealSense Wrist Polygon Test"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback) # 마우스 클릭 -> openCV -> mouse_callback 자동실행

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        h, w, _ = color_image.shape

        # -------------------------
        # 1) 위험영역 다각형 시각화
        # -------------------------
        if len(clicked_points) >= 3:
            overlay = color_image.copy() # 원본화면알 복사해서 복사본에만 칠하겠다. 원본에 칠하면 너무 진해서 보기불편하므로 원본 + 색칠된 복사본을 섞는다.

            polygon = np.array(clicked_points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 255, 255)) # fillPoly는 [polygon1, polygon2, polygo3] 구조인데, 다각형 1개만 쓰니깐 [polygon]

            alpha = 0.25
            color_image = cv2.addWeighted(overlay, alpha, color_image, 1 - alpha, 0) # alpha 0.25로 원본 75%, overlay 25%로 섞어서 반투명하게.

        # 점 표시
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)

            label = f"{i}"
            cv2.putText(color_image, label, (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 선 연결
        for i in range(len(clicked_points) - 1):
            x1, y1 = clicked_points[i]
            x2, y2 = clicked_points[i + 1]
            cv2.line(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 마지막 점과 첫 점 연결
        if len(clicked_points) >= 3:
            x1, y1 = clicked_points[-1]
            x2, y2 = clicked_points[0]
            cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # -------------------------
        # 2) Pose로 손목 추적
        # -------------------------
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 오른팔 기준
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            shoulder_x = max(0, min(w - 1, int(shoulder.x * w)))
            shoulder_y = max(0, min(h - 1, int(shoulder.y * h)))

            elbow_x = max(0, min(w - 1, int(elbow.x * w)))
            elbow_y = max(0, min(h - 1, int(elbow.y * h)))

            wrist_x = max(0, min(w - 1, int(wrist.x * w)))
            wrist_y = max(0, min(h - 1, int(wrist.y * h)))

            # depth
            shoulder_z = get_stable_depth(depth_frame, shoulder_x, shoulder_y, window_size=5)
            elbow_z = get_stable_depth(depth_frame, elbow_x, elbow_y, window_size=5)
            wrist_z = get_stable_depth(depth_frame, wrist_x, wrist_y, window_size=5)

            # 상지 표시
            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)

            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            cv2.putText(color_image, "Shoulder", (shoulder_x + 10, shoulder_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, "Elbow", (elbow_x + 10, elbow_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, "Wrist", (wrist_x + 10, wrist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 관절 정보 표시
            text1 = f"Shoulder z={shoulder_z:.3f} m"
            text2 = f"Elbow z={elbow_z:.3f} m"
            text3 = f"Wrist z={wrist_z:.3f} m"

            cv2.putText(color_image, text1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, text2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, text3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # -------------------------
            # 3) 손목이 위험영역 안에 있는지 판정
            # -------------------------
            if len(clicked_points) >= 3:
                polygon = np.array(clicked_points, dtype=np.int32) # python list로 저장된 clicked points를 OpenCV가 이해할 수 있는 NumPy배열로 변환한다.
                result = cv2.pointPolygonTest(polygon, (wrist_x, wrist_y), False) # 핵심! cv2.pointPolygonTest함수는 점 하나가 다각형에 대해 안쪽이면 >0, 경계면 =0, 바깥이면 <0

                if result > 0:
                    status_text = "WRIST: INSIDE DANGER ZONE"
                    status_color = (0, 0, 255)
                elif result == 0:
                    status_text = "WRIST: ON EDGE"
                    status_color = (0, 255, 255)
                else:
                    status_text = "WRIST: SAFE"
                    status_color = (0, 255, 0)

                cv2.putText(color_image, status_text, (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # 안내 문구
        cv2.putText(color_image,
                    "Left:add zone | Right:remove last | c:clear | q:quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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