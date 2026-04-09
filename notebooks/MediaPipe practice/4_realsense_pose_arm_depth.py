import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# Depth 안정화 함수
def get_stable_depth(depth_frame, x, y, window_size=5):
    depths = []
    half = window_size // 2

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px = x + dx
            py = y + dy

            if px < 0 or py < 0 or px >= width or py >= height: # px, py가 화면 밖을 넘어가지 않도록.
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

window_name = "RealSense Pose Arm Depth"

try: #try 뜻=아래 코드를 실행하다 문제생겨도 마지막에 정리(cleanup)는 꼭 한다. 끝에 "finally"로 정리.
    while True: # 계속 반복해라.초당 카메라 영상이 30프레임이 들어오니깐..
        frames = pipeline.wait_for_frames() # 카메라에서 다음 프레임(데이터 묶음)이 올 때까지 기다렸다가 가져와라
        color_frame = frames.get_color_frame() # 하나의 frame 내에서 RGB 영상을 가져와라
        depth_frame = frames.get_depth_frame() # 하나의 frame 내에서 거리정보를 가져와라

        if not color_frame or not depth_frame:
            continue # 둘 중 하나라도 없으면 이 프레임은 버리고 다음으로 넘어가라

        color_image = np.asanyarray(color_frame.get_data()) # 카메라 데이터를 우리가 계산할 수 있는 형태 (NumPy 배열)로 바꾼다.
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) # BGR -> RGB 변환 (openCV는 기본이 BGR인데, MediaPipe는 입력으로 RGB 요구)

        # Pose 추출
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            h, w, _ = color_image.shape
            landmarks = results.pose_landmarks.landmark

            # 오른팔 기준
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # 픽셀 좌표 변환
            shoulder_x = int(shoulder.x * w)
            shoulder_y = int(shoulder.y * h)

            elbow_x = int(elbow.x * w)
            elbow_y = int(elbow.y * h)

            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)

            # 안정화된 depth 읽기
            shoulder_z = get_stable_depth(depth_frame, shoulder_x, shoulder_y, window_size=5)
            elbow_z = get_stable_depth(depth_frame, elbow_x, elbow_y, window_size=5)
            wrist_z = get_stable_depth(depth_frame, wrist_x, wrist_y, window_size=5)

            # 점 그리기
            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)

            # 선 연결
            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            # 라벨
            cv2.putText(color_image, "Shoulder", (shoulder_x + 10, shoulder_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, "Elbow", (elbow_x + 10, elbow_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, "Wrist", (wrist_x + 10, wrist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 좌표 + depth 텍스트
            text1 = f"Shoulder: ({shoulder_x}, {shoulder_y}), z={shoulder_z:.3f} m"
            text2 = f"Elbow:    ({elbow_x}, {elbow_y}), z={elbow_z:.3f} m"
            text3 = f"Wrist:    ({wrist_x}, {wrist_y}), z={wrist_z:.3f} m"

            cv2.putText(color_image, text1, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, text2, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, text3, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally: # try와 연결해서, 카메라 & 창을 깨끗하게 정리하겠다.
    pipeline.stop()
    cv2.destroyAllWindows()