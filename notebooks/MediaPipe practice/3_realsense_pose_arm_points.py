import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# MediaPipe Pose 초기화 (신체 33개 랜드마크를 추적가능하다)
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

window_name = "RealSense Pose Arm Points"

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # Pose 추출
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            h, w, _ = color_image.shape
            landmarks = results.pose_landmarks.landmark

            # 오른쪽 어깨, 팔꿈치, 손목
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

            # 점 찍기
            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)

            # 선 연결
            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            # 라벨 표시
            cv2.putText(color_image, "Shoulder", (shoulder_x + 10, shoulder_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, "Elbow", (elbow_x + 10, elbow_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, "Wrist", (wrist_x + 10, wrist_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()