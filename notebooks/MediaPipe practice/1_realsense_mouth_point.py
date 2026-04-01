import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

window_name = "RealSense Mouth Depth"

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = color_image.shape

                # 윗입술 / 아랫입술
                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]

                # 픽셀 좌표 변환
                upper_x = int(upper_lip.x * w)
                upper_y = int(upper_lip.y * h)

                lower_x = int(lower_lip.x * w)
                lower_y = int(lower_lip.y * h)

                # 입 중심점
                mouth_x = (upper_x + lower_x) // 2
                mouth_y = (upper_y + lower_y) // 2

                # 입 점의 거리값 읽기
                mouth_depth = depth_frame.get_distance(mouth_x, mouth_y)

                # 시각화
                cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
                cv2.circle(color_image, (upper_x, upper_y), 4, (255, 0, 0), -1)
                cv2.circle(color_image, (lower_x, lower_y), 4, (0, 255, 0), -1)

                # 텍스트 표시
                text1 = f"Mouth pixel: ({mouth_x}, {mouth_y})"
                text2 = f"Mouth depth: {mouth_depth:.3f} m"

                cv2.putText(color_image, text1, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, text2, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()