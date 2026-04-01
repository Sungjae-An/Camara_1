import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# MediaPipe 최신버전은 mp.solutions.face_mesh가 안되기에, 버전다운 (0.10.9)을 하는게 편하다.

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1, # 얼굴 1개만 추적
    refine_landmarks=True # 입, 눈 주변을 더 정교하게
)

# RealSense 설정
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

window_name = "RealSense Mouth Point"

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # OpenCV BGR -> RGB 변환 (MediaPipe는 입력이 RGB라서 맞춰주기위함)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # 얼굴 랜드마크 추출
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = color_image.shape

                # 윗입술, 아랫입술 점 선택
                upper_lip = face_landmarks.landmark[13] # 13번 랜드마크=윗입술
                lower_lip = face_landmarks.landmark[14] # 14번 랜드마크=아랫입술

                # 픽셀 좌표로 변환 (MediaPipe 좌표는 픽셀이 아닌, 0~1 비율좌표이므로)
                upper_x = int(upper_lip.x * w)
                upper_y = int(upper_lip.y * h)

                lower_x = int(lower_lip.x * w)
                lower_y = int(lower_lip.y * h)

                # 입 중심점 계산
                mouth_x = (upper_x + lower_x) // 2
                mouth_y = (upper_y + lower_y) // 2

                # 점 표시
                cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)

                # 윗입술/아랫입술도 참고로 표시
                cv2.circle(color_image, (upper_x, upper_y), 4, (255, 0, 0), -1)
                cv2.circle(color_image, (lower_x, lower_y), 4, (0, 255, 0), -1)

                # 텍스트 표시
                text = f"Mouth: ({mouth_x}, {mouth_y})"
                cv2.putText(
                    color_image,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()