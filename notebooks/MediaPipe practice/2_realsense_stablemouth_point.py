import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp


def get_stable_depth(depth_frame, x, y, window_size=5): # x,y는 입 중심점, window size 5라는 뜻은 주변 5x5 픽셀을 보겠다는 뜻.
    depths = []
    half = window_size // 2

    for dy in range(-half, half + 1): 
        for dx in range(-half, half + 1):
            px = x + dx
            py = y + dy

            if px < 0 or py < 0:
                continue

            d = depth_frame.get_distance(px, py) # 각 픽셀의 depth 읽어라

            if d > 0: # Depth가 0인 데이터는 빼라
                depths.append(d)

    if len(depths) == 0:
        return 0.0

    return float(np.median(depths)) # median값을 반환해라.


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

window_name = "RealSense Mouth Stable Depth"

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

                upper_lip = face_landmarks.landmark[13]
                lower_lip = face_landmarks.landmark[14]

                upper_x = int(upper_lip.x * w)
                upper_y = int(upper_lip.y * h)

                lower_x = int(lower_lip.x * w)
                lower_y = int(lower_lip.y * h)

                mouth_x = (upper_x + lower_x) // 2
                mouth_y = (upper_y + lower_y) // 2

                raw_depth = depth_frame.get_distance(mouth_x, mouth_y)
                stable_depth = get_stable_depth(depth_frame, mouth_x, mouth_y, window_size=5)

                # 점 표시
                cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
                cv2.circle(color_image, (upper_x, upper_y), 4, (255, 0, 0), -1)
                cv2.circle(color_image, (lower_x, lower_y), 4, (0, 255, 0), -1)

                text1 = f"Mouth pixel: ({mouth_x}, {mouth_y})"
                text2 = f"Raw depth: {raw_depth:.3f} m"
                text3 = f"Stable depth: {stable_depth:.3f} m"

                cv2.putText(color_image, text1, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, text2, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(color_image, text3, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()