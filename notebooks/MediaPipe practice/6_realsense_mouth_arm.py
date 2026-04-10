import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ============================================================
# [0] 전역 변수: 마우스 클릭으로 저장되는 안전영역 점들
# ============================================================
clicked_points = []


# ============================================================
# [1] 마우스 콜백 함수
#     - 왼쪽 클릭: 점 추가
#     - 오른쪽 클릭: 마지막 점 제거
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"점 추가: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if clicked_points:
            removed = clicked_points.pop()
            print(f"점 제거: {removed}")


# ============================================================
# [2] 안정적인 깊이값 계산 함수
#     - 한 픽셀만 보면 노이즈가 많아서 주변 5x5 픽셀의 중간값을 사용
#     - depth_frame: 카메라에서 받은 깊이 이미지
#     - x, y: 측정하고 싶은 위치
#     - 반환값: 미터 단위 거리 (예: 0.452 = 45.2cm)
# ============================================================
def get_stable_depth(depth_frame, x, y, window_size=5):
    depths = []
    half = window_size // 2

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px, py = x + dx, y + dy

            # 이미지 경계 밖으로 나가면 건너뜀
            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0:  # 0은 측정 실패를 의미하므로 제외
                depths.append(d)

    if len(depths) == 0:
        return 0.0

    return float(np.median(depths))  # 중간값 반환 (튀는 값 영향 최소화)


# ============================================================
# [3] MediaPipe 초기화
#     - Face Mesh: 얼굴의 468개 점을 인식 → 입 위치 추출에 사용
#     - Pose: 몸의 33개 관절을 인식 → 팔 위치 추출에 사용
#     두 분석기를 동시에 만들어두고, 매 프레임마다 둘 다 실행할 거예요
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,   # False = 동영상 모드 (연속 추적)
    max_num_faces=1,           # 얼굴 1개만 인식
    refine_landmarks=True      # 입술/눈 등 세밀한 점도 인식
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,        # 0=빠름, 1=보통, 2=정확 → 1로 균형
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# [4] RealSense 카메라 설정 및 시작
#     - 색상 이미지(BGR): 화면에 보여줄 용도
#     - 깊이 이미지(Z16): 각 픽셀까지의 거리 측정용
#     - 둘 다 640x480, 30fps
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

window_name = "통합: 입 + 팔 + 안전영역"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


# ============================================================
# [5] 메인 루프: 매 프레임마다 반복 실행
# ============================================================
try:
    while True:

        # --- 5-1) 카메라에서 프레임 1번만 받기 ---
        # 이전에는 두 코드가 각각 프레임을 받았는데,
        # 이제 한 번만 받고 Face Mesh / Pose 둘 다에 전달!
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # BGR (OpenCV용)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # RGB (MediaPipe용)
        h, w, _ = color_image.shape


        # --- 5-2) 안전영역 다각형 그리기 ---
        # 3개 이상의 점이 있을 때만 다각형을 채워서 표시
        if len(clicked_points) >= 3:
            overlay = color_image.copy()
            polygon = np.array(clicked_points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 255, 255))  # 노란색으로 채우기
            color_image = cv2.addWeighted(overlay, 0.25, color_image, 0.75, 0)  # 25% 투명도

        # 클릭한 점들과 연결선 표시
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(color_image, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i in range(len(clicked_points) - 1):
            cv2.line(color_image, clicked_points[i], clicked_points[i + 1], (255, 0, 0), 2)

        if len(clicked_points) >= 3:
            cv2.line(color_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)


        # --- 5-3) Face Mesh로 입 위치 인식 ---
        # rgb_image를 Face Mesh에 넣으면 468개 얼굴 점을 돌려줌
        # 그 중 13번(윗입술), 14번(아랫입술)의 중간점 = 입 중심
        face_results = face_mesh.process(rgb_image)
        mouth_x, mouth_y, mouth_z = None, None, None  # 나중에 팔과 비교할 때 쓸 변수

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]  # 첫 번째 얼굴만 사용

            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            # landmark 좌표는 0~1 사이 비율값 → 실제 픽셀로 변환
            upper_x = int(upper_lip.x * w)
            upper_y = int(upper_lip.y * h)
            lower_x = int(lower_lip.x * w)
            lower_y = int(lower_lip.y * h)

            mouth_x = (upper_x + lower_x) // 2
            mouth_y = (upper_y + lower_y) // 2
            mouth_z = get_stable_depth(depth_frame, mouth_x, mouth_y)

            # 입 위치 표시
            cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
            cv2.putText(color_image, f"Mouth z={mouth_z:.3f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-4) Pose로 팔 관절 인식 ---
        # rgb_image를 Pose에 넣으면 33개 관절 점을 돌려줌
        # 오른쪽 어깨(RIGHT_SHOULDER), 팔꿈치(RIGHT_ELBOW), 손목(RIGHT_WRIST) 사용
        pose_results = pose.process(rgb_image)
        wrist_x, wrist_y = None, None  # 나중에 안전영역 판정에 쓸 변수

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # 비율값 → 픽셀 변환 (화면 밖으로 나가지 않도록 clamp)
            shoulder_x = max(0, min(w-1, int(shoulder.x * w)))
            shoulder_y = max(0, min(h-1, int(shoulder.y * h)))
            elbow_x    = max(0, min(w-1, int(elbow.x * w)))
            elbow_y    = max(0, min(h-1, int(elbow.y * h)))
            wrist_x    = max(0, min(w-1, int(wrist.x * w)))
            wrist_y    = max(0, min(h-1, int(wrist.y * h)))

            # 깊이값 측정
            shoulder_z = get_stable_depth(depth_frame, shoulder_x, shoulder_y)
            elbow_z    = get_stable_depth(depth_frame, elbow_x, elbow_y)
            wrist_z    = get_stable_depth(depth_frame, wrist_x, wrist_y)

            # 관절 점과 선 표시
            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)
            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            # 관절 이름 표시
            cv2.putText(color_image, f"Shoulder z={shoulder_z:.3f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Elbow z={elbow_z:.3f}m", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, f"Wrist z={wrist_z:.3f}m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-5) 안전영역 판정 ---
        # 손목 좌표가 인식되었고, 안전영역이 3점 이상 정의되어 있을 때만 판정
        if wrist_x is not None and len(clicked_points) >= 3:
            polygon = np.array(clicked_points, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, (wrist_x, wrist_y), False)
            # result > 0 : 안에 있음
            # result = 0 : 경계선 위
            # result < 0 : 밖에 있음

            if result > 0:
                status_text  = "WRIST: INSIDE DANGER ZONE"
                status_color = (0, 0, 255)   # 빨간색
            elif result == 0:
                status_text  = "WRIST: ON EDGE"
                status_color = (0, 255, 255)  # 노란색
            else:
                status_text  = "WRIST: SAFE"
                status_color = (0, 255, 0)    # 초록색

            cv2.putText(color_image, status_text, (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)


        # --- 5-6) 안내 문구 및 화면 출력 ---
        cv2.putText(color_image,
                    "Left:add | Right:remove | c:clear | q:quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            clicked_points.clear()
            print("안전영역 초기화")

# ============================================================
# [6] 종료 처리
#     - 오류가 나도 반드시 카메라와 창을 닫아야 함
#     - finally 블록은 어떤 상황에서도 실행됨
# ============================================================
finally:
    pipeline.stop()
    cv2.destroyAllWindows()