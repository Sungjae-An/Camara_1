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
#     - 한 픽셀만 보면 노이즈가 많아서 주변 픽셀의 중간값 사용
#     - window_size=15 로 키워서 0.000m 오류 개선
# ============================================================
def get_stable_depth(depth_frame, x, y, window_size=15):
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
#     - Face Mesh: 얼굴 468개 점 인식 → 입 위치 추출
#     - Pose: 몸 33개 관절 인식 → 팔 위치 추출
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# [4] RealSense 카메라 설정 및 시작
#     - 색상 이미지: 화면에 보여줄 용도
#     - 깊이 이미지: 거리 측정용
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

window_name = "통합: 입 + 팔 + 안전영역 + 3D거리"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)


# ============================================================
# [5] 메인 루프: 매 프레임마다 반복 실행
# ============================================================
try:
    while True:

        # --- 5-1) 카메라에서 프레임 1번만 받기 ---
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # BGR (OpenCV용)
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # RGB (MediaPipe용)
        h, w, _ = color_image.shape


        # --- 5-2) 안전영역 다각형 그리기 ---
        if len(clicked_points) >= 3:
            overlay = color_image.copy()
            polygon = np.array(clicked_points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 255, 255))
            color_image = cv2.addWeighted(overlay, 0.25, color_image, 0.75, 0)

        # 클릭한 점들과 번호 표시
        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(color_image, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 점과 점 사이 선 연결
        for i in range(len(clicked_points) - 1):
            cv2.line(color_image, clicked_points[i], clicked_points[i + 1], (255, 0, 0), 2)

        # 마지막 점과 첫 점 연결 (다각형 닫기)
        if len(clicked_points) >= 3:
            cv2.line(color_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)


        # --- 5-3) Face Mesh로 입 위치 인식 ---
        # rgb_image를 Face Mesh에 넣으면 468개 얼굴 점을 돌려줌
        # 13번(윗입술), 14번(아랫입술)의 중간점 = 입 중심
        face_results = face_mesh.process(rgb_image)
        mouth_x, mouth_y, mouth_z = None, None, None

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

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
        # RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST 사용
        pose_results = pose.process(rgb_image)
        wrist_x, wrist_y, wrist_z = None, None, None

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

            # 관절 깊이값 텍스트 표시
            cv2.putText(color_image, f"Shoulder z={shoulder_z:.3f}m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Elbow z={elbow_z:.3f}m", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, f"Wrist z={wrist_z:.3f}m", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-5) 안전영역 판정 ---
        # 손목 좌표가 인식됐고, 안전영역이 3점 이상 정의됐을 때만 판정
        if wrist_x is not None and len(clicked_points) >= 3:
            polygon = np.array(clicked_points, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, (wrist_x, wrist_y), False)
            # result > 0 : 안에 있음
            # result = 0 : 경계선 위
            # result < 0 : 밖에 있음

            if result > 0:
                status_text  = "WRIST: INSIDE DANGER ZONE"
                status_color = (0, 0, 255)    # 빨간색
            elif result == 0:
                status_text  = "WRIST: ON EDGE"
                status_color = (0, 255, 255)  # 노란색
            else:
                status_text  = "WRIST: SAFE"
                status_color = (0, 255, 0)    # 초록색

            cv2.putText(color_image, status_text, (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)


        # --- 5-6) 손목 → 입 3D 거리 계산 ---
        # mouth_z와 wrist_z가 둘 다 측정됐을 때만 계산
        # 0.0이면 측정 실패이므로 제외
        if (mouth_x is not None and wrist_x is not None
                and mouth_z > 0 and wrist_z > 0):

            # intrinsics = 카메라 렌즈 고유 특성값: 초점거리 (fx, fy), 렌즈 중심점 (cx, cy), 해상도
            # 픽셀 좌표를 실제 미터 좌표로 변환할 때 필요해요
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                # .profile: 프레임의 설정정보, .as_video_stream_profile(): 영상 스트림 형식으로 변환, .intrinsics: 렌즈 특성값 꺼내기

            # 픽셀 좌표 + 깊이값 → 실제 3D 좌표(미터) 변환
            mouth_point = rs.rs2_deproject_pixel_to_point(  # rs2: realsense 2세대, deproject: 역투영 (픽셀 to 실제공간), poxel_to_point: 픽셀좌표를 3D 점으로 변환
                depth_intrin, [mouth_x, mouth_y], mouth_z)
            wrist_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            # mouth_point = [x(m), y(m), z(m)]
            # wrist_point = [x(m), y(m), z(m)]

            # 3D 거리 계산: √(dx² + dy² + dz²)
            dx = mouth_point[0] - wrist_point[0]
            dy = mouth_point[1] - wrist_point[1]
            dz = mouth_point[2] - wrist_point[2]
            distance_3d = (dx**2 + dy**2 + dz**2) ** 0.5
            # ** 0.5 = 제곱근(√) 을 구하는 파이썬 표현

            # 거리에 따라 색상과 메시지 변경
            # THRESHOLD = 입에 "도착했다" 고 판단하는 기준 거리
            # 일단 0.3m(30cm)로 설정, 실제 테스트 후 조정하세요
            THRESHOLD = 0.3

            if distance_3d < THRESHOLD:
                dist_color = (0, 0, 255)   # 빨강: 입에 가까움! (도착!)
                dist_text  = f"NEAR MOUTH! {distance_3d:.3f}m"
            else:
                dist_color = (0, 255, 0)   # 초록: 아직 멀리 있음
                dist_text  = f"Wrist->Mouth: {distance_3d:.3f}m"

            # 화면에 거리 텍스트 표시
            cv2.putText(color_image, dist_text, (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_color, 2)

            # 손목과 입을 선으로 연결 (거리 시각화)
            cv2.line(color_image,
                     (wrist_x, wrist_y),
                     (mouth_x, mouth_y),
                     dist_color, 2)


        # --- 5-7) 안내 문구 및 화면 출력 ---
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