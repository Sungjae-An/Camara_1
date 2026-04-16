import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# openCV는 한글 인식못함. cv2.putText() 는 한글 넣으면 안됨.
# print()는 VS code 터미널에 출력하는거라 한글 넣어도 인식 됨.

# ============================================================
# [0] 전역 변수
# ============================================================
clicked_points = []   # 안전영역 점들

# 식판 칸 정보를 저장하는 딕셔너리
# {"밥": {"x": 100, "y": 200, "z": 0.5}, "국": {...}, ...}
food_zones = {}

# 현재 입력중인 칸 이름 (키보드로 입력)
current_zone_name = ""

# 모드: "setting" = 식판 설정 중, "running" = 실행 중
mode = "running"

# Depth 0일때 이전 프레임 유지하기 위함
last_wrist_z = 0.0
last_mouth_z = 0.0


# ============================================================
# [1] 마우스 콜백 함수
#     - 설정 모드일 때만 식판 칸 클릭 가능
#     - 실행 모드일 때는 안전영역 클릭
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global clicked_points, food_zones, current_zone_name, mode

    if event == cv2.EVENT_LBUTTONDOWN:

        if mode == "setting":
            # 설정 모드: 칸 이름이 입력된 상태에서만 클릭 가능
            if current_zone_name != "":
                # depth_frame은 param으로 받아옴
                depth_frame = param
                z = get_stable_depth(depth_frame, x, y)

                # 딕셔너리에 저장
                # {"밥": {"x": 100, "y": 200, "z": 0.5}} 형태
                food_zones[current_zone_name] = {"x": x, "y": y, "z": z}
                print(f"칸 저장: {current_zone_name} → ({x}, {y}, z={z:.3f}m)")

                # 이름 초기화 (다음 칸 입력 준비)
                current_zone_name = ""
            else:
                print("먼저 칸 이름을 입력하세요! (키보드로 입력 후 Enter)")

        elif mode == "running":
            # 실행 모드: 안전영역 점 추가
            clicked_points.append((x, y))
            print(f"안전영역 점 추가: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if mode == "running":
            if clicked_points:
                removed = clicked_points.pop()
                print(f"안전영역 점 제거: {removed}")


# ============================================================
# [2] 안정적인 깊이값 계산 함수
# ============================================================
def get_stable_depth(depth_frame, x, y, window_size=15):
    depths = []
    half = window_size // 2

    width = depth_frame.get_width()
    height = depth_frame.get_height()

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            px, py = x + dx, y + dy

            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0:
                depths.append(d)

    if len(depths) == 0:
        return 0.0

    return float(np.median(depths))


# ============================================================
# [3] MediaPipe 초기화
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
# 원래 D455의 깊이 센서는 색상 센서보다 해상도가 낮다. depth만 해상도를 최적 (840*480)으로 낮춤.
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color) # 깊이 이미지를 색상 이미지 기준으로 맞춰줌

window_name = "통합: 입 + 팔 + 안전영역 + 식판"
cv2.namedWindow(window_name)


# ============================================================
# [5] 메인 루프
# ============================================================
try:
    while True:

        # --- 5-1) 프레임 받기 ---
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)   # 색상 기준으로 깊이 정렬
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 마우스 콜백에 depth_frame 전달
        # → 클릭할 때 깊이값도 같이 저장하려고
        cv2.setMouseCallback(window_name, mouse_callback, depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape


        # --- 5-2) 현재 모드 표시 ---
        if mode == "setting":
            # 설정 모드: 상단에 안내 문구 표시
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 100, 200), -1)
            cv2.putText(color_image,
                        f"SETTING MODE | Type name + Enter, then Click | Input: [{current_zone_name}]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # 실행 모드: 상단에 초록 배경
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 150, 0), -1)
            cv2.putText(color_image,
                        "RUNNING MODE | s:setting | c:clear zones | q:quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # --- 5-3) 안전영역 다각형 그리기 ---
        if len(clicked_points) >= 3:
            overlay = color_image.copy()
            polygon = np.array(clicked_points, dtype=np.int32)
            cv2.fillPoly(overlay, [polygon], (0, 255, 255))
            color_image = cv2.addWeighted(overlay, 0.25, color_image, 0.75, 0)

        for i, (x, y) in enumerate(clicked_points):
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(color_image, str(i), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for i in range(len(clicked_points) - 1):
            cv2.line(color_image, clicked_points[i], clicked_points[i + 1], (255, 0, 0), 2)

        if len(clicked_points) >= 3:
            cv2.line(color_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)


        # --- 5-4) 식판 칸 표시 ---
        # food_zones 딕셔너리를 순회하면서 각 칸을 화면에 표시
        # .items() = 딕셔너리에서 (이름, 값) 쌍을 하나씩 꺼내는 함수
        for zone_name, zone_data in food_zones.items():
            zx = zone_data["x"]
            zy = zone_data["y"]
            zz = zone_data["z"]

            # 각 칸 위치에 원과 이름 표시
            cv2.circle(color_image, (zx, zy), 15, (255, 165, 0), 2)  # 주황색 원
            cv2.putText(color_image, f"{zone_name}({zz:.2f}m)",
                        (zx + 18, zy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


        # --- 5-5) Face Mesh로 입 위치 인식 ---
        face_results = face_mesh.process(rgb_image)
        mouth_x, mouth_y, mouth_z = None, None, None

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]

            upper_x = int(upper_lip.x * w)
            upper_y = int(upper_lip.y * h)
            lower_x = int(lower_lip.x * w)
            lower_y = int(lower_lip.y * h)

            mouth_x = (upper_x + lower_x) // 2
            mouth_y = (upper_y + lower_y) // 2
            mouth_z = get_stable_depth(depth_frame, mouth_x, mouth_y)

            cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
            cv2.putText(color_image, f"Mouth z={mouth_z:.3f}m", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-6) Pose로 팔 관절 인식 ---
        pose_results = pose.process(rgb_image)
        wrist_x, wrist_y, wrist_z = None, None, None

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark

            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            shoulder_x = max(0, min(w-1, int(shoulder.x * w)))
            shoulder_y = max(0, min(h-1, int(shoulder.y * h)))
            elbow_x    = max(0, min(w-1, int(elbow.x * w)))
            elbow_y    = max(0, min(h-1, int(elbow.y * h)))
            wrist_x    = max(0, min(w-1, int(wrist.x * w)))
            wrist_y    = max(0, min(h-1, int(wrist.y * h)))

            shoulder_z = get_stable_depth(depth_frame, shoulder_x, shoulder_y)
            elbow_z    = get_stable_depth(depth_frame, elbow_x, elbow_y)
            wrist_z    = get_stable_depth(depth_frame, wrist_x, wrist_y, window_size=25) # wrist는 depth 잘잡고싶어서 window size 키움
            if wrist_z > 0:
                last_wrist_z = wrist_z      # 유효한 값이면 저장
            else:
                wrist_z = last_wrist_z      # 0이면 마지막 유효값 사용

            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)
            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            cv2.putText(color_image, f"Shoulder z={shoulder_z:.3f}m", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Elbow z={elbow_z:.3f}m", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, f"Wrist z={wrist_z:.3f}m", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-7) 안전영역 판정 ---
        if wrist_x is not None and len(clicked_points) >= 3:
            polygon = np.array(clicked_points, dtype=np.int32)
            result = cv2.pointPolygonTest(polygon, (wrist_x, wrist_y), False)

            if result > 0:
                status_text  = "WRIST: INSIDE DANGER ZONE"
                status_color = (0, 0, 255)
            elif result == 0:
                status_text  = "WRIST: ON EDGE"
                status_color = (0, 255, 255)
            else:
                status_text  = "WRIST: SAFE"
                status_color = (0, 255, 0)

            cv2.putText(color_image, status_text, (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)


        # --- 5-8) 손목 → 입 3D 거리 계산 ---
        if (mouth_x is not None and wrist_x is not None
                and mouth_z > 0 and wrist_z > 0):

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            mouth_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [mouth_x, mouth_y], mouth_z)
            wrist_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            dx = mouth_point[0] - wrist_point[0]
            dy = mouth_point[1] - wrist_point[1]
            dz = mouth_point[2] - wrist_point[2]
            distance_3d = (dx**2 + dy**2 + dz**2) ** 0.5

            THRESHOLD = 0.3

            if distance_3d < THRESHOLD:
                dist_color = (0, 0, 255)
                dist_text  = f"NEAR MOUTH! {distance_3d:.3f}m"
            else:
                dist_color = (0, 255, 0)
                dist_text  = f"Wrist->Mouth: {distance_3d:.3f}m"

            cv2.putText(color_image, dist_text, (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, dist_color, 2)

            cv2.line(color_image,
                     (wrist_x, wrist_y),
                     (mouth_x, mouth_y),
                     dist_color, 2)


        # --- 5-9) 손목과 가장 가까운 식판 칸 찾기 ---
        # 실행 모드이고, 손목이 인식됐고, 식판 칸이 하나 이상 있을 때만
        if mode == "running" and wrist_x is not None and len(food_zones) > 0 and wrist_z > 0:

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            # 손목의 실제 3D 좌표
            wrist_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            # 가장 가까운 칸을 찾기 위한 변수
            # 처음엔 무한대로 설정해두고, 더 가까운 칸이 나오면 갱신
            min_distance = float('inf')  # inf = 무한대
            nearest_zone = None          # 가장 가까운 칸 이름

            # 모든 식판 칸을 순회하면서 손목과의 거리 계산
            for zone_name, zone_data in food_zones.items():
                zz = zone_data["z"]

                if zz <= 0:  # 깊이 측정 실패한 칸은 건너뜀
                    continue

                # 각 칸의 실제 3D 좌표
                zone_point = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [zone_data["x"], zone_data["y"]], zz)

                # 손목과 칸 사이의 3D 거리
                dx = wrist_point[0] - zone_point[0]
                dy = wrist_point[1] - zone_point[1]
                dz = wrist_point[2] - zone_point[2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5

                # 더 가까운 칸이 나오면 갱신
                if dist < min_distance:
                    min_distance = dist
                    nearest_zone = zone_name

            # 가장 가까운 칸 화면에 표시
            if nearest_zone is not None:
                zone_text = f"Nearest: {nearest_zone} ({min_distance:.3f}m)"
                cv2.putText(color_image, zone_text, (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

                # 손목과 가장 가까운 칸을 선으로 연결
                zx = food_zones[nearest_zone]["x"]
                zy = food_zones[nearest_zone]["y"]
                cv2.line(color_image, (wrist_x, wrist_y), (zx, zy), (255, 165, 0), 2)


        # --- 5-10) 안내 문구 및 화면 출력 ---
        cv2.putText(color_image,
                    "s:setting | r:running | c:clear zones | q:quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, color_image)


        # --- 5-11) 키보드 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF

        if mode == "setting":
            # 설정 모드일 때는 글자 입력을 먼저 처리
            if key == 27:  # ESC키 = 27 → 설정 모드 탈출
                mode = "running"
                current_zone_name = ""
                print(f"Running mode! Saved zones: {list(food_zones.keys())}")

            elif key == 13:  # Enter키 = 13
                if current_zone_name != "":
                    print(f"Name confirmed: [{current_zone_name}] → Click on the zone!")

            elif key == 8:  # Backspace키 = 8
                current_zone_name = current_zone_name[:-1]
                print(f"Input: [{current_zone_name}]")

            elif 32 <= key <= 126:  # 일반 문자
                current_zone_name += chr(key)           # chr(115)='s' [ord()의 반대 코드]
                print(f"Input: [{current_zone_name}]")

        else:
            # 실행 모드일 때만 단축키 처리
            if key == ord('q'):
                break

            elif key == ord('s'):
                mode = "setting"
                current_zone_name = ""
                print("Setting mode! Type zone name + Enter, then click!")

            elif key == ord('r'):
                mode = "running"
                print(f"Running mode! Saved zones: {list(food_zones.keys())}")

            elif key == ord('c'):
                food_zones.clear()
                current_zone_name = ""
                print("Zone settings cleared!")


# ============================================================
# [6] 종료 처리
# ============================================================
finally:
    pipeline.stop()
    cv2.destroyAllWindows()