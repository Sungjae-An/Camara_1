import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ============================================================
# [0] 전역 변수
# ============================================================
clicked_points = []     # 안전영역 점들
food_zones = {}         # 식판 칸 정보
current_zone_name = ""  # 현재 입력중인 칸 이름
mode = "running"        # 현재 모드

# 테이블 평면 관련
table_points_3d = []    # 테이블 클릭 점들의 실제 3D 좌표
table_plane = None      # 계산된 평면 방정식 (a, b, c, d)
table_mode = False      # 테이블 설정 중인지 여부

# depth 이전 프레임 유지
last_wrist_z = 0.0
last_mouth_z = 0.0


# ============================================================
# [1] 평면 방정식 계산 함수
#     - 3D 점들을 입력받아 평면 방정식 계수(a,b,c,d)를 반환
#     - np.linalg.lstsq = 점들을 가장 잘 설명하는 평면을 계산
# ============================================================
def calculate_plane(points_3d):
    # points_3d = [[x1,y1,z1], [x2,y2,z2], ...]
    points = np.array(points_3d)

    # 평면 방정식: ax + by + d = z 형태로 변환해서 풀기
    # (나중에 다시 ax + by + cz + d = 0 형태로 변환)
    A = np.column_stack([points[:, 0],   # x값들
                         points[:, 1],   # y값들
                         np.ones(len(points))])  # 1들 (d항을 위해)
    b = points[:, 2]  # z값들

    # lstsq = least squares (최소제곱법)
    # → 모든 점과의 오차가 가장 작은 평면을 찾아줌
    result = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, d = result[0]

    # ax + by + d = z
    # → ax + by - z + d = 0
    # → 법선벡터 = (a, b, -1), d = d
    return (a, b_coef, -1, d)


# ============================================================
# [2] 점과 평면 사이의 거리 계산 함수
#     - 손목이 테이블 평면으로부터 얼마나 떨어져 있는지
# ============================================================
def distance_point_to_plane(point, plane):
    a, b, c, d = plane
    x, y, z = point

    # 거리 공식: |ax + by + cz + d| / √(a²+b²+c²)
    numerator = abs(a*x + b*y + c*z + d)
    denominator = (a**2 + b**2 + c**2) ** 0.5

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# [3] 마우스 콜백 함수
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global clicked_points, food_zones, current_zone_name
    global mode, table_points_3d, table_plane, table_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame = param

        if table_mode:
            # 테이블 설정 모드: 클릭한 점의 3D 좌표 저장
            z = get_stable_depth(depth_frame, x, y)
            if z > 0:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], z)
                table_points_3d.append(point_3d)
                print(f"테이블 점 추가: ({x},{y}) → 3D{point_3d}")

                # 4개 이상 점이 모이면 평면 자동 계산
                if len(table_points_3d) >= 4:
                    table_plane = calculate_plane(table_points_3d)
                    print(f"평면 계산 완료! 계수: {table_plane}")

            else:
                print("깊이값을 못 읽었어요. 다른 곳을 클릭해보세요!")

        elif mode == "setting":
            if current_zone_name != "":
                z = get_stable_depth(depth_frame, x, y)
                food_zones[current_zone_name] = {"x": x, "y": y, "z": z}
                print(f"칸 저장: {current_zone_name} → ({x},{y}, z={z:.3f}m)")
                current_zone_name = ""
            else:
                print("먼저 칸 이름을 입력하세요!")

        elif mode == "running":
            clicked_points.append((x, y))
            print(f"안전영역 점 추가: ({x},{y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if table_mode:
            # 테이블 점 마지막 것 제거
            if table_points_3d:
                removed = table_points_3d.pop()
                print(f"테이블 점 제거: {removed}")
                # 점이 4개 미만이 되면 평면 초기화
                if len(table_points_3d) < 4:
                    table_plane = None
                    print("평면 초기화 (점이 4개 미만)")

        elif mode == "running":
            if clicked_points:
                removed = clicked_points.pop()
                print(f"안전영역 점 제거: {removed}")


# ============================================================
# [4] 안정적인 깊이값 계산 함수
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
# [5] MediaPipe 초기화
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
# [6] RealSense 카메라 설정 및 시작
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)  # 색상 기준으로 깊이 정렬

window_name = "Wearable Feeding Assistant"
cv2.namedWindow(window_name)


# ============================================================
# [7] 메인 루프
# ============================================================
try:
    while True:

        # --- 7-1) 프레임 받기 ---
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        cv2.setMouseCallback(window_name, mouse_callback, depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _ = color_image.shape


        # --- 7-2) 모드 표시 ---
        if table_mode:
            cv2.rectangle(color_image, (0, 0), (w, 40), (150, 0, 150), -1)
            cv2.putText(color_image,
                        f"TABLE MODE | Click table surface | Points: {len(table_points_3d)} | ESC: done",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif mode == "setting":
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 100, 200), -1)
            cv2.putText(color_image,
                        f"SETTING MODE | Input: [{current_zone_name}] | ESC: done",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 150, 0), -1)
            cv2.putText(color_image,
                        "RUNNING MODE | s:setting | t:table | c:clear | q:quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # --- 7-3) 테이블 클릭 점 표시 ---
        for i, pt in enumerate(table_points_3d):
            # pt는 3D 좌표라서 화면 표시용 픽셀 좌표가 없음
            # → 테이블 점 개수만 화면에 표시
            pass

        # 평면 계산 완료 여부 표시
        if table_plane is not None:
            cv2.putText(color_image, "Table plane: OK",
                        (w - 200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 150), 2)
        else:
            cv2.putText(color_image,
                        f"Table plane: need {max(0, 4-len(table_points_3d))} more points",
                        (w - 300, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)


        # --- 7-4) 안전영역 다각형 그리기 ---
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
            cv2.line(color_image, clicked_points[i], clicked_points[i+1], (255, 0, 0), 2)

        if len(clicked_points) >= 3:
            cv2.line(color_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)


        # --- 7-5) 식판 칸 표시 ---
        for zone_name, zone_data in food_zones.items():
            zx = zone_data["x"]
            zy = zone_data["y"]
            zz = zone_data["z"]
            cv2.circle(color_image, (zx, zy), 15, (255, 165, 0), 2)
            cv2.putText(color_image, f"{zone_name}({zz:.2f}m)",
                        (zx + 18, zy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


        # --- 7-6) Face Mesh로 입 위치 인식 ---
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

            # depth 0일때 이전 프레임 유지
            if mouth_z > 0:
                last_mouth_z = mouth_z
            else:
                mouth_z = last_mouth_z

            cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
            cv2.putText(color_image, f"Mouth z={mouth_z:.3f}m", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 7-7) Pose로 팔 관절 인식 ---
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
            wrist_z    = get_stable_depth(depth_frame, wrist_x, wrist_y, window_size=25)

            # depth 0일때 이전 프레임 유지
            if wrist_z > 0:
                last_wrist_z = wrist_z
            else:
                wrist_z = last_wrist_z

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


        # --- 7-8) 테이블 평면 기반 위험 판정 ---
        # 손목이 인식됐고, 평면이 계산됐을 때만 판정
        if (wrist_x is not None and wrist_z > 0 and table_plane is not None):

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            wrist_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            # 손목과 테이블 평면 사이의 실제 거리
            dist_to_table = distance_point_to_plane(wrist_point, table_plane)

            # 거리에 따라 3단계 판정
            # SAFE_HEIGHT: 이 거리 이상이면 안전 (테이블 위 허공)
            # WARN_HEIGHT: 이 거리 이하면 주의
            SAFE_HEIGHT = 0.10   # 10cm 이상 → 안전
            WARN_HEIGHT = 0.05   # 5cm 이하  → 위험

            if dist_to_table >= SAFE_HEIGHT:
                danger_text  = f"SAFE (table dist: {dist_to_table:.3f}m)"
                danger_color = (0, 255, 0)     # 초록
            elif dist_to_table >= WARN_HEIGHT:
                danger_text  = f"WARNING (table dist: {dist_to_table:.3f}m)"
                danger_color = (0, 255, 255)   # 노랑
            else:
                danger_text  = f"DANGER! (table dist: {dist_to_table:.3f}m)"
                danger_color = (0, 0, 255)     # 빨강

            cv2.putText(color_image, danger_text, (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, danger_color, 2)


        # --- 7-9) 손목 → 입 3D 거리 계산 ---
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

            THRESHOLD = 0.1   # 10cm 이내면 도착

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


        # --- 7-10) 손목과 가장 가까운 식판 칸 찾기 ---
        if (mode == "running" and wrist_x is not None
                and len(food_zones) > 0 and wrist_z > 0):

            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            wrist_point = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            min_distance = float('inf')
            nearest_zone = None

            for zone_name, zone_data in food_zones.items():
                zz = zone_data["z"]
                if zz <= 0:
                    continue

                zone_point = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [zone_data["x"], zone_data["y"]], zz)

                dx = wrist_point[0] - zone_point[0]
                dy = wrist_point[1] - zone_point[1]
                dz = wrist_point[2] - zone_point[2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5

                if dist < min_distance:
                    min_distance = dist
                    nearest_zone = zone_name

            if nearest_zone is not None:
                zone_text = f"Nearest: {nearest_zone} ({min_distance:.3f}m)"
                cv2.putText(color_image, zone_text, (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

                zx = food_zones[nearest_zone]["x"]
                zy = food_zones[nearest_zone]["y"]
                cv2.line(color_image, (wrist_x, wrist_y), (zx, zy), (255, 165, 0), 2)


        # --- 7-11) 안내 문구 및 화면 출력 ---
        cv2.putText(color_image,
                    "s:setting | t:table | r:running | c:clear zones | q:quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, color_image)


        # --- 7-12) 키보드 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF

        if table_mode:
            # 테이블 설정 모드에서는 ESC만 탈출
            if key == 27:  # ESC
                table_mode = False
                if table_plane is not None:
                    print("Table plane ready!")
                else:
                    print(f"Need {max(0, 4-len(table_points_3d))} more points!")

        elif mode == "setting":
            if key == 27:  # ESC → 실행 모드로
                mode = "running"
                current_zone_name = ""
                print(f"Running mode! Zones: {list(food_zones.keys())}")

            elif key == 13:  # Enter
                if current_zone_name != "":
                    print(f"Name confirmed: [{current_zone_name}] → Click!")

            elif key == 8:  # Backspace
                current_zone_name = current_zone_name[:-1]
                print(f"Input: [{current_zone_name}]")

            elif 32 <= key <= 126:
                current_zone_name += chr(key)
                print(f"Input: [{current_zone_name}]")

        else:
            # 실행 모드
            if key == ord('q'):
                break
            elif key == ord('s'):
                mode = "setting"
                current_zone_name = ""
                print("Setting mode!")
            elif key == ord('t'):
                table_mode = True
                print("Table mode! Click 4+ points on table surface!")
            elif key == ord('r'):
                mode = "running"
                print(f"Running mode! Zones: {list(food_zones.keys())}")
            elif key == ord('c'):
                food_zones.clear()
                current_zone_name = ""
                print("Zones cleared!")


# ============================================================
# [8] 종료 처리
# ============================================================
finally:
    pipeline.stop()
    cv2.destroyAllWindows()