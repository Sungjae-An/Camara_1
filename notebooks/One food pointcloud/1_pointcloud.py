import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ============================================================
# [0] 전역 변수
# ============================================================
mode = "running"

# 테이블 평면 관련
table_points_3d = []
table_plane = None
table_mode = False

# depth 이전 프레임 유지
last_wrist_z = 0.0
last_mouth_z = 0.0

# ROI 관련
food_roi = {}
drag_start = None
drag_end = None
dragging = False
current_roi_name = ""
roi_mode = False


# ============================================================
# [1] 평면 방정식 계산 함수
# ============================================================
def calculate_plane(points_3d):
    points = np.array(points_3d)
    A = np.column_stack([points[:, 0],
                         points[:, 1],
                         np.ones(len(points))])
    b = points[:, 2]
    result = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, d = result[0]
    return (a, b_coef, -1, d)


# ============================================================
# [2] 점과 평면 사이의 거리 계산 함수
# ============================================================
def distance_point_to_plane(point, plane):
    a, b, c, d = plane
    x, y, z = point
    numerator = abs(a*x + b*y + c*z + d)
    denominator = (a**2 + b**2 + c**2) ** 0.5
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ============================================================
# [3] 안정적인 깊이값 계산 함수
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
# [4] 포인트 클라우드 추출 함수
# ============================================================
def get_food_pointcloud(depth_frame, roi):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    points_3d = []

    x1, y1 = roi["x1"], roi["y1"]
    x2, y2 = roi["x2"], roi["y2"]

    step = 5  # 5픽셀마다 샘플링

    for py in range(y1, y2, step):
        for px in range(x1, x2, step):
            z = get_stable_depth(depth_frame, px, py, window_size=3)
            if z > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [px, py], z)
                points_3d.append(point_3d)

    return points_3d


# ============================================================
# [5] 음식 표면 분석 함수
# ============================================================
def analyze_food_surface(points_3d, table_plane):
    if len(points_3d) < 10:
        return None

    heights = []
    for point in points_3d:
        if table_plane is not None:
            h = distance_point_to_plane(point, table_plane)
        else:
            h = point[2]
        heights.append(h)

    heights = np.array(heights)
    food_heights = heights[heights > 0.01]  # 1cm 이상만 음식으로 간주

    if len(food_heights) == 0:
        return None

    return {
        "avg_height":  float(np.mean(food_heights)),
        "max_height":  float(np.max(food_heights)),
        "std_height":  float(np.std(food_heights)),
        "food_amount": len(food_heights) / len(heights)
    }


# ============================================================
# [6] 마우스 콜백 함수
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global table_points_3d, table_plane, table_mode
    global drag_start, drag_end, dragging
    global food_roi, current_roi_name, roi_mode

    depth_frame = param

    # ── ROI 드래그 모드 ──
    if roi_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start = (x, y)
            dragging = True

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drag_end = (x, y)
            dragging = False

            if current_roi_name != "" and drag_start is not None:
                x1 = min(drag_start[0], drag_end[0])
                y1 = min(drag_start[1], drag_end[1])
                x2 = max(drag_start[0], drag_end[0])
                y2 = max(drag_start[1], drag_end[1])
                food_roi[current_roi_name] = {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                }
                print(f"ROI saved: {current_roi_name} → ({x1},{y1})~({x2},{y2})")
                current_roi_name = ""
                drag_start = None
                drag_end = None
        return

    # ── 테이블 모드 ──
    if event == cv2.EVENT_LBUTTONDOWN:
        if table_mode:
            z = get_stable_depth(depth_frame, x, y)
            if z > 0:
                depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                point_3d = rs.rs2_deproject_pixel_to_point(
                    depth_intrin, [x, y], z)
                table_points_3d.append(point_3d)
                print(f"Table point added: ({x},{y}) z={z:.3f}m")
                if len(table_points_3d) >= 4:
                    table_plane = calculate_plane(table_points_3d)
                    print("Table plane calculated!")
            else:
                print("Depth not detected. Try another spot!")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if table_mode:
            if table_points_3d:
                table_points_3d.pop()
                if len(table_points_3d) < 4:
                    table_plane = None
                    print("Table plane reset (less than 4 points)")


# ============================================================
# [7] MediaPipe 초기화
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
# [8] RealSense 카메라 설정 및 시작
# ============================================================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848,  480, rs.format.z16,  30)
pipeline.start(config)
align = rs.align(rs.stream.color)

window_name = "Wearable Feeding Assistant"
cv2.namedWindow(window_name)


# ============================================================
# [9] 메인 루프
# ============================================================
try:
    while True:

        # --- 9-1) 프레임 받기 ---
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        cv2.setMouseCallback(window_name, mouse_callback, depth_frame)

        color_image = np.asanyarray(color_frame.get_data())
        rgb_image   = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        h, w, _     = color_image.shape


        # --- 9-2) 모드 표시 ---
        if roi_mode:
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 150, 150), -1)
            cv2.putText(color_image,
                        f"ROI MODE | Type name + Enter, drag area | Input:[{current_roi_name}]",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        elif table_mode:
            cv2.rectangle(color_image, (0, 0), (w, 40), (150, 0, 150), -1)
            cv2.putText(color_image,
                        f"TABLE MODE | Click table surface | Points:{len(table_points_3d)} | ESC:done",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        else:
            cv2.rectangle(color_image, (0, 0), (w, 40), (0, 150, 0), -1)
            cv2.putText(color_image,
                        "RUNNING | t:table | v:ROI | c:clear | q:quit",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # --- 9-3) 테이블 평면 상태 표시 ---
        if table_plane is not None:
            cv2.putText(color_image, "Table plane: READY",
                        (w - 280, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 150), 2)
        else:
            need = max(0, 4 - len(table_points_3d))
            cv2.putText(color_image,
                        f"Table plane: need {need} more points (t key)",
                        (w - 400, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)


        # --- 9-4) ROI 영역 표시 및 음식 표면 분석 ---
        for roi_name, roi in food_roi.items():
            rx1, ry1 = roi["x1"], roi["y1"]
            rx2, ry2 = roi["x2"], roi["y2"]

            cv2.rectangle(color_image, (rx1, ry1), (rx2, ry2), (0, 200, 200), 2)
            cv2.putText(color_image, roi_name, (rx1, ry1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

            points_3d = get_food_pointcloud(depth_frame, roi)
            analysis  = analyze_food_surface(points_3d, table_plane)

            if analysis is not None:
                avg_h  = analysis["avg_height"]
                std_h  = analysis["std_height"]
                amount = analysis["food_amount"]

                if avg_h > 0.03:
                    info_color = (0, 255, 0)
                elif avg_h > 0.01:
                    info_color = (0, 255, 255)
                else:
                    info_color = (0, 0, 255)

                cv2.putText(color_image,
                            f"h={avg_h*100:.1f}cm std={std_h*100:.1f}cm amt={amount*100:.0f}%",
                            (rx1, ry2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, info_color, 2)

        # 드래그 중 실시간 사각형 표시
        if dragging and drag_start and drag_end:
            cv2.rectangle(color_image, drag_start, drag_end, (0, 200, 200), 1)


        # --- 9-5) Face Mesh로 입 위치 인식 ---
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

            if mouth_z > 0:
                last_mouth_z = mouth_z
            else:
                mouth_z = last_mouth_z

            cv2.circle(color_image, (mouth_x, mouth_y), 6, (0, 0, 255), -1)
            cv2.putText(color_image, f"Mouth z={mouth_z:.3f}m", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 9-6) Pose로 팔 관절 인식 ---
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


        # --- 9-7) 테이블 평면 기반 위험 판정 ---
        if wrist_x is not None and wrist_z > 0 and table_plane is not None:
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            wrist_point  = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [wrist_x, wrist_y], wrist_z)

            dist_to_table = distance_point_to_plane(wrist_point, table_plane)

            SAFE_HEIGHT = 0.10   # 10cm 이상 → 안전
            WARN_HEIGHT = 0.05   # 5cm 이하  → 위험

            if dist_to_table >= SAFE_HEIGHT:
                danger_color = (0, 255, 0)
                danger_text  = f"Table dist: {dist_to_table:.3f}m SAFE"
            elif dist_to_table >= WARN_HEIGHT:
                danger_color = (0, 255, 255)
                danger_text  = f"Table dist: {dist_to_table:.3f}m WARNING"
                cv2.rectangle(color_image, (0, 0), (w-1, h-1), (0, 255, 255), 4)
            else:
                danger_color = (0, 0, 255)
                danger_text  = f"Table dist: {dist_to_table:.3f}m DANGER!"
                cv2.rectangle(color_image, (0, 0), (w-1, h-1), (0, 0, 255), 8)

            cv2.putText(color_image, danger_text, (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, danger_color, 2)


        # --- 9-8) 손목 → 입 3D 거리 계산 ---
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

            THRESHOLD = 0.1  # 10cm 이내면 도착

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


        # --- 9-9) 화면 출력 ---
        cv2.imshow(window_name, color_image)


        # --- 9-10) 키보드 입력 처리 ---
        key = cv2.waitKey(1) & 0xFF

        if roi_mode:
            if key == 27:  # ESC
                roi_mode = False
                current_roi_name = ""
                drag_start = None
                drag_end   = None
                print(f"ROI mode done! Saved: {list(food_roi.keys())}")
            elif key == 13:  # Enter
                if current_roi_name != "":
                    print(f"ROI name confirmed: [{current_roi_name}] → Now drag!")
            elif key == 8:   # Backspace
                current_roi_name = current_roi_name[:-1]
                print(f"Input: [{current_roi_name}]")
            elif 32 <= key <= 126:
                current_roi_name += chr(key)
                print(f"Input: [{current_roi_name}]")

        elif table_mode:
            if key == 27:  # ESC
                table_mode = False
                if table_plane is not None:
                    print("Table plane ready!")
                else:
                    print(f"Need {max(0, 4-len(table_points_3d))} more points!")

        else:
            if key == ord('q'):
                break
            elif key == ord('t'):
                table_mode = True
                print("Table mode! Click 4+ points on table surface!")
            elif key == ord('v'):
                roi_mode = True
                current_roi_name = ""
                print("ROI mode! Type name + Enter, then drag!")
            elif key == ord('c'):
                food_roi.clear()
                table_points_3d.clear()
                table_plane = None
                print("All cleared!")


# ============================================================
# [10] 종료 처리
# ============================================================
finally:
    pipeline.stop()
    cv2.destroyAllWindows()