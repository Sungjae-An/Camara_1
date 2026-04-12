import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

# ============================================================
# [0] 전역 변수: 마우스 클릭으로 저장되는 안전영역 점들
# ============================================================
clicked_points = [] # 우선 빈 리스트를 만드는것. 변수는 1개 값만 담지만, 리스트는 여러개 값을 담을 수 있다.


# ============================================================
# [1] 마우스 콜백 함수
#     - 왼쪽 클릭: 점 추가
#     - 오른쪽 클릭: 마지막 점 제거
# ============================================================
def mouse_callback(event, x, y, flags, param): # def=나만의 함수를 만드는 명령. flags=shift/ctrl 같이 눌렀는지, param= 추가 데이터 전달용 (우린 안씀)
    global clicked_points # 함수 밖에 있는 clicked_points를 수정하겠다! python에선 원래 함수 안에서 함수 밖의 변수를 수정할 수 없기에 global 선언하는것.

    if event == cv2.EVENT_LBUTTONDOWN: # OpenCV는 마우스 동작마다 다른 숫자를 배정한다.
        clicked_points.append((x, y)) # append: 리스트에 항목을 뒤에 추가하는 명령.
        print(f"점 추가: ({x}, {y})") # f는 f-string: f를 붙이면 문자열 안에 변수를 {}안에 바로 넣으면 해당 변수값이 출력되게 함. .3f: 소수점 3자리까지만 표시

    elif event == cv2.EVENT_RBUTTONDOWN: # if 조건1 -> elif 조건2 -> elif 조건3 -> else (위 조건 전부 틀리면 실행)
        if clicked_points:
            removed = clicked_points.pop()
            print(f"점 제거: {removed}")

            # 특징은 if -> elif -> elif 순서대로 위에서부터 조건을 확인. 조건이 맞으면 거기서 멈추고 나머지 조건은 확인 안함. 순서가 중요!!


# ============================================================
# [2] 안정적인 깊이값 계산 함수
#     - 한 픽셀만 보면 노이즈가 많아서 주변 5x5 픽셀의 중간값을 사용
#     - depth_frame: 카메라에서 받은 깊이 이미지
#     - x, y: 측정하고 싶은 위치
#     - 반환값: 미터 단위 거리 (예: 0.452 = 45.2cm)
# ============================================================
def get_stable_depth(depth_frame, x, y, window_size=5): # 새 함수 선언. depth_frame: 카메라에서 받은 깊이 이미지 전체
    depths = []                 # 깊이값을 모으는 빈 리스트 생성
    half = window_size // 2     # //는 소수점을 버리는 나누기 (ex) 5//2 = 2)

    width = depth_frame.get_width()     # 이미지 크기 확인
    height = depth_frame.get_height()   # 이미지 크기 확인

    for dy in range(-half, half + 1):   # half=2 니깐, range (=2,3) = -2,-1,0,1,2
        for dx in range(-half, half + 1):
            px, py = x + dx, y + dy     # 이걸 통해 5x5=25개 픽셀을 모두 지정.

            # 이미지 경계 밖으로 나가면 건너뜀
            if px < 0 or py < 0 or px >= width or py >= height:
                continue

            d = depth_frame.get_distance(px, py)

            if d > 0:  # 0은 측정 실패를 의미하므로 제외
                depths.append(d)

    if len(depths) == 0:                # len=length, 리스트 항목 몇개인지 세는 함수. 
        return 0.0                      # 측정값이 하나도 없다면 returen 0.0

    return float(np.median(depths))     # np.median()은 numpy전용 숫자형식 -> float()로 파이썬 일반숫자로 변환. 중간값 반환 (튀는 값 영향 최소화)
                                        # return으로 계산 결과를 함수 밖으로 내보낸다 -> 이후 함수 밖에서 mouth_z=get_stable_depth(depth_frame,mouth_x,mouth_y)에서 받는다.

# ============================================================
# [3] MediaPipe 초기화
#     - Face Mesh: 얼굴의 468개 점을 인식 → 입 위치 추출에 사용
#     - Pose: 몸의 33개 관절을 인식 → 팔 위치 추출에 사용
#     두 분석기를 동시에 만들어두고, 매 프레임마다 둘 다 실행할 거예요
# ============================================================
mp_face_mesh = mp.solutions.face_mesh   # mediapipe함 (mp)에서 face_mesh 설계도를 꺼낸다.
face_mesh = mp_face_mesh.FaceMesh(      # face_mesh 설계도로 face_mesh 라는 인식기를 만들자.
    static_image_mode=False,            # False = 동영상 모드 (연속 추적) (True는 매 프레임을 독립된 사진으로 봐서 느리지만 정확)
    max_num_faces=1,                    # 얼굴 1개만 인식
    refine_landmarks=True               # 입술/눈 등 세밀한 점도 인식. (False는 기본 468개 점만 인식. True는 입술, 눈 주변 점이 더 추가됨)
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,                 # 0=빠름, 1=보통, 2=정확 → 1로 균형
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# [4] RealSense 카메라 설정 및 시작
#     - 색상 이미지(BGR): 화면에 보여줄 용도
#     - 깊이 이미지(Z16): 각 픽셀까지의 거리 측정용
#     - 둘 다 640x480, 30fps
# ============================================================
pipeline = rs.pipeline()                # Pipeline을 통해서 카메라 -> 파이썬 코드로 프레임들이 가는것.
config = rs.config()                    # 빈 설정지를 생성
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)     # 색상이미지를 가로640*세로480픽셀로, 색상형식은 파랑/초록/빨강 각 8비트(256가지)로, 초당 30프레임 달라
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      # 깊이 이미지를 거리값 16비트 (1m 거리를 65536칸으로 쪼개서 (1칸=0.015mm))로, 초당 30프레임 달라
pipeline.start(config)                  # config 설정대로 카메라 파이프라인 켜줘. 카메라를 켜야만 프레임을 받을 수 있다!!

window_name = "통합: 입 + 팔 + 안전영역"    
cv2.namedWindow(window_name)                        # window_name을 가진 창을 화면에 만들어줘.
cv2.setMouseCallback(window_name, mouse_callback)   # 이 창에서 마우스 이벤트가 생기면 mouse_callback을 실행해줘


# ============================================================
# [5] 메인 루프: 매 프레임마다 반복 실행
# ============================================================
try:                # try 뜻: 실행하다 오류나면 맨 마지막에 finally 코드는 꼭 실행해줘. 안그러면 에러났을때 pipeline.stop() 안돼서 카메라가 안꺼진채임.
    while True:     # True 조건이면 무한루프로 매 프레임 진행시켜

        # --- 5-1) 카메라에서 프레임 1번만 받기 ---
        # 이전에는 두 코드가 각각 프레임을 받았는데,
        # 이제 한 번만 받고 Face Mesh / Pose 둘 다에 전달!
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())  # BGR (OpenCV용), np.asanyarray를 통해 Realsense 전용 데이터를 NumPy 배열로 전환해야 openCV, mediapipe가 이미지 읽을 수 있다.
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)  # RGB (MediaPipe용), OpenCV는 BGR순서로 색을 읽지만, MediaPipe는 RGB로 읽는다.
        h, w, _ = color_image.shape     # _는 "버린다"는 파이썬 표현, .shape 로 (380, 640, 3)을 (480, 640, _)로 3개 변수에 저장함.
                                        # 같은 이미지를 openCV용 (화면표시) BGR로 하나, MediaPipe용 (얼굴/관절인식) RGB로 하나 만드는 것이다.

        # --- 5-2) 안전영역 다각형 그리기 ---
        # 3개 이상의 점이 있을 때만 다각형을 채워서 표시
        # 먼저 노란색 칠하는 layer를 만들고, 그 위에 점, 선, 번호 layer를 덮어야 노란색으로 점/선이 가려지지 않음.
        if len(clicked_points) >= 3:
            overlay = color_image.copy()                                # 원본 이미지를 복사해서 overlay라는 사본 만들기
            polygon = np.array(clicked_points, dtype=np.int32)          # dtype=np.int32를 통해, 숫자를 정수(int)로 저장.
            cv2.fillPoly(overlay, [polygon], (0, 255, 255))             # 노란색으로 채우기
            color_image = cv2.addWeighted(overlay, 0.25, color_image, 0.75, 0)  # 25% 투명도, 맨 끝의 0은 밝기보정을 안한다는 뜻.

        # 클릭한 점들과 연결선 표시
        for i, (x, y) in enumerate(clicked_points):                     # enumerate로 인덱스 (순서번호, i)를 같이 줘라.
            cv2.circle(color_image, (x, y), 5, (0, 0, 255), -1)         # (x,y)에 반지름 5픽셀 원을 빨간색(BGR)으로, 원 안을 꽉 채워서 (-1) 그려줘
            cv2.putText(color_image, str(i), (x + 8, y - 8),            # str(i)를 통해 숫자를 문자로 변환.
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 맨 마지막 1은 글씨 두께.

        for i in range(len(clicked_points) - 1):                        # -1 을 해서 마지막 점은 다음 점이 없으니깐 그 전가지만 연결해라.
            cv2.line(color_image, clicked_points[i], clicked_points[i + 1], (255, 0, 0), 2)

        if len(clicked_points) >= 3:                                    # clicked_points[-1]에서 -1은 리스트 맨 마지막 점.
            cv2.line(color_image, clicked_points[-1], clicked_points[0], (0, 255, 255), 2)


        # --- 5-3) Face Mesh로 입 위치 인식 ---
        # rgb_image를 Face Mesh에 넣으면 468개 얼굴 점을 돌려줌
        # 그 중 13번(윗입술), 14번(아랫입술)의 중간점 = 입 중심
        face_results = face_mesh.process(rgb_image)     # rgb_image를 넣어서 결과를 face_results에 저장
        mouth_x, mouth_y, mouth_z = None, None, None    # 초기화

        if face_results.multi_face_landmarks:                      # if 가 있어야 인식이 실패할때 FaceMesh 코드 블록을 건너뛰고 에러 안난다.
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
            cv2.putText(color_image, f"Mouth z={mouth_z:.3f}m", (10, 30),       # mouth_z:.3f 로 소수점 3자리까지 표시
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # --- 5-4) Pose로 팔 관절 인식 ---
        # rgb_image를 Pose에 넣으면 33개 관절 점을 돌려줌
        # 오른쪽 어깨(RIGHT_SHOULDER), 팔꿈치(RIGHT_ELBOW), 손목(RIGHT_WRIST) 사용
        pose_results = pose.process(rgb_image)
        wrist_x, wrist_y = None, None  # 나중에 안전영역 판정에 쓸 변수

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark        # .landmark를 붙여서 33개 관절 목록을 꺼낸다.

            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist    = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # 비율값 → 픽셀 변환 (화면 밖으로 나가지 않도록 clamp)
            shoulder_x = max(0, min(w-1, int(shoulder.x * w)))     # Pose는 FaceMesh와 달리 clamp (값이 범위 못벗어나게)를 추가함. 팔이 화면 밖으로 나갈 수 있으므로.
            shoulder_y = max(0, min(h-1, int(shoulder.y * h)))     # shoulder_y는 최대 h-1값을 가질 수 있구나
            elbow_x    = max(0, min(w-1, int(elbow.x * w)))
            elbow_y    = max(0, min(h-1, int(elbow.y * h)))
            wrist_x    = max(0, min(w-1, int(wrist.x * w)))
            wrist_y    = max(0, min(h-1, int(wrist.y * h)))

            # 깊이값 측정
            shoulder_z = get_stable_depth(depth_frame, shoulder_x, shoulder_y, window_size=15)
            elbow_z    = get_stable_depth(depth_frame, elbow_x, elbow_y, window_size=15)
            wrist_z    = get_stable_depth(depth_frame, wrist_x, wrist_y, window_size=15)

            # 관절 점과 선 표시
            cv2.circle(color_image, (shoulder_x, shoulder_y), 8, (255, 0, 0), -1)
            cv2.circle(color_image, (elbow_x, elbow_y), 8, (0, 255, 0), -1)
            cv2.circle(color_image, (wrist_x, wrist_y), 8, (0, 0, 255), -1)
            cv2.line(color_image, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (255, 255, 0), 2)   # cv2.line의 맨 끝 2는 선 두께
            cv2.line(color_image, (elbow_x, elbow_y), (wrist_x, wrist_y), (255, 255, 0), 2)

            # 관절 이름 표시
            cv2.putText(color_image, f"Shoulder z={shoulder_z:.3f}m", (10, 60),             # (10,60): 화면왼쪽에서 10픽셀, 위에서 60픽셀 위치
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(color_image, f"Elbow z={elbow_z:.3f}m", (10, 90),                   # (10,60)보다 1줄 아래가 (10,90). 30픽셀간격
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

        key = cv2.waitKey(1) & 0xFF         # 매 프레임마다 1 miㅣlisec동안 키보드 입력을 기다려. 0이면 키 입력까지 무한히 기다려서 화면이 멈춰버림.
                                            # 0xFF: WaitKey의 운영체제 호환성때문. 윈도우는 8비트로 0x71 이어도 리눅스는 0xFFFFFF71 이렇게 32비트 숫자를 반환함.
        if key == ord('q'):                 # ord()는 글자를 숫자로 변환하는 함수. ord('q')=113
            break
        elif key == ord('c'):
            clicked_points.clear()          # .clear()는 .append()의 반대
            print("안전영역 초기화")

# ============================================================
# [6] 종료 처리
#     - 오류가 나도 반드시 카메라와 창을 닫아야 함
#     - finally 블록은 어떤 상황에서도 실행됨
# ============================================================
finally:
    pipeline.stop()
    cv2.destroyAllWindows()