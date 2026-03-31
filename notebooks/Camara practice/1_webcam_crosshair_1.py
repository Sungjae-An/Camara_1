import cv2

# 웹캠 열기
cap = cv2.VideoCapture(1) #Realsense D455 카메라번호가 "2"

# 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 1장 읽기
    ret, frame = cap.read()

    # 프레임을 못 읽으면 종료
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 화면 크기 구하기
    height, width, _ = frame.shape

    # 화면 중앙 좌표
    center_x = width // 2
    center_y = height // 2

    # 중앙에 십자선 그리기
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

    # 중앙 좌표를 글자로 표시
    text = f"Center: ({center_x}, {center_y})"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("Webcam with Crosshair", frame)

    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()