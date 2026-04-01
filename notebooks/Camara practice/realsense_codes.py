# pyrealsense 코드 이해

pipeline = 카메라 데이터 흐름 시작
config = 어떤 스트림 (=영상)을 받을지 설정
frames = 한 시점의 컬러/깊이 묶음
depth_frame.get_distance(x, y) = 특정 좌표의 거리(m)

점 색깔 바꾸기:
cv2.circle(color_image, (x, y), 5, (255, 0, 0), -1)

점 크기 바꾸기:
cv2.circle(color_image, (x, y), 8, (0, 0, 255), -1)

c키 눌러서 모든 점 지우기
if key == ord('q'):
    break
elif key == ord('c'):
    clicked_points.clear()
    print("All points cleared")

# Numpy

numpy 배열 = 숫자가 들어있는 엑셀시트
각 영상은 세로 픽셀수 * 가로 픽셀수 * 색상 (B,G,R) 이기때문.

print(color_image[200][300]) # 200행 300열 픽셀의 B,G,R 값을 출력해라.

Depth도 마찬가지다. Depth는 색이 없이 거리값만 있는 표.

나중에 다음과 같은 작업이 필요하다:

작업	             NumPy 사용
입 주변 평균 거리	  배열 평균
테이블 평면 찾기	  행렬 계산
팔 각도 계산	      벡터 계산
충돌 거리 계산	      거리 계산
AI 학습 데이터	      배열