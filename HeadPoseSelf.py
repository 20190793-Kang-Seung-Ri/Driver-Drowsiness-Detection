import numpy as np
import math
import cv2

# 3D 모델의 점들을 정의합니다.
model_points = np.array([
    (0.0, 0.0, 0.0),          # 코 끝점 34
    (0.0, -330.0, -65.0),     # 턱 9
    (-225.0, 170.0, -135.0),  # 왼쪽 눈 왼쪽 모서리 37
    (225.0, 170.0, -135.0),   # 오른쪽 눈 오른쪽 모서리 46
    (-150.0, -150.0, -125.0), # 왼쪽 입 모서리 49
    (150.0, -150.0, -125.0),  # 오른쪽 입 모서리 55
    (0, 0, 100)               # 28번 랜드마크(미간)
])

def calculateSlope(point1, point2):
    x_diff = point2[0] - point1[0]
    y_diff = point2[1] - point1[1]
    
    # 기울기 계산
    slope_rad = math.atan2(y_diff, x_diff)
    if slope_rad < 0:
        slope_rad += math.pi  # 음수일 경우 양수로 변환
    
    slope_deg = np.degrees(slope_rad)  # 라디안을 각도로 변환
    
    # 기울기를 0에서 180도 범위로 맞춤
    if slope_deg > 180:
        slope_deg = 360 - slope_deg
    
    return slope_deg

# 회전 행렬이 유효한지 확인합니다.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# 회전 행렬을 오일러 각도로 변환합니다.
# 결과는 MATLAB과 동일하지만 오일러 각도의 순서는 다릅니다 (x와 z가 교환됩니다).
def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def getHeadTiltAndCoords(size, image_points, frame_height):
    # 카메라 특성 설정
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # 렌즈 왜곡 없음으로 가정합니다.

    # solvePnP를 사용하여 회전 및 변환 벡터 계산
    (_, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points,
                                                             camera_matrix, dist_coeffs,
                                                             flags=cv2.SOLVEPNP_ITERATIVE)

    # 코 끝점을 3D 좌표로 변환합니다.
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),
                                               rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # 회전 벡터에서 회전 행렬을 얻습니다.
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # 머리 기울기 각도 계산
    head_tilt_degree = abs([-180] - np.rad2deg([rotationMatrixToEulerAngles(rotation_matrix)[0]]))

    # 머리 기울기에 따른 시작 및 끝점 계산
    # starting_point = (int(image_points[0][0]), int(image_points[0][1]))
    # ending_point = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    # ending_point_alternate = (ending_point[0], frame_height // 2)
    starting_point = (int(image_points[6][0]), int(image_points[6][1]))
    ending_point = (int(image_points[1][0]), int(image_points[1][1]))
    ending_point_alternate = (ending_point[0], frame_height // 2)

    # 미간이랑 턱 기울기 계산
    slope = calculateSlope(image_points[6], image_points[1])

    return slope, starting_point, ending_point, ending_point_alternate
