from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # 눈의 종횡비(eye aspect ratio, EAR)를 계산하는 함수

    # Parameters:
    #     eye (list): 눈의 랜드마크 좌표가 포함된 리스트

    # Returns:
    #     float: 계산된 눈의 종횡비 값
    
    # 눈의 상하 랜드마크 좌표 간의 유클리드 거리 계산
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 눈의 좌우 랜드마크 좌표 간의 유클리드 거리 계산
    C = dist.euclidean(eye[0], eye[3])
    # 눈의 종횡비 계산
    ear = (A + B) / (2.0 * C)
    # 눈의 종횡비 반환
    return ear
