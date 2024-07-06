from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    # 입의 종횡비(mouth aspect ratio, MAR)를 계산하는 함수

    # Parameters:
    #     mouth (list): 입의 랜드마크 좌표가 포함된 리스트

    # Returns:
    #     float: 계산된 입의 종횡비 값

    # 입의 상하 랜드마크 좌표 간의 유클리드 거리 계산
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57

    # 입의 좌우 랜드마크 좌표 간의 유클리드 거리 계산
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55

    # 입의 종횡비 계산
    mar = (A + B) / (2.0 * C)

    # 입의 종횡비 반환
    return mar
