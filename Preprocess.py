import cv2

# 적응적 임계값 처리는 이미지의 각 부분에 대해 다른 임계값을 적용하여 조명 변화에 더 잘 적응할 수 있습니다.
def adaptive_thresholding(frame):
    # 가우시안 블러링 적용
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # 적응적 이진화 적용
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    return thresh

# 히스토그램 균일화는 이미지의 대비를 향상시켜 세부 정보를 뚜렷하게 만들어줍니다. 이는 조명이 부족한 경우에 유용할 수 있습니다.
def histogram_equalization(frame):
    # 히스토그램 균일화 적용
    equalized = cv2.equalizeHist(frame)

    return equalized

def brightness_correction_up(frame):
    # 밝기 보정
    brightness_factor = 1.2  # 밝기를 조절할 상수 (1.0보다 크면 밝아지고, 작으면 어두워짐)
    brightened_image = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    return brightened_image

def brightness_correction_down(frame):
    # 밝기 보정
    brightness_factor = 0.8  # 밝기를 조절할 상수 (1.0보다 크면 밝아지고, 작으면 어두워짐)
    brightened_image = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

    return brightened_image

def preprocess(frame):
    # 여기에 전처리 작업 추가
    # 예시: 색상 보정, 적응적 임계값 처리, 히스토그램 균일화 등

    # 프레임 전처리
    preprocessed_histogram_equalization = histogram_equalization(frame)
    preprocessed_brightness_correction_down = brightness_correction_down(preprocessed_histogram_equalization)

    # 색상 변환 (BGR에서 RGB로)
    frame_rgb = cv2.cvtColor(preprocessed_brightness_correction_down, cv2.COLOR_BGR2RGB)
    
    # 전처리된 이미지 반환
    return frame_rgb