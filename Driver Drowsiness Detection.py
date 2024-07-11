#!/usr/bin/env python
# 필요한 라이브러리 임포트
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
from Preprocess import preprocess

# dlib의 얼굴 감지기 및 랜드마크 예측기 초기화
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# 비디오 스트림 초기화 및 카메라 센서 워밍업
print("[INFO] initializing camera...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi 사용 시
time.sleep(2.0)

# 프레임 크기 설정
frame_width = 854
frame_height = 480

# 2D 이미지 포인트 초기화. 이미지가 변경되면 벡터도 변경 필요
image_points = np.array([
    (359, 391), # 코 끝
    (399, 561), # 턱
    (337, 297), # 왼쪽 눈의 왼쪽 모서리
    (513, 301), # 오른쪽 눈의 오른쪽 모서리
    (345, 465), # 왼쪽 입 모서리
    (453, 469), # 오른쪽 입 모서리
    (400, 390)  # 28번 랜드마크(미간)
], dtype="double")

# 눈과 입의 랜드마크 인덱스 가져오기
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)

# EAR, MAR 임계값 및 카운터 초기화
EYE_AR_THRESH_Close = 0.17
EYE_AR_THRESH_Drowsy = 0.30
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# 눈 감기 시작 시간 초기화
eye_closed_start_time = None

# 비디오 스트림에서 프레임을 반복적으로 읽어들임
while True:
    # 비디오 프레임 읽기, 크기 조정 및 그레이스케일 변환
    frame = vs.read()
    frame = imutils.resize(frame, width=854, height=480)
    frame = cv2.flip(frame, 1)  # 프레임 좌우 반전
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 프레임 전처리
    gray = preprocess(gray)
    
    size = gray.shape

    # 그레이스케일 프레임에서 얼굴 감지
    rects = detector(gray, 0)

    # 감지된 얼굴 수를 표시
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 감지된 각 얼굴에 대해 처리
    for rect in rects:
        # 얼굴의 경계 상자 계산 및 그리기
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(gray, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        
        # 얼굴 랜드마크 예측 및 numpy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 눈의 좌표 추출 및 EAR 계산
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # 눈 윤곽선 그리기
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(gray, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(gray, [rightEyeHull], -1, (0, 255, 0), 1)

        # EAR 값을 프레임에 표시
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 눈 깜빡임 및 졸림 상태 감지
        if ear < EYE_AR_THRESH_Close:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                eye_closed_duration = time.time() - eye_closed_start_time
                if eye_closed_duration >= 1.5:
                    cv2.putText(frame, "Drowsiness Driving", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif ear >= EYE_AR_THRESH_Close and ear < EYE_AR_THRESH_Drowsy:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Drowsy!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            eye_closed_start_time = None
        else:
            COUNTER = 0
            eye_closed_start_time = None

        # 입의 좌표 추출 및 MAR 계산
        mouth = shape[mStart:mEnd]
        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR

        # 입 윤곽선 그리기
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(gray, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 하품 상태 감지
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 얼굴 랜드마크 포인트 그리기 및 주요 랜드마크 업데이트
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                image_points[0] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                image_points[1] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 27:
                image_points[6] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                image_points[2] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                image_points[3] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                image_points[4] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                image_points[5] = np.array([x, y], dtype='double')
                cv2.circle(gray, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else: # 나머지 랜드마크는 빨간색으로 표시
                cv2.circle(gray, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(gray, str(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # 얼굴에 주요 이미지 포인트 그리기
        for p in image_points:
            cv2.circle(gray, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        # 고개 기울기 계산 및 표시
        (head_tilt_degree, start_point, end_point, end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)
        cv2.line(gray, start_point, end_point, (255, 0, 0), 2)
        cv2.line(gray, start_point, end_point_alt, (0, 0, 255), 2)
        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 프레임을 화면에 표시
    cv2.imshow("Frame", frame)
    cv2.imshow("Gray", gray)
    key = cv2.waitKey(1) & 0xFF

    # 'q' 키를 누르면 루프 종료
    if key == ord("q"):
        break

# 자원 정리
cv2.destroyAllWindows()
vs.stop()
