import os
import glob
import shutil
import random
from tqdm import tqdm

# 졸음 데이터 세트를 로드합니다.

raw_data = '/home/delixus/Desktop/drowsiness_detection/mrlEyes_2018_01'
for dirpath, dirname, filename in os.walk(raw_data):
    for file in tqdm([f for f in filename if f.endswith('.png')]):
        if file.split('_')[4] == '0':
            path='/home/delixus/Desktop/drowsiness_detection/data/train/closed'
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy(src=dirpath + '/' + file, dst= path)
        elif file.split('_')[4] == '1':
            path='/home/delixus/Desktop/drowsiness_detection/data/train/open'
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy(src=dirpath + '/' + file, dst= path)        

# 졸음 감지를 위한 특정 전처리 단계를 수행합니다.

def create_test_closed(source, destination, percent):
    '''
    divides closed eyes images into given percent and moves from
    source to destination.
    
    Arguments:
    source(path): path of source directory
    destination(path): path of destination directory
    percent(float): percent of data to be divided(range: 0 to 1)
    '''
    path, dirs, files_closed = next(os.walk(source))
    file_count_closed = len(files_closed)
    percentage = file_count_closed * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')    

# 센서 또는 이미지 데이터를 정규화합니다.

def create_test_open(source, destination, percent):
    '''
    divides open eyes images into given percent and moves from
    source to destination.
    
    Arguments:
    source(path): path of source directory
    destination(path): path of destination directory
    percent(float): percent of data to be divided(range: 0 to 1)
    '''
    path, dirs, files_open = next(os.walk(source))
    file_count_open = len(files_open)
    percentage = file_count_open * percent
    to_move = random.sample(glob.glob(source + "/*.png"), int(percentage))

    for f in enumerate(to_move):
        if not os.path.exists(destination):
            os.makedirs(destination)
        shutil.move(f[1], destination)
    print(f'moved {int(percentage)} images to the destination successfully.')

# 졸음 감지와 관련된 엔지니어 기능을 제공합니다.

create_test_closed('/home/delixus/Desktop/drowsiness_detection/data/train/closed', 
                    '/home/delixus/Desktop/drowsiness_detection/data/test/closed', 
                    0.2)

# 준비된 졸음 데이터를 저장합니다.

create_test_open('/home/delixus/Desktop/drowsiness_detection/data/train/open', 
                    '/home/delixus/Desktop/drowsiness_detection/data/test/open', 
                    0.2)

