import os
import shutil
import glob
from tqdm import tqdm

"""
subject ID : xxx

image number : xxx

gender : 0 - male, 1 - famale

glasses : 0 - no, 1 - yes

eye state : 0 - close, 1 - open

reflections : 0 - none, 1 - low, 2 - high

lighting conditions / image quality : 0 - bad, 1 - good

sensor type : 01 - RealSense SR300 640x480, 02 - IDS Imaging, 1280x1024, 03 - Aptina Imagin 752x480

example : s001_00123_0_0_0_0_0_01.png

"""

# 데이터 클리닝을 수행합니다. 

Raw_DIR= r'C:\_AppleBanana\dataset'
for dirpath, dirname, filenames in os.walk(Raw_DIR):
    for i in tqdm([f for f in filenames if f.endswith('.jpg')]):
        if i.split('_')[3]=='E01':
            shutil.copy(src=dirpath+'/'+i, 
                        dst=r'C:\_AppleBanana\dataset\Open Eyes')
        
        elif i.split('_')[3]=='E03':
            shutil.copy(src=dirpath+'/'+i, 
                        dst=r'C:\_AppleBanana\dataset\Close Eyes')

