# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
# import pytorch as torch

# %%
base_path = './raw/한국어 글자체 이미지/02.인쇄체/'
file = json.load(open(base_path + 'printed_data_info.json'))

# %%
file.keys() #dict_keys(['info', 'images', 'annotations', 'licenses'])
file['info'] #{'name': 'Text in the wild Dataset', 'date_created': '2019-10-14 04:31:48'}
type(file['images']) #list

# 전처리 참조 링크: https://cvml.tistory.com/21
seoul_id = [d['image_id'] for d in file['annotations'] if d['attributes']['font'] == '서울한강']
seoul_img = [d for d in file['images'] if d['id'] in seoul_id]
seoul_info = [d for d in file['annotations'] if d['attributes']['font'] == '서울한강']

# %%
def show_char(num):
    try:
        img = cv2.imread(base_path+'syllable/'+seoul_img[num]['file_name'])
        plt.imshow(img)
    except:
        img = cv2.imread(base_path+'word/'+seoul_img[num]['file_name'])
        plt.imshow(img)
# %% 마지막 단어는 16431번째 단어 인류
show_char(16430)
seoul_img = seoul_img[:16431]

# %% 마지막 글자는 10638번째 꿩
show_char(10637)

# %%
