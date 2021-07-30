# %%
import numpy as np
import pandas as pd
# import pytorch as torch
import json

# %%
base_path = './raw/한국어 글자체 이미지/02.인쇄체/'
file = json.load(open(base_path + 'printed_data_info.json'))
# /Users/jaehwi/Projects/SGJ/raw//
# %%
file.keys() #dict_keys(['info', 'images', 'annotations', 'licenses'])
file['info'] #{'name': 'Text in the wild Dataset', 'date_created': '2019-10-14 04:31:48'}
type(file['images']) #list


# 전처리 참조 링크: https://cvml.tistory.com/21

# %%
file['images'][:3]
file['annotations'][:3]


# %%
# %%
seoul_id = [d['image_id'] for d in file['annotations'] if d['attributes']['font'] == '서울한강']
# %%
seoul_id[:3]
# %%
seoul_img = [d for d in file['images'] if d['id'] in seoul_id]
# %%
seoul_img[:3]

# %%
import matplotlib.pyplot as plt
import cv2
img = cv2.imread('/data/ljh/dataset/ocr/Goods/'+goods[0]['file_name'])
plt.imshow(img)