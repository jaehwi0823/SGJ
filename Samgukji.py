import numpy as np
import pandas as pd
# import pytorch as torch
import json

file = json.load(open('./textinthewild_data_info.json'))
file.keys() #dict_keys(['info', 'images', 'annotations', 'licenses'])
file['info'] #{'name': 'Text in the wild Dataset', 'date_created': '2019-10-14 04:31:48'}
type(file['images']) #list
