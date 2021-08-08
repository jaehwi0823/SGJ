# %%
import numpy as np
from numpy import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import PIL
from torchvision import transforms
import torchvision.transforms as T
import torch

# %%
base_path = './raw/한국어 글자체 이미지/02.인쇄체/'
file = json.load(open(base_path + 'printed_data_info.json'))
df_images = pd.DataFrame(file['images'])
annotations = [{'image_id':img['image_id'], 
                'text':img['text'], 
                'font':img['attributes']['font'],
                'type':img['attributes']['type'],
                'is_aug':img['attributes']['is_aug']} for img in file['annotations']]
df_annotations = pd.DataFrame(annotations)

df_all_info = pd.merge(df_annotations, df_images, left_on='image_id', right_on='id', how='left')
seoul = df_all_info[(df_all_info.font == '서울한강') & (df_all_info.type == '글자(음절)')]


# %%
def show_char(filen):
    # print("file: ", filen)
    try:
        # print("syllable!")
        img = cv2.imread(base_path+'syllable/'+filen)
    except:
        # print("word!")
        img = cv2.imread(base_path+'word/'+filen)
    plt.imshow(img)

# test
# show_char(seoul.iloc[375, 8])
show_char(df_all_info['file_name'][3])


# %% file check: id 기준 10637까지
for i in range(len(seoul)):
    try:
        show_char(seoul.iloc[i, 8])
    except:
        print("error id: ", i)
        break

# %% 데이터가 있는 만큼만
seoul = seoul[:10638]
labels = list(zip(seoul.file_name, seoul.text))
img_labels = pd.DataFrame(labels)
img_labels.set_axis(labels=['filename','character'], axis=1, inplace=True)
# 이미지가 한 글자에 하나씩밖에 없음.. augmentation 필요
any(img_labels.groupby('character').filename.nunique() > 1)

# %% Custom torch Dataset
class HangulImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.tt = transforms.ToTensor()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = PIL.Image.open(img_path)
        image = self.tt(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        # print(image.shape)
        return image, label


class RandomColorShift(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        # print(sample.shape)
        # img, landmarks = sample['image'], sample['landmarks']

        r = torch.rand((3,))
        sample[0] = sample[0] + r[0]/2
        sample[1] = sample[1] + r[1]/2
        sample[2] = sample[2] + r[2]/2

        return sample


# %%
torchvision_transform = transforms.Compose([
    transforms.Resize((50, 50)),
    RandomColorShift(),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5, 
        saturation=0.5, 
        hue=0.5
    ),
])

Hdata = HangulImageDataset(img_labels, './raw/한국어 글자체 이미지/02.인쇄체/syllable/', 
                           transform=torchvision_transform)
train_dataloader = DataLoader(Hdata, batch_size=64, shuffle=False)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# dataloader sample test
sample = train_features[0].squeeze()
label = train_labels[0]

sample = sample.permute(1,2,0)
plt.imshow(sample, cmap="gray")
plt.show()
print(f"Label: {label}")






# %%
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100*100, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to('cpu')
print(model)


# %%
