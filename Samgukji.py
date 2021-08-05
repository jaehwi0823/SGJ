# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import PIL
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
    print("file: ", filen)
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
        
        print(image.shape)
        return image, label
        

# %%
# './raw/한국어 글자체 이미지/02.인쇄체/syllable/'
rs = T.Resize((100, 100))
Hdata = HangulImageDataset(img_labels, './raw/한국어 글자체 이미지/02.인쇄체/syllable/', rs)
train_dataloader = DataLoader(Hdata, batch_size=64, shuffle=False)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# %% dataloader sample test
img = train_features[0].squeeze()
label = train_labels[0]
img = img.permute(1,2,0)
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")




# %%
# torchvision_transform = transforms.Compose([
#     transforms.Resize((256, 256)), 
#     transforms.RandomCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
# ])

# torchvision_dataset = TorchvisionDataset(
#     file_paths=["/content/gdrive/My Drive/test.png"],
#     labels=[1],
#     transform=torchvision_transform,
# )


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
