# %%
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cv2
import os
import PIL
import time
import copy
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, models
import torch.optim as optim
from torch.optim import lr_scheduler


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
    transforms.Resize((128, 128)),
    RandomColorShift(),
    transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5, 
        saturation=0.5, 
        hue=0.5
    ),
    # transforms.ToTensor(),
])

Hdata = HangulImageDataset(img_labels, './raw/한국어 글자체 이미지/02.인쇄체/syllable/', 
                           transform=torchvision_transform)
train_dataloader = DataLoader(Hdata, batch_size=64, shuffle=False)
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")

# dataloader sample test
# sample = train_features[0].squeeze()
# label = train_labels[0]

# sample = sample.permute(1,2,0)
# plt.imshow(sample, cmap="gray")
# plt.show()
# print(f"Label: {label}")


# %% pre-trained model
resnet18_pretrained = models.resnet18(pretrained=True)
num_classes = 10638
num_ftrs = resnet18_pretrained.fc.in_features
resnet18_pretrained.fc = nn.Sequential(
                            nn.Linear(num_ftrs, 4096),
                            nn.ReLU(),
                            nn.Linear(4096, 10638),
                         )
device = torch.device('cpu')
resnet18_pretrained.to(device)


# %%
from torchsummary import summary
summary(resnet18_pretrained, input_size=(3, 128, 128), device=device.type)
# %%


# Loss Function
criterion = nn.CrossEntropyLoss()
# 모든 매개변수들이 최적화되었는지 관찰
optimizer_ft = optim.SGD(resnet18_pretrained.parameters(), lr=0.001, momentum=0.9)
# 7 에폭마다 0.1씩 학습률 감소
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # 모델을 학습 모드로 설정

        running_loss = 0.0
        running_corrects = 0

        # 데이터를 반복
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 매개변수 경사도를 0으로 설정
            optimizer.zero_grad()

            # 순전파
            # 학습 시에만 연산 기록을 추적
            # with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # update
            loss.backward()
            optimizer.step()

            # 통계
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / num_classes
        epoch_acc = running_corrects.double() / num_classes
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "train", epoch_loss, epoch_acc))
        scheduler.step()

        # 모델을 깊은 복사(deep copy)함
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

# %%
model_ft = train_model(resnet18_pretrained, criterion, optimizer_ft, 
                       exp_lr_scheduler, num_epochs=25)

ip, lb = next(iter(train_dataloader))
type(lb)

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터를 반복
            for inputs, labels in train_dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)
    return model

