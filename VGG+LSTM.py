import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

# to play the audio files
from IPython.display import Audio

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


import torch
import librosa
import cv2 as cv
from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel
import torch.nn as nn
from tqdm import tqdm
from top_1_top_5 import confusion_matrix
import random
import subprocess



def max_min(img):
    min1 = np.min(img)
    max1 = np.max(img)

    return (img - min1) / (max1 - min1)

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape)
    return data

def extract_audio_from_video(video_path):
    # 使用FFmpeg从视频中提取音频并以原始格式输出到stdout
    cmd = ['ffmpeg', '-i', video_path, '-f', 's16le', '-ac', '1', '-ar', '44100', '-']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    # 循环读取音频数据流
    audio_data = b''
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        audio_data += chunk

    # 关闭FFmpeg进程
    process.stdout.close()
    process.wait()

    return np.frombuffer(audio_data, dtype=np.int16)


class class_img_data(Dataset):

    def __init__(self, file_name):
        #         self.path = ".\\"   self.path +
        self.train_data = pd.read_csv(".\\" + str(file_name) + ".csv")
        self.train_data.columns = ["lab", "Path"]
        self.img_path_l = self.train_data["Path"]
        self.img_label = self.train_data["lab"]

    def __len__(self):
        return len(self.train_data["Path"])

    def __getitem__(self, idx):
        image_path = self.img_path_l[idx]
        label = self.img_label[idx]
        data, fs = librosa.load(image_path, sr=None, mono=False)

        if_add_noise = random.random()


        #add noise
        if if_add_noise > 0.7:

            data = noise(data)
        else:
            pass

        D = np.abs(librosa.stft(data)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=fs)



        image = librosa.power_to_db(S, ref=np.max)
        # deltas = librosa.feature.delta(log_mel_spec)
        # delta_deltas = librosa.feature.delta(log_mel_spec, order=2)

        # if_deltas_delta = random.random()
        #
        #
        # #add deltas
        # if if_deltas_delta > 0.6:
        #     if if_deltas_delta > 0.8:
        #         image = librosa.feature.delta(image)
        #     else:
        #         if if_deltas_delta > 0.6:
        #
        #             image = librosa.feature.delta(image, order=2)
        #
        #         else:
        #             pass
        # else:
        #     pass
        #

        try:
            width, height = image.shape
        except:
            channel,width, height = image.shape
            image = image[0]



        target_size = 128
        start_x = (width - target_size) // 2
        start_y = (height - target_size) // 2
        end_x = start_x + target_size
        end_y = start_y + target_size

        image = cv.resize(image[start_x:end_x, start_y:end_y], (target_size, target_size),
                          interpolation=cv.INTER_CUBIC)


        #         image = cv.resize(image,(64,64))
        image = max_min(image)
        image = image.reshape((128, 128, 1))

        #         image = image.astype(np.float32)/255.0
        tensor_image = torch.Tensor(image)
        tensor_image = tensor_image.transpose(0, 2)




        return tensor_image, label


train_dataset = class_img_data(file_name="train")
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

vaild_dataset = class_img_data(file_name="val")
vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=10, shuffle=True)

vaild_dataset = class_img_data(file_name="test")
vaild_loader = torch.utils.data.DataLoader(vaild_dataset, batch_size=10, shuffle=False)


class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ELU()
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.reshape = nn.Flatten()

        self.lstm = nn.LSTM(input_size=128, hidden_size=32, batch_first=True)

        self.fc = nn.Linear(32, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.reshape(x)

        x, _ = self.lstm(x)
        # x = x[:, -1, :]

        x = self.fc(x)
        x = self.softmax(x)

        return x

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")


model = baseline()#.to(device)
model = DataParallel(model,device_ids = [0])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

n_total_steps = len(data_loader)

num_epoch = 100

for epoch in range(num_epoch):

    epoch_acc = []
    epoch_loss = []

    for i, (batch, labels) in tqdm(enumerate(data_loader)):

        batch = batch.clone().detach().to(torch.float32).to("cuda")
        labels = labels.to('cuda')

        pred_y = model(batch)
        _, predicted = torch.max(pred_y, 1)

        loss = criterion(pred_y, labels)

        if np.isnan(loss.to("cpu").item()):
            print("loss",loss.to("cpu").item())

        loss.backward()

        acc = np.diag(confusion_matrix(predicted.to("cpu").numpy(), labels.to("cpu").numpy())).sum() / labels.shape[0]

        epoch_acc.append(acc)
        epoch_loss.append(loss)

        optimizer.step()
        optimizer.zero_grad()

        if i % 2 == 0:
            print('epoch: %d batch: %d acc: %.2f loss: %.4f' % (epoch, i, acc, loss))

    L = sum(epoch_loss) / i
    A = sum(epoch_acc) / i


    # break


    with torch.no_grad():

        model.eval()
        for i, (v_batch, v_labels) in enumerate(data_loader):

            v_batch = v_batch.clone().detach().to(torch.float32).to("cuda")
            v_labels = v_labels.to("cuda")

            vail_y = model(v_batch)
            _, predicted = torch.max(vail_y, 1)
            v_acc = np.diag(confusion_matrix(predicted.to("cpu").numpy(), v_labels.to("cpu").numpy())).sum() / v_labels.shape[0]

            v_loss = criterion(vail_y, v_labels)
            break

    path = 'epoch_result_th.csv'
    with open(path, 'a') as f:
        f.write('{0},{1},{2},{3}'.format(round(L.to("cpu").item(),4), round(A,4), round(v_loss.to("cpu").item(),4),round(v_acc,4)))
    f.close()

    if epoch == 3 :
        folder_name = "models_th"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        torch.save(model, os.path.join(folder_name, "epoch_{0}_model.pth".format(epoch)))







