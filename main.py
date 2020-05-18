import glob
import cv2
from random import shuffle
from pprint import pprint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from dataSets import TrainingDataSet

import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F

from utils import ContrastiveLoss, train, modelResize, validation

#definindo o path pra importar imagens
bananas_path = './frutas/bananas/*.jpg'
macas_path = './frutas/macas/*.jpg'

#Carregando os paths das imagens
all_path_bananas = glob.glob(bananas_path)
all_path_macas = glob.glob(macas_path)

#Definindo as labelsn banana = 0 e maca = 1
labels_banana = [0 for bananas in all_path_bananas]
labels_maca = [1 for macas in all_path_macas]

imagens_bananas = []
imagens_macas = []

#Loop para ler as imagens e adicionar nas listas 
for banana, maca in zip(all_path_bananas, all_path_macas):
    imgB = cv2.imread(banana, 1)
    imgB = cv2.resize(imgB, (250, 250))
    imagens_bananas.append(imgB)

    imgM = cv2.imread(maca, 1)
    imgM = cv2.resize(imgM, (250, 250))
    imagens_macas.append(imgM)

#Definindo as tuplas de imagem e label de cada atributo
tuplas_bananas = list(zip(imagens_bananas, labels_banana))
tuplas_macas = list(zip(imagens_macas, labels_maca))

#juntando as duas tuplas
all_tuplas = tuplas_bananas + tuplas_macas

#Embaralhando
shuffle(all_tuplas)

#separando entre images e labels novamente
imagens, labels = zip(*all_tuplas)

#Separando em treino e validacao
train_x, val_x, train_y, val_y = train_test_split(imagens, labels, test_size = 0.33)

train_data_set = TrainingDataSet(train_x, train_y)

train_data_loader = DataLoader(train_data_set, shuffle=True, num_workers=2, batch_size=4)

lr = 0.001
epoches = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = models.resnet50(pretrained=True)
resnet = modelResize(resnet, 2048, 2)
resnet = resnet.to(device)


#Train the newtwork
train(resnet, lr, train_data_loader, device, epoches)

#Validation
validation_data_set = TrainingDataSet(val_x, val_y)
validation__data_loader = DataLoader(validation_data_set, shuffle=True, num_workers=2, batch_size=4)

validation(resnet, validation__data_loader, device)

#Saving our model
PATH = './resnet.pth'
torch.save(resnet.state_dict(), PATH)

