import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import glob

from dataSets import TrainingDataSet

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def readingImages(bananas_path, macas_path):
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
    
    return imagens_bananas, imagens_macas, labels_banana, labels_maca


def creatingDataSets(imagens_banana, imagens_maca, labels_banana, labels_maca):
    images = np.concatenate((imagens_banana, imagens_maca))
    labels = np.concatenate((labels_banana, labels_maca))

    #Separando em treino e validacao
    train_x, val_x, train_y, val_y = train_test_split(images, labels, test_size = 0.33)

    train_data_set = TrainingDataSet(train_x, train_y)
    train_data_loader = DataLoader(train_data_set, shuffle=True, num_workers=2, batch_size=4)
    
    #Validation
    validation_data_set = TrainingDataSet(val_x, val_y)
    validation__data_loader = DataLoader(validation_data_set, shuffle=True, num_workers=2, batch_size=4)

    return train_data_loader, validation__data_loader


#Fuction to resize the network
def modelResize(model, input_num, output_num):
    model.fc = nn.Sequential(
            nn.Linear(
                in_features=input_num,
                out_features=output_num
            ),

            nn.Sigmoid()
        )
    return model


#Function for one shotting learning
def oneshot(output1, output2, label):
    total = 0
    corrects = 0

    distance = F.pairwise_distance(output1, output2).cpu()

    for j in range(distance.size()[0]):

        if ((distance.data.numpy()[j] < 0.8)):
            if label.cpu().data.numpy()[j] == 1:
                corrects += 1
                total += 1
                
            else:
                total += 1

        else:
            if label.cpu().data.numpy()[j] == 0:
                corrects += 1
                total += 1
            else:
                total +=1
                
    return [corrects, total]
        

