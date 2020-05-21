import numpy as np
import pandas as pd

import torch
import torchvision.models as models

from utils import modelResize, readingImages, creatingDataSets
from train import train, validation

#definindo o path pra importar imagens
bananas_path = './frutas/bananas/*.jpg'
macas_path = './frutas/macas/*.jpg'


lr = 0.001
epoches = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet = models.resnet50(pretrained=True)
resnet = modelResize(resnet, 2048, 2)
resnet = resnet.to(device)

images_banana, images_maca, labels_banana, labels_maca = readingImages(bananas_path, macas_path)

train_data_loader, validation_data_loader = creatingDataSets(images_banana, images_maca, labels_banana, labels_maca)

#Train the newtwork
train(resnet, lr, train_data_loader, device, epoches)

#Validation
resnet = resnet = models.resnet50(pretrained=True)
resnet = modelResize(resnet, 2048, 2)
resnet.load_state_dict(torch.load('./resnet.pth'))

validation(resnet, validation_data_loader, device)

    