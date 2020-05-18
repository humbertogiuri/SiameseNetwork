import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

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
        

#Function to train the network
def train(model, lr, data_loader, device, epoches = 100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_function = ContrastiveLoss()

    #Iterator over the epoches
    for i in range(epoches):

        print(f'Epoche {i + 1}/{epoches} Starting:')
        
        corrects = 0
        total = 0

        for i, data in enumerate(data_loader):

            image1, image2, label = data
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)

            #clear the calculated grad in previous batch
            optimizer.zero_grad()
            
            #Applying the view function to images
            image1 = image1.view(-1, 3, 250, 250).float()
            image2 = image2.view(-1, 3, 250, 250).float() 
            label  = label.float()

            #Get responses from the network
            output1 = model(image1)
            output2 = model(image2)
            
            #loss calculation
            loss = loss_function(output1, output2, label)
            loss.backward()

            #Optimizing
            optimizer.step()

            out = oneshot(output1, output2, label)
            corrects += out[0]
            total += out[1] 

        print(f'Loss in this epoch: {loss}')

        result = corrects * 100 / total
        print(f'Accuracy in this epoche: {result}%\n')

    print(f'Finished training!!!\n')

def validation(model, validation_data_loader, device):

    with torch.no_grad():
        
        corrects = 0
        total = 0

        for data in validation_data_loader:

            image1, image2, label = data
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)

            #Applying the view function to images
            image1 = image1.view(-1, 3, 250, 250).float()
            image2 = image2.view(-1, 3, 250, 250).float() 
            label  = label.float()
            
            output1 = model(image1)
            output2 = model(image2)
            
            out = oneshot(output1, output2, label)
            corrects += out[0]
            total += out[1]

        result = corrects * 100 / total
        print(f'Accuracy to validation data: {result}%\n')
    print('Finished Validation\n')