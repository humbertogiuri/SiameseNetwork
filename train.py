import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from utils import ContrastiveLoss, oneshot

#Function to train the network
def train(model, lr, data_loader, device, epoches = 100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_function = ContrastiveLoss()

    best_acc = 0.0

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

        #Computing acc
        current_acc = corrects * 100 / total
        print(f'Accuracy in this epoche: {current_acc}%')

        #Save the network if the acc is better
        if current_acc > best_acc:
            best_acc = current_acc
            PATH = './resnet.pth'
            torch.save(model.state_dict(), PATH)

            print('Model Save!!')

        print('\n')
    
    print(f'Finished training!!!\n')


def validation(model, validation_data_loader, device):
    model.to(device)
    
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