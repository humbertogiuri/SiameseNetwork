from torch.utils.data import Dataset
import random

#Class to define one training dataSet
class TrainingDataSet(Dataset):

    def __init__(self, train_data, labels):
        self.samples = train_data
        self.labels = labels
    

    def __len__(self):
        return len(self.labels)
    
    
    #Returns 2 images and whether they are of same class
    def __getitem__(self, index):
        image1 = self.samples[index]
        label1 = self.labels[index]

        #choice a random image
        index2 = random.randint(0, len(self.labels) - 1)
        image2 = self.samples[index2]
        label2 = self.labels[index2]

        #Compares whether the two images are from the same class
        iguals = 0

        if label1 == label2:
            iguals = 1 #Same class
        
        return image1, image2, iguals